from contextlib import nullcontext
from typing import Optional, Dict

from nanorlhf.nanoray.core.actor import ActorRef
from nanorlhf.nanoray.core.object_ref import ObjectRef
from nanorlhf.nanoray.core.object_store import ObjectStore
from nanorlhf.nanoray.core.task import Task
from nanorlhf.nanoray.runtime.process_pool import ProcessPool
from nanorlhf.nanoray.utils import new_actor_id


def _invoke(fn, args, kwargs):
    """
    Top-level helper for process-based execution

    Notes:
        - Must be at module top-level so that it is picklable by `multiprocessing`.
        - Mirrors the core execution: `return fn(*args, **kwargs)`.
    """
    return fn(*args, **(kwargs or {}))


class Worker:
    """
    Minimal worker that executes `Task`s, stores results into an `ObjectStore`,
    and returns `ObjectRef`s to the caller.

    The worker can run regular function tasks in the current process (default)
    or submit them to a process pool for CPU-bound or isolation-friendly execution.

    **ActorCreate/ActorCall are always executed in-process** because actor instances
    live in this worker's memory.

    Args:
        store (ObjectStore): The node-local object store.
        pool (Optional[ProcessPool]): Optional process-based executor.

    Examples:
       >>> # Local (in-process) function execution
        >>> from nanorlhf.nanoray.core.task import Task
        >>> from nanorlhf.nanoray.core.object_store import ObjectStore
        >>> def add(x, y): return x + y
        >>> store = ObjectStore("node-A")
        >>> w = Worker(store=store, node_id="A")
        >>> task = Task.from_call(add, (3, 4))
        >>> ref = w.execute_task(task)
        >>> store.get(ref)
        7

        >>> # With a process pool (regular function tasks only)
        >>> from nanorlhf.nanoray.runtime.process_pool import ProcessPool
        >>> pool = ProcessPool(max_workers=2)
        >>> w2 = Worker(store=store, node_id="A", pool=pool)
        >>> ref2 = w2.execute_task(task)  # executed in a separate process
        >>> store.get(ref2)
        7
        >>> pool.shutdown()

        >>> # Actors: creation and method calls are *always* in-process
        >>> from nanorlhf.nanoray.api.remote import actor
        >>> from nanorlhf.nanoray.api.session import get
        >>> @actor
        ... class Counter:
        ...     def __init__(self): self.x = 0
        ...     def inc(self, n=1): self.x += n; return self.x
        >>> # ActorCreate path
        >>> h_ref = w.execute_task(Task(fn=ActorCreate(Counter, (), {})))
        >>> handle = store.get(h_ref) if hasattr(store, "get") else get(h_ref)
        >>> # ActorCall path
        >>> r_ref = w.execute_task(Task(fn=ActorCall(handle.actor_id, "inc"), args=(2,)))
        >>> store.get(r_ref)
        2
    """

    def __init__(self, store: ObjectStore, node_id: Optional[str] = None, pool: Optional[ProcessPool] = None):
        self.store = store
        self.node_id = node_id or store.node_id
        self.pool = pool
        self._actors: Dict[str, object] = {}  # local actor registry

    def execute_task(self, task: Task) -> ObjectRef:
        """
        Execute the given `Task` and return an `ObjectRef` to the result.

        Args:
            task (Task): Declarative description of a remote function call.

        Returns:
            ObjectRef: Handle to the value produced by `task.fn(*task.args, **task.kwargs)`.

        Notes:
            - Ownership: the produced object is stored on this worker's store, so the
              returned `ObjectRef.owner_node_id` will be `store.node_id`.

        Discussion:
            Q. Which code paths exist?
                1) ActorCreate   -> instantiate the actor locally and return `ActorRef`
                2) ActorCall     -> lookup instance locally and invoke method
                3) Regular call  -> run in-process or via process pool
        """
        ctx = getattr(task, "runtime_env", None)
        ctx_mgr = ctx.apply() if ctx is not None else nullcontext()

        try:
            with ctx_mgr:
                fn = task.fn

                # Actor creation (always local):
                if isinstance(fn, dict) and fn.get("kind") == "actor_create":
                    actor_id = new_actor_id()
                    cls = fn["cls"]
                    init_args = tuple(fn.get("args", ()))
                    init_kwargs = dict(fn.get("kwargs", {}) or {})
                    instance = cls(*init_args, **init_kwargs)
                    self._actors[actor_id] = instance
                    ref = ActorRef(
                        actor_id=actor_id,
                        owner_node_id=self.node_id,
                    )
                    return self.store.put(ref)

                # Actor method call (always local):
                if isinstance(fn, dict) and fn.get("kind") == "actor_call":
                    actor_id = fn["actor_id"]
                    method_name = fn["method"]
                    instance = self._actors.get(actor_id)
                    if instance is None:
                        raise RuntimeError(f"Actor {actor_id} not found on node {self.node_id}.")
                    method = getattr(instance, method_name, None)
                    if method is None or not callable(method):
                        raise AttributeError(f"Actor method {method_name} not found.")
                    result = method(*task.args, **(task.kwargs or {}))
                    ref = self.store.put(result)
                    return ref

                # Regular function call (local or pool)
                if self.pool is None:
                    result = _invoke(fn, task.args, task.kwargs)
                else:
                    future = self.pool.submit(_invoke, fn, task.args, task.kwargs)
                    result = future.result()

                ref = self.store.put(result)
                return ref

        except Exception as e:
            raise RuntimeError(f"Task failed in worker@{self.node_id}") from e

    def rpc_read_object_bytes(self, object_id: str) -> bytes:
        """
        Return serialized bytes for a local object.

        Args:
            object_id (str): The local object id.

        Returns:
            bytes: Serialized payload.

        """
        return self.store.get_bytes(object_id)

    def rpc_execute_task(self, task: Task) -> ObjectRef:
        """
        Execute a task on behalf of a remote caller.

        Args:
            task (Task): The remote execution request.

        Returns:
            ObjectRef: Handle to the produced result (owned by this node).
        """
        ref = self.execute_task(task)

        try:
            payload = self.store.get_bytes(ref.object_id)
            size = len(payload)
        except Exception:
            size = None

        return ObjectRef(
            object_id=ref.object_id,
            owner_node_id=self.store.node_id,
            size_bytes=size
        )
