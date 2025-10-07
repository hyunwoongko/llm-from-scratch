from contextlib import nullcontext
from typing import Optional

from nanorlhf.nanoray.core.object_ref import ObjectRef
from nanorlhf.nanoray.core.object_store import ObjectStore
from nanorlhf.nanoray.core.task import Task
from nanorlhf.nanoray.runtime.process_pool import ProcessPool


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
    `Worker` executes a `Task`, stores the result into an `ObjectStore`,
    and returns an `ObjectRef` to the caller.

    The worker can run tasks in the current process (default) or submit them to
    a process pool for CPU-bound or isolation-friendly execution.

    Args:
        store (ObjectStore): The node-local object store.
        pool (Optional[ProcessPool]): Optional process-based executor.

    Examples:
        >>> # Local (in-process) execution
        >>> from nanorlhf.nanoray.core.task import Task
        >>> from nanorlhf.nanoray.core.object_store import ObjectStore
        >>> def add(x, y): return x + y
        >>> store = ObjectStore("node-A")
        >>> w = Worker(store=store)
        >>> task = Task.from_call(add, (3, 4))
        >>> ref = w.execute_task(task)
        >>> store.get(ref)
        7

        >>> # With a process pool (teaching example)
        >>> from nanorlhf.nanoray.runtime.process_pool import ProcessPool
        >>> pool = ProcessPool(max_workers=2)
        >>> w2 = Worker(store=store, pool=pool)
        >>> ref2 = w2.execute_task(task)  # executed in a separate process
        >>> store.get(ref2)
        7
        >>> pool.shutdown()
    """

    def __init__(self, store: ObjectStore, pool: Optional[ProcessPool] = None):
        self.store = store
        self.pool = pool

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
        """
        node_id = self.store.node_id

        try:
            with task.runtime_env.apply() if task.runtime_env is not None else nullcontext():
                if self.pool is None:
                    result = _invoke(task.fn, task.args, task.kwargs)
                else:
                    future = self.pool.submit(_invoke, task.fn, task.args, task.kwargs)
                    result = future.result()
            ref = self.store.put(result)
            return ref

        except Exception as e:
            raise RuntimeError(
                f"Task {task.task_id} failed in worker@{node_id}"
            ) from e

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
