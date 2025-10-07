from typing import Dict, Tuple, Any, Optional, List

from nanorlhf.nanoray.core.object_ref import ObjectRef
from nanorlhf.nanoray.core.object_store import ObjectStore
from nanorlhf.nanoray.core.task import Task
from nanorlhf.nanoray.network.router import Router
from nanorlhf.nanoray.network.rpc_client import RpcClient
from nanorlhf.nanoray.scheduler.policies import SchedulingPolicy
from nanorlhf.nanoray.scheduler.scheduler import Scheduler, WorkerLike


class Session:
    """
    Runtime session that owns a `Scheduler` and exposes driver utilities.

    This class centralizes the "driver-side" API and networking hooks:
    it submits tasks to the scheduler, drains the queue, and retrieves
    values referenced by `ObjectRef`s across local and remote nodes.

    Args:
        policy (SchedulingPolicy): Node selection policy (e.g., FIFO/RoundRobin).
        nodes (Dict[str, Tuple[WorkerLike, Dict[str, Any]]]):
            Mapping of `node_id -> (worker, capacity_dict)`.
            `capacity_dict` fields: `{"cpus": float, "gpus": float, "resources": dict}`.
        default_node_id (Optional[str]): Node to use for `put()` convenience.
        router (Optional[Router]): Router used for remote object placement/lookup.
        rpc (Optional[RpcClient]): RPC client used for remote object transfer.

    Attributes:
        scheduler (Scheduler): The scheduler instance.
        _workers (Dict[str, WorkerLike]): All workers (local + remote proxies).
        _local_workers (Dict[str, WorkerLike]): Only local workers (expose `.store`).
        _default_node_id (str): Default node id for `put()` or caching.
        _driver_store (Optional[ObjectStore]): Fallback cache when there is no local worker.
        _router (Optional[Router]): Router for object routing (may be None).
        _rpc (Optional[RpcClient]): Rpc client for remote fetch (may be None).
        _aliases (Dict[str, str]): Mapping remote object_id -> local cached object_id.

    Examples:
        >>> # Minimal single-node session (local only)
        >>> from nanorlhf.nanoray.core.object_store import ObjectStore
        >>> from nanorlhf.nanoray.runtime.worker import Worker
        >>> from nanorlhf.nanoray.scheduler.policies import FIFO
        >>> nodes = {"A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}})}
        >>> sess = Session(policy=FIFO(), nodes=nodes, default_node_id="A")
        >>> ref = sess.put(123)
        >>> sess.get(ref)
        123

    Discussion:
        Q. Why a "driver cache" `ObjectStore`?
            In multi-node tests where all workers are remote (no local `.store`),
            we still need a local place to cache remotely fetched bytes for fast
            subsequent `get()`. A private driver-side `ObjectStore("__driver__")`
            keeps the logic simple and explicit.

        Q. Why keep an alias map instead of reusing the remote id?
            Local stores generate their own `object_id`. After fetching bytes from
            a remote owner, we re-materialize the object locally; the alias map
            remembers that "remote_id -> local_id" for instant local hits later.

        Q. What is the `get()` lookup order?
            (1) alias cache → (2) owner-first if the owner is local →
            (3) scan local workers → (4) fetch from remote (router+rpc) and cache.
    """

    def __init__(
        self,
        policy: SchedulingPolicy,
        nodes: Dict[str, Tuple[WorkerLike, Dict[str, Any]]],
        default_node_id: Optional[str] = None,
        *,
        router: Optional[Router] = None,
        rpc: Optional[RpcClient] = None,
    ):
        self.scheduler = Scheduler(policy=policy, nodes=nodes)

        # All workers (local + remote proxies)
        self._workers: Dict[str, WorkerLike] = {nid: w for nid, (w, _) in nodes.items()}

        # Only local workers (those exposing a `.store` attribute)
        self._local_workers: Dict[str, WorkerLike] = {
            nid: w for nid, w in self._workers.items() if hasattr(w, "store")
        }

        # Default node id resolution and driver cache setup
        if self._local_workers:
            self._default_node_id = (
                default_node_id
                if (default_node_id in self._local_workers)
                else next(iter(self._local_workers))
            )
            self._driver_store: Optional[ObjectStore] = None
        else:
            # No local workers: pick any known id for identification
            self._default_node_id = (
                default_node_id if (default_node_id in self._workers) else next(iter(self._workers))
            )
            # Driver-side cache for remote fetches
            self._driver_store = ObjectStore("__driver__")

        self._router = router
        self._rpc = rpc

        # Remote object_id -> local cached object_id
        self._aliases: Dict[str, str] = {}

    # ---------- internal helpers ----------

    def _cache_store(self) -> ObjectStore:
        """
        Return the store used to cache fetched objects locally.

        Returns:
            ObjectStore: Default local worker's store if present, otherwise the driver store.
        """
        if self._local_workers:
            return getattr(self._local_workers[self._default_node_id], "store")
        assert self._driver_store is not None
        return self._driver_store

    # ---------- driver surface ----------

    def submit(self, task: Task) -> Optional[ObjectRef]:
        """
        Submit a task to the scheduler (enqueue-only).

        The teaching scheduler is enqueue-only: this method always returns
        `None`. Call `drain()` to advance the queue and collect produced refs.

        Args:
            task (Task): Declarative description of a remote function call.

        Returns:
            Optional[ObjectRef]: Always `None` in the enqueue-only model.

        Examples:
            >>> # build tasks, submit all, then drain()
            >>> refs = []
            >>> for i in range(3):
            ...     _ = sess.submit(Task.from_call(lambda x: x + 1, args=(i,)))
            >>> refs += sess.drain()
        """
        return self.scheduler.submit(task)

    def drain(self) -> List[ObjectRef]:
        """
        Advance scheduling until the pending queue no longer progresses.

        Returns:
            List[ObjectRef]: Refs produced during draining.

        Examples:
            >>> # After submit(), use drain() to actually run and collect results
            >>> refs = sess.drain()
            >>> values = [sess.get(r) for r in refs]
        """
        return self.scheduler.drain()

    def put(self, value: Any, *, node_id: Optional[str] = None) -> ObjectRef:
        """
        Store a Python object and return an `ObjectRef`.

        If there is at least one local worker, the value is stored in that
        worker's `ObjectStore` (by `node_id` or by the default local node).
        If there are no local workers, the value is stored in the driver cache.

        Args:
            value (Any): Python object to store.
            node_id (Optional[str]): Target local node id; defaults to the default local node.

        Returns:
            ObjectRef: Handle to the stored value.

        Discussion:
            Q. Why allow `put()` with no local workers?
                For remote-only experiments, it is useful to still put values
                deterministically into the driver cache (e.g., for small config blobs).
        """
        if self._local_workers:
            nid = node_id if (node_id in self._local_workers) else self._default_node_id
            worker = self._local_workers[nid]
            return worker.store.put(value)  # type: ignore[attr-defined]
        return self._cache_store().put(value)

    def get(self, ref: ObjectRef) -> Any:
        """
        Retrieve the Python value for a given `ObjectRef`.

        Lookup order:
            1) alias cache
            2) owner-first (only if the owner is local)
            3) scan local workers
            4) remote fetch via router+rpc (then cache locally)

        Args:
            ref (ObjectRef): Handle whose value should be fetched.

        Returns:
            Any: The stored Python object.

        Raises:
            RuntimeError: If object is not found locally and networking is not configured.

        Examples:
            >>> # After drain(), get values back (local or remote)
            >>> [sess.get(r) for r in refs]  # doctest: +SKIP
        """
        # 1) alias cache fast-path
        local_id = self._aliases.get(ref.object_id)
        if local_id is not None:
            store = self._cache_store()
            if store.has(local_id):
                return store.get(ObjectRef(object_id=local_id, owner_node_id=getattr(store, "node_id", None)))

        # 2) owner-first (when owner is local)
        if ref.owner_node_id and (ref.owner_node_id in self._local_workers):
            store = getattr(self._local_workers[ref.owner_node_id], "store")
            if store.has(ref.object_id):
                return store.get(ref)

        # 3) scan local workers
        for w in self._local_workers.values():
            store = getattr(w, "store")
            if store.has(ref.object_id):
                return store.get(ref)

        # 4) remote fetch (requires router+rpc)
        if self._router is not None and self._rpc is not None:
            owner_id = self._router.route_object(ref) or ref.owner_node_id
            if owner_id is None:
                raise RuntimeError("No owner information available for remote get().")

            payload = self._rpc.get_object(owner_id, ref.object_id)
            cache = self._cache_store()
            local_ref = cache.put_bytes(payload)  # new local id
            self._aliases[ref.object_id] = local_ref.object_id
            return cache.get(local_ref)

        raise RuntimeError(
            "Object not found in local stores and networking hooks are not configured. "
            "Provide `router` and `rpc` to enable remote get()."
        )


_GLOBAL_SESSION: Optional[Session] = None


def init_session(
    policy: SchedulingPolicy,
    nodes: Dict[str, Tuple[WorkerLike, Dict[str, Any]]],
    default_node_id: Optional[str] = None,
) -> Session:
    """
    Initialize the global session.

    Args:
        policy (SchedulingPolicy): Node selection policy (e.g., FIFO/RoundRobin).
        nodes (Dict[str, Tuple[WorkerLike, Dict[str, Any]]]):
            Mapping of `node_id -> (worker, capacity_dict)`.
            `capacity_dict` fields: `{"cpus": float, "gpus": float, "resources": dict}`.
        default_node_id (Optional[str]): Node to use for `put()` convenience.

    Returns:
        Session: The created global session.

    Examples:
        >>> from nanorlhf.nanoray.scheduler.policies import RoundRobin
        >>> from nanorlhf.nanoray.core.object_store import ObjectStore
        >>> from nanorlhf.nanoray.runtime.worker import Worker
        >>> nodes = {
        ...   "A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}}),
        ...   "B": (Worker(store=ObjectStore("B")), {"cpus": 1.0, "gpus": 0.0, "resources": {}}),
        ... }
        >>> _ = init_session(RoundRobin(), nodes, default_node_id="A")
    """
    global _GLOBAL_SESSION
    _GLOBAL_SESSION = Session(policy=policy, nodes=nodes, default_node_id=default_node_id)
    return _GLOBAL_SESSION


def get_session() -> Session:
    """
    Return the global session.

    Returns:
        Session: The global session.

    Raises:
        RuntimeError: If the global session has not been initialized.

    Examples:
        >>> _ = get_session()  # doctest: +ELLIPSIS
    """
    if _GLOBAL_SESSION is None:
        raise RuntimeError("Global session is not initialized. Call `init_session(...)` first.")
    return _GLOBAL_SESSION


def get(ref: ObjectRef) -> Any:
    """
    Convenience function: fetch a value via the global session.

    Args:
        ref (ObjectRef): Handle whose value should be fetched.

    Returns:
        Any: The stored Python object.

    Examples:
        >>> # Single-node smoke
        >>> from nanorlhf.nanoray.core.object_store import ObjectStore
        >>> from nanorlhf.nanoray.runtime.worker import Worker
        >>> from nanorlhf.nanoray.scheduler.policies import FIFO
        >>> nodes = {"A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}})}
        >>> _ = init_session(FIFO(), nodes, default_node_id="A")
        >>> ref = put(123)
        >>> get(ref)
        123
    """
    sess = get_session()
    return sess.get(ref)


def put(value: Any, *, node_id: Optional[str] = None) -> ObjectRef:
    """
    Convenience function: store a value via the global session.

    Args:
        value (Any): Python object to store.
        node_id (Optional[str]): Target local node id; defaults to the session default.

    Returns:
        ObjectRef: Handle to the stored value.

    Examples:
        >>> ref = put({"x": 1})  # doctest: +ELLIPSIS
    """
    sess = get_session()
    return sess.put(value, node_id=node_id)


def submit(task: Task) -> Optional[ObjectRef]:
    """
    Convenience function: submit a task via the global session (enqueue-only).

    Args:
        task (Task): Declarative description of a remote function call.

    Returns:
        Optional[ObjectRef]: Always `None` in the enqueue-only model.

    Examples:
        >>> def add(x, y): return x + y
        >>> _ = submit(Task.from_call(add, args=(3, 4)))
        >>> refs = drain()
        >>> get(refs[-1])
        7
    """
    sess = get_session()
    return sess.submit(task)


def drain() -> List[ObjectRef]:
    """
    Convenience function: advance scheduling in the global session.

    Returns:
        List[ObjectRef]: Refs produced during draining.

    Examples:
        >>> refs = drain()  # doctest: +ELLIPSIS
    """
    sess = get_session()
    return sess.drain()
