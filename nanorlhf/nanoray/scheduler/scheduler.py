import heapq
from typing import Dict, List, Tuple, Optional, Protocol

from nanorlhf.nanoray.core.object_ref import ObjectRef
from nanorlhf.nanoray.core.task_spec import TaskSpec
from nanorlhf.nanoray.scheduler.node_state import NodeState
from nanorlhf.nanoray.scheduler.policies import SchedulingPolicy


class WorkerLike(Protocol):
    """
    Minimal execution surface the scheduler needs.

    Discussion:
        Q. What is `Protocol`?
            A structural interface (duck typing) that lets us avoid cyclic imports.
            Both `Worker` and `RemoteWorkerProxy` can satisfy this protocol.
    """

    def execute_task(self, spec: TaskSpec) -> ObjectRef: ...


class Scheduler:
    """
    Minimal scheduler implementation

    Args:
        policy (SchedulingPolicy): Node selection strategy (e.g., `FIFO`, `RoundRobin`)
        nodes (Dict[str, Tuple[WorkerLike, Dict[str, Any]]]):
            Mapping of `node_id -> (worker, capacity_dict)`.
            `capacity_dict` fields: `{"cpus": float, "gpus": float, "resources": dict}`.

    Examples:
        >>> from nanorlhf.nanoray.scheduler.policies import RoundRobin
        >>> from nanorlhf.nanoray.core.object_store import ObjectStore
        >>> from nanorlhf.nanoray.core.task_spec import TaskSpec
        >>> from nanorlhf.nanoray.runtime.worker import Worker
        >>> def add(x, y): return x + y
        >>> nodes = {
        ...   "A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}}),
        ...   "B": (Worker(store=ObjectStore("B")), {"cpus": 2.0, "gpus": 0.0, "resources": {}}),
        ... }
        >>> sched = Scheduler(policy=RoundRobin(), nodes=nodes)
        >>> specs = [TaskSpec.from_call(add, (i, i)) for i in range(4)]
        >>> refs = [sched.submit(s) for s in specs] + sched.drain()
        >>> all(r is not None for r in refs)
        True

    Discussion:
        Q. What does the scheduler do end-to-end?
            (1) Accepts `TaskSpec`s
            (2) Chooses a node using a `SchedulingPolicy`
            (3) Executes on that node's `Worker`
            (4) Returns the produced `ObjectRef`

        Q. How does queueing work when placement fails?
            Tasks that cannot be placed immediately go into a priority queue ordered
            by `priority` (higher first) and FIFO within the same priority.

        Q. Where are `num_cpus`, `num_gpus`, `resources`, and `priority` used?
            They are consumed at *placement*: we filter by available capacity, enqueue
            by `priority`, and pick one candidate via the policy. `Worker` only executes.

        Q. Why `WorkerLike`?
            To decouple placement from execution transport. A local `Worker` and a
            remote-RPC-backed `WorkerProxy` can both satisfy the same protocol,
            so the scheduler doesn't need to know *how* execution happens.
    """

    def __init__(
        self,
        policy: SchedulingPolicy,
        nodes: Dict[str, tuple[WorkerLike, Dict[str, float]]]
    ):
        self.policy = policy
        self.nodes = nodes
        self._workers: Dict[str, WorkerLike] = {}
        self._state: Dict[str, NodeState] = {}

        order: List[str] = []
        for nid, (worker, cap) in nodes.items():
            self._workers[nid] = worker
            self._state[nid] = NodeState(
                total_cpus=cap.get("cpus", 1.0),
                total_gpus=cap.get("gpus", 0.0),
                total_custom=cap.get("resources", {}),
            )
            order.append(nid)

        self._order = order
        self.policy.set_node_order(order)

        # priority queue of pending tasks: (-priority, seq, task_spec)
        self._q: List[Tuple[int, int, TaskSpec]] = []
        self._seq = 0

    def submit(self, spec: TaskSpec) -> Optional[ObjectRef]:
        """
        Try to place and execute the task immediately.
        Otherwise, enqueue it for later placement.

        Args:
            spec (TaskSpec): Declarative description of a remote function call.

        Returns:
            Optional[ObjectRef]: Result reference if placed now, else `None`.
        """
        neg_pri = -int(spec.priority or 0)
        heapq.heappush(self._q, (neg_pri, self._seq, spec))
        self._seq += 1
        return None

    def drain(self) -> List[ObjectRef]:
        """
        Keep scheduling until queue is empty or no further progress can be made.

        Returns:
            List[ObjectRef]: Refs for tasks that were scheduled during draining.

        Discussion:
            Q. What is this method doing, step by step?
                We run in "rounds" (passes) over the pending queue:

                1) Start a round with `progressed=False` and an empty `pending`.

                2) Pop every task in order (priority first, FIFO within same priority).
                   For each task:
                     - Try to place it now (`_try_place`).
                     - If placed: append the returned `ObjectRef` to `produced`
                       and set `progressed=True`.
                     - If not placed: append the tuple back into `pending`.

                3) After inspecting the whole heap once, push every item in `pending`
                   back into the heap unchanged. This preserves both *priority* and
                   *FIFO order* because we reinsert the same `(priority, seq, spec)` tuples.

                4) If at least one task ran (`progressed=True`), we do another round,
                   because completed tasks may have freed resources and unlocked others.
                   If none ran (`progressed=False`), looping again would not change the
                   state, so we stop.

                In short:
                    - Each round tries to place every pending task once.
                    - Tasks that cannot be placed remain in the queue for future rounds.
                    - We stop when either the queue is empty or a full round makes no progress.

            Q. Does it terminate?
                Yes. Either the heap empties (everything placed) or a whole round
                makes no placements (`progressed=False`), so another round would be
                identical and we exit deterministically.
        """
        produced: List[ObjectRef] = []
        progressed = True

        while self._q and progressed:
            progressed = False
            pending: List[Tuple[int, int, TaskSpec]] = []
            while self._q:
                priority, seq, spec = heapq.heappop(self._q)
                ref = self._try_place(spec)
                if ref is None:
                    pending.append((priority, seq, spec))
                else:
                    produced.append(ref)
                    progressed = True
            for item in pending:
                heapq.heappush(self._q, item)
        return produced

    def _eligible_nodes(self, spec: TaskSpec) -> List[str]:
        """
        Find nodes that have enough available capacity for the given task spec.

        Args:
            spec (TaskSpec): The task specification to check.

        Returns:
            List[str]: Node IDs that can run the task.
        """
        return [
            nid for nid, state in self._state.items()
            if state.can_run(spec)
        ]

    def _try_place(self, spec: TaskSpec) -> Optional[ObjectRef]:
        """
        Attempt to place the task on an eligible node using the scheduling policy.

        Args:
            spec (TaskSpec): The task specification to place.

        Returns:
            Optional[ObjectRef]: Result reference if placed, else `None`.
        """
        cands = self._eligible_nodes(spec)
        if not cands:
            return None

        nid = self.policy.select(cands)
        if nid is None:
            return None

        st = self._state[nid]
        st.allocate(spec)

        try:
            worker = self._workers[nid]
            ref = worker.execute_task(spec)
            return ref
        finally:
            st.release(spec)
