import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

from nanorlhf.nanoray.core.runtime_env import RuntimeEnv

T = TypeVar("T")


@dataclass(frozen=True)
class Task(Generic[T]):
    """
    `Task` is an immutable data record that describes a single remote function call.
    Instead of invoking a Python function immediately, the runtime the call as a declarative
    task and submits it to the scheduler for placement and execution.

    Attributes:
        task_id (str): A globally unique id (e.g., `"task-9ff8c3a2"`).
        fn (Callable[..., T]): The Python function to execute remotely.
        args (Tuple[Any, ...]): Positional arguments to pass to `fn`.
        kwargs (Dict[str, Any]): Keyword arguments to pass to `fn`.
        num_cpus (float): CPU resource requirement (default 1.0).
        num_gpus (float) : GPU resource requirement (default 0.0).
        resources (Optional[Dict[str, float]]): Custom resources (e.g., {"ram_gb": 4.0}).
        runtime_env (Optional[RuntimeEnv]): Optional environment description.
        placement_group (Optional[str]): Optional placement group id for colocation/anti-colocation.
        priority (int): Scheduling hint (higher = more urgent, default 0).

    Examples:
        >>> def add(x, y): return x + y
        >>> task = Task(fn=add, args=(1, 2), kwargs={}, num_cpus=0.5)
        >>> print(task.fn is add, task.args, task.num_cpus)
        True (1, 2) 0.5

    Discussion:
        Q. Why keep a `Task` instead of calling the function right away?
            In a distributed runtime, *when* and *where* to execute matters.
            By turning a call into a task, the scheduler can choose the best node
            (considering resources and placement groups) before any execution happens.

        Q. Why immutable?
            Once submitted, the task is shared across components (driver, scheduler, worker).
            Making it immutable prevents accidental mutation after scheduling decisions.
    """

    # identify & call
    task_id: str
    fn: Callable[..., T]
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # resources & context
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    resources: Optional[Dict[str, float]] = None
    runtime_env: Optional[RuntimeEnv] = None
    placement_group: Optional[str] = None
    priority: int = None

    @classmethod
    def from_call(
        cls,
        fn: Callable[..., T],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        *,
        num_cpus: float = 1.0,
        num_gpus: float = 0.0,
        resources: Optional[Dict[str, float]] = None,
        runtime_env: Optional[RuntimeEnv] = None,
        placement_group: Optional[str] = None,
        priority: int = 0,
    ) -> "Task[T]":
        """
        Build a `Task` from a Python callable and its arguments.

        Args:
            fn (Callable[..., T]): The function to run remotely.
            args (Tuple[Any, ...]): Positional arguments.
            kwargs (Optional[Dict[str, Any]]): Keyword arguments (default `{}`).
            num_cpus (float): CPU requirement (default `1.0`).
            num_gpus (float): GPU requirement (default `0.0`).
            resources (Optional[Dict[str, float]]): Custom resources (e.g., `{"ram_gb": 4}`).
            runtime_env (Optional[RuntimeEnv]): Optional runtime environment task.
            placement_group (Optional[str]): Placement group id, if any.
            priority (int): Scheduling hint (teaching-only), default `0`.

        Returns:
            Task[T]: A new immutable task taskification.

        Examples:
            >>> def mul(a, b): return a * b
            >>> Task.from_call(mul, (3, 4)).args
            (3, 4)
        """
        return cls(
            task_id=f"task-{uuid.uuid4().hex[:8]}",
            fn=fn,
            args=args,
            kwargs={} if kwargs is None else kwargs,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            resources=resources,
            runtime_env=runtime_env,
            placement_group=placement_group,
            priority=priority,
        )

    def summary(self) -> str:
        """
        Return a short human-friendly summary for logging and debugging.

        Returns:
            str: A string like `"task-9f3a12ab fn=add cpus=1.0 gpus=0.0 pg=None"`.
        """
        fn_name = getattr(self.fn, "__name__", str(self.fn))

        return (
            f"{self.task_id} fn={fn_name} "
            f"cpus={self.num_cpus} gpus={self.num_gpus} "
            f"pg={self.placement_group}"
        )
