from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol

from nanorlhf.nanoray.core.object_ref import ObjectRef
from nanorlhf.nanoray.core.runtime_env import RuntimeEnv
from nanorlhf.nanoray.core.task_spec import TaskSpec


class _Submittable(Protocol):
    def __call__(self, spec: TaskSpec) -> Optional[ObjectRef]: ...


@dataclass(frozen=True)
class RemoteFunction:
    """
    A thin Ray-like wrapper that turns a Python function into a remote-invocable one.

    Attributes:
        fn (Callable[..., Any]): The original Python function to execute remotely.
        num_cpus (float): CPU requirement for scheduling (teaching-scale).
        num_gpus (float): GPU requirement for scheduling.
        resources (Dict[str, float]): Custom named resource requirements.
        priority (int): Higher value means earlier scheduling.
        runtime_env (Optional[RuntimeEnv]): Scoped env applied around this task.

    Examples:
        >>> from nanorlhf.nanoray.api.initialization import init, shutdown
        >>> from nanorlhf.nanoray.api.remote import remote
        >>> from nanorlhf.nanoray.api.session import get, drain
        >>>
        >>> @remote(num_cpus=1.0, priority=10)
        ... def add(x, y): return x + y
        ...
        >>> sess = init()                      # zero-arg init
        >>> r = add.remote(3, 4)               # Optional[ObjectRef]
        >>> refs = ([r] if r is not None else []) + drain()
        >>> get(refs[-1])
        7
        >>> shutdown()

    Discussion:
        Q. Why not always return an ObjectRef from `.remote(...)`?
            Our teaching scheduler may queue tasks if resources are unavailable.
            We keep this visible by returning `Optional[ObjectRef]` and using
            `drain()` to advance queued work. This makes scheduling state explicit.

        Q. What if I want a `TaskSpec` without submitting?
            Use `.spec(*args, **kwargs)` to build a `TaskSpec`. This works even
            without a global session and can be submitted manually later.

        Q. How does this differ from Ray?
            Ray’s `.remote` typically returns an `ObjectRef` immediately
            (submission is asynchronous). We model a minimal, transparent flow
            for lecture purposes and keep the API surface small.
    """

    fn: Callable[..., Any]
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    resources: Dict[str, float] = None  # type: ignore[assignment]
    priority: int = 0
    runtime_env: Optional[RuntimeEnv] = None

    def __post_init__(self) -> None:
        if self.resources is None:
            object.__setattr__(self, "resources", {})

    # -------- user-facing API --------

    def options(
        self,
        *,
        num_cpus: Optional[float] = None,
        num_gpus: Optional[float] = None,
        resources: Optional[Dict[str, float]] = None,
        priority: Optional[int] = None,
        runtime_env: Optional[RuntimeEnv] = None,
    ) -> "RemoteFunction":
        """
        Return a new RemoteFunction with per-call overrides.

        Examples:
            >>> @remote()
            ... def f(x): return x * 2
            ...
            >>> g = f.options(priority=100)
            >>> # g.remote(...) submits with higher scheduling priority
        """
        return RemoteFunction(
            fn=self.fn,
            num_cpus=self.num_cpus if num_cpus is None else float(num_cpus),
            num_gpus=self.num_gpus if num_gpus is None else float(num_gpus),
            resources=self.resources if resources is None else dict(resources),
            priority=self.priority if priority is None else int(priority),
            runtime_env=self.runtime_env if runtime_env is None else runtime_env,
        )

    def spec(self, *args: Any, **kwargs: Any) -> TaskSpec:
        """
        Build a `TaskSpec` for this function call without submitting it.

        Returns:
            TaskSpec: Declarative description of the remote call.

        Examples:
            >>> @remote()
            ... def add(x, y): return x + y
            ...
            >>> spec = add.spec(3, 4)  # create TaskSpec
        """
        return TaskSpec.from_call(
            self.fn,
            args=tuple(args),
            kwargs=dict(kwargs),
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
            resources=dict(self.resources),
            priority=int(self.priority),
            runtime_env=self.runtime_env,
        )

    def remote(self, *args: Any, **kwargs: Any) -> Optional[ObjectRef]:
        """
        Submit this function call to the global session's scheduler.

        Returns:
            Optional[ObjectRef]: Reference to the result if placed now; otherwise `None`.

        Raises:
            RuntimeError: If a global session is not initialized.

        Examples:
            >>> @remote()
            ... def add(x, y): return x + y
            ...
            >>> # after init()
            >>> r = add.remote(3, 4)   # may be None if queued
        """
        try:
            # Lazy import to avoid import cycles at module import time
            from nanorlhf.nanoray.api.session import submit  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Global session is not available; call init() first.") from exc

        spec = self.spec(*args, **kwargs)
        return submit(spec)  # Optional[ObjectRef]


def remote(_fn: Optional[Callable[..., Any]] = None, **opts: Any) -> Callable[..., RemoteFunction] | RemoteFunction:
    """
    Decorator/adapter that wraps a Python function into a `RemoteFunction`.

    Usage patterns:
        @remote(num_cpus=1.0)
        def f(x): ...

        # or:
        def g(x): ...
        g = remote(g, priority=5)

    Args:
        _fn (Optional[Callable]): The function to wrap when used as `remote(fn, ...)`.
        **opts: Options forwarded to `RemoteFunction` (num_cpus, num_gpus, resources, priority, runtime_env).

    Returns:
        RemoteFunction or a decorator that produces one.

    Examples:
        >>> @remote(priority=1)
        ... def mul(x, y): return x * y
        ...
        >>> # submit immediately (requires init())
        >>> r = mul.remote(6, 7)
        >>> # or build a spec without submitting:
        >>> spec = mul.spec(6, 7)
    """
    def _wrap(fn: Callable[..., Any]) -> RemoteFunction:
        return RemoteFunction(fn=fn, **opts)

    if _fn is None:
        return _wrap
    return _wrap(_fn)
