from concurrent.futures import ProcessPoolExecutor, Future
from typing import Any, Callable


class ProcessPool:
    """
    A tiny wrapper around `concurrent.futures.ProcessPoolExecutor`

    It intentionally exposes a minimal surface (`submit`, `shutdown`)
    while still providing real multiprocess execution for cpu-bound tasks.

    Args:
        max_workers (int): Number of worker processes to spawn.

    Examples:
        >>> from nanorlhf.nanoray.runtime.process_pool import ProcessPool
        >>> def square(x): return x * x
        >>> pool = ProcessPool(max_workers=4)
        >>> future = pool.submit(square, 5)
        >>> future.result()
        25
        >>> pool.shutdown()

    Discussion:
        Q. Why a process pool instead of threads?
            Python's GIL can serialize cpu-bound bytecode.
            Using processes bypasses the GIL and gives true parallelism
            for pure-Python CPU work (at a higher IPC cost).
    """

    def __init__(self, max_workers: int = 1):
        self._executor = ProcessPoolExecutor(max_workers=max_workers)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """
        Submit a callable to the process pool.

        Args:
            fn (Callable[..., Any]): The function to execute in a worker process.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Future: A future whose `.result()` blocks until completion.
        """
        return self._executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True):
        """
        Cleanly shut down the process pool, optionally waiting for tasks to finish.

        Args:
            wait (bool): If `True`, block until all tasks complete before returning.
                If `False`, return immediately and let tasks finish in the background.
                Default is `True`.
        """
        self._executor.shutdown(wait=wait)