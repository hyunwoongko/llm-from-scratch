# tests/test_local_end_to_end.py
"""
Local end-to-end tests for nanoray (no RPC). Aims to be realistic but runnable on laptops.

Discussion:
    Q. Why no RPC here?
       The goal is to stress core scheduling/execution/serialization locally first.
       RPC tests will be added later to isolate transport concerns.

    Q. Why pytest?
       Concise assertions, easy param/skips, and good reporting. The tests avoid
       timing assumptions to stay stable across machines.
"""

import os
import tempfile
from typing import Any, Tuple

import pytest

from nanorlhf.nanoray.api.session import Session
from nanorlhf.nanoray.core import serialization as ser
from nanorlhf.nanoray.core.object_store import ObjectStore
from nanorlhf.nanoray.core.runtime_env import RuntimeEnv
from nanorlhf.nanoray.core.task import Task
from nanorlhf.nanoray.runtime.worker import Worker
from nanorlhf.nanoray.scheduler.policies import FIFO, RoundRobin


# ---------- 1) basic put/get -------------------------------------------------

def test_put_get_local_store() -> None:
    """
    Basic sanity: putting and getting a Python object through the default node.

    Discussion:
        Q. Why start with the simplest case?
           It validates id/ownership wiring and avoids confounding factors before
           we test scheduling and env behavior.
    """
    nodes = {"A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}})}
    sess = Session(policy=FIFO(), nodes=nodes, default_node_id="A")

    ref = sess.put({"k": "v"})
    out = sess.get(ref)
    assert out == {"k": "v"}


# ---------- 2) priority & round draining semantics ---------------------------

def _make_priority_task(tag: str) -> Any:
    def fn(x: int) -> Tuple[str, int]:
        # returns (tag, x) so we can check the observed order
        return (tag, x)

    return fn


def test_scheduler_priority_and_fifo_rounds() -> None:
    """
    One CPU node; tasks are queued with different priorities.
    We expect higher priority first, then FIFO among equal priorities.

    Discussion:
        Q. Why single CPU?
           To force sequential placement so the drain order reflects the heap
           ordering (priority, then sequence).
    """
    nodes = {"A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}})}
    sess = Session(policy=FIFO(), nodes=nodes, default_node_id="A")

    f = _make_priority_task("P")
    # build tasks with explicit priorities
    tasks = [
        Task.from_call(f, args=(1,), priority=0),
        Task.from_call(f, args=(2,), priority=10),
        Task.from_call(f, args=(3,), priority=5),
        Task.from_call(f, args=(4,), priority=10),
        Task.from_call(f, args=(5,), priority=5),
    ]
    # submit in arbitrary order
    refs = [sess.submit(s) for s in tasks]
    # anything placed immediately comes back as ObjectRef; rest are queued
    refs = [r for r in refs if r is not None]
    # drain the queue (produces in scheduling order)
    refs += sess.drain()

    # fetch values and observe relative ordering
    vals = [sess.get(r) for r in refs]
    # Expected order by priority desc, FIFO within same priority:
    # priority 10: args=2,4  then priority 5: args=3,5  then priority 0: args=1
    assert [v[1] for v in vals] == [2, 4, 3, 5, 1]


# ---------- 3) remote decorator with heavier numeric work --------------------

np = pytest.importorskip("numpy")  # heavy-ish test requires numpy

from nanorlhf.nanoray.api.remote import remote  # after numpy import to keep skips clean


@remote(num_cpus=1.0, priority=5)
def mmul(A, B):
    """Matrix multiply using numpy."""
    return (A @ B).astype("float32")  # keep memory modest


def test_remote_many_mmul_roundrobin() -> None:
    """
    Submit multiple matrix multiplications and distribute them RoundRobin.

    Discussion:
        Q. Why RoundRobin?
           To demonstrate multi-node placement effects even without RPC. Both nodes
           are local Workers; policy decides where to run.
    """
    # two local nodes (no process pools yet, but fine for correctness)
    nodes = {
        "A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}}),
        "B": (Worker(store=ObjectStore("B")), {"cpus": 1.0, "gpus": 0.0, "resources": {}}),
    }
    sess = Session(policy=RoundRobin(), nodes=nodes, default_node_id="A")

    # build inputs: 16 small-ish multiplications (128x128) to stay fast on laptops
    rng = np.random.default_rng(0)
    payloads = [(rng.standard_normal((128, 128), dtype="float32"),
                 rng.standard_normal((128, 128), dtype="float32")) for _ in range(16)]

    # submit via remote decorator (global session not required since we call .task())
    tasks = [mmul.options(priority=10 if i < 4 else 0).task(A, B) for i, (A, B) in enumerate(payloads)]
    refs = [sess.submit(s) for s in tasks]
    refs = [r for r in refs if r is not None] + sess.drain()

    # verify numerically against numpy ground truth
    for r, (A, B) in zip(refs, payloads):
        got = sess.get(r)
        expect = (A @ B).astype("float32")
        np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-5)


# ---------- 4) runtime env scoping (env vars + cwd) --------------------------

def _echo_env_and_cwd(var: str) -> Tuple[str, str | None]:
    import os
    return os.getcwd(), os.getenv(var)


def test_runtime_env_scoped_changes_tmpdir(tmp_path: tempfile.TemporaryDirectory) -> None:
    """
    RuntimeEnv applies env vars and cwd as a *scoped* context around execution.

    Discussion:
        Q. Why check cwd and env together?
           It proves that the context manager in `RuntimeEnv.apply()` snapshots &
           restores process state correctly even when both are present.
    """
    nodes = {"A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}})}
    sess = Session(policy=FIFO(), nodes=nodes, default_node_id="A")

    env = RuntimeEnv(env_vars={"FOO": "BAR"}, cwd=str(tmp_path))
    task = Task.from_call(_echo_env_and_cwd, args=("FOO",), runtime_env=env)
    r = sess.submit(task) or sess.drain()[-1]
    cwd, foo = sess.get(r)
    assert cwd == str(tmp_path)
    assert foo == "BAR"

    # Ensure process state has been restored
    assert os.getenv("FOO") != "BAR"
    assert os.getcwd() != str(tmp_path)


# ---------- 5) serialization: GPU->CPU, zstd frame, roundtrip ----------------

def test_serialization_roundtrip_and_header() -> None:
    """
    Ensure framed dumps/loads roundtrip and header is self-describing.

    Discussion:
        Q. What do we assert about compression?
           We don't assume zstd is installed. Instead we assert the header is
           NRAY1 and ALG is either NONE or ZSTD, and that `loads(dumps(x)) == x`.
    """
    big_bytes = b"x" * (2_000_000)  # 2MB to trigger compression if available
    buf = ser.dumps(big_bytes, compression="zstd", compress_threshold=1_000_000, zstd_level=3)
    assert buf.startswith(b"NRAY1")
    alg = buf[5:9]
    assert alg in {b"NONE", b"ZSTD"}
    out = ser.loads(buf)
    assert out == big_bytes


torch = None
try:
    import torch  # type: ignore
except Exception:
    torch = None  # pragma: no cover


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_serialization_moves_cuda_to_cpu_if_available() -> None:
    """
    If CUDA tensor is present, `dumps` should move it to CPU for safe pickling.

    Discussion:
        Q. Why allow CPU-only pass?
           On laptops without CUDA, the test is skipped. The important bit is that
           when CUDA exists, the transport normalizes tensors to CPU automatically.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x = torch.ones((4, 4), device="cuda", dtype=torch.float32)
    buf = ser.dumps(x, compression=None)
    y = ser.loads(buf)
    assert isinstance(y, torch.Tensor)
    assert y.device.type == "cpu"
    assert torch.allclose(y, torch.ones((4, 4), dtype=torch.float32))


# ---------- 6) global wrappers (optional smoke) ------------------------------

def test_global_wrappers_smoke() -> None:
    """
    Smoke test for `init_session/get_session/get/put/submit/drain` wrappers.

    Discussion:
        Q. Why not use global wrappers everywhere?
           Tests prefer local Session instances to avoid shared global state.
           This smoke ensures the wrappers work at least once.
    """
    from nanorlhf.nanoray.api.session import (
        init_session, get_session,
        get as gget,
        put as gput,
        submit as gsubmit,
        drain as gdrain
    )

    nodes = {"A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}})}
    init_session(FIFO(), nodes, default_node_id="A")
    sess = get_session()

    r1 = gput({"x": 1})
    assert gget(r1) == {"x": 1}

    def add(a, b): return a + b

    r2 = gsubmit(Task.from_call(add, args=(2, 5)))
    refs = [r2] if r2 is not None else []
    refs += gdrain()
    assert gget(refs[-1]) == 7
