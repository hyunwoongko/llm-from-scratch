import os

import pytest

from nanorlhf.nanoray import init, shutdown, NodeConfig, get
from nanorlhf.nanoray.api.remote import remote, actor

pytestmark = pytest.mark.skipif(os.getenv("RUN_RPC_TESTS") != "1",
                                reason="RUN_RPC_TESTS != 1")


@pytest.fixture(autouse=True)
def _remote_two_nodes():
    cfg = {
        "A": NodeConfig(local=False, address="http://127.0.0.1:8003", remote_token="tA", cpus=1.0),
        "B": NodeConfig(local=False, address="http://127.0.0.1:8004", remote_token="tB", cpus=1.0),
    }
    _ = init(cfg, default_node_id="A")
    try:
        yield
    finally:
        shutdown()


def test_remote_fn_round_robin_remote_servers():
    @remote()
    def whoami(i):
        import os, platform
        return platform.node(), os.getpid(), i

    refs = [whoami.remote(i) for i in range(8)]
    vals = [get(r) for r in refs]
    owners = [r.owner_node_id for r in refs]
    assert owners == ["A", "B", "A", "B", "A", "B", "A", "B"]
    assert [v[2] for v in vals] == list(range(8))


def test_actor_sticky_remote_server():
    @actor
    class Counter:
        def __init__(self):
            self.x = 0

        def inc(self, n=1):
            self.x += n
            return self.x

    h = get(Counter.options(pinned_node_id="B").remote())
    out = get(h.inc.remote(5))
    assert out == 5
    assert h.owner_node_id == "B"
