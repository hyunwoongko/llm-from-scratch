from nanorlhf.nanoray.api.remote import actor
from nanorlhf.nanoray.api.session import init_session, get as get_val
from nanorlhf.nanoray.core.object_store import ObjectStore
from nanorlhf.nanoray.runtime.worker import Worker
from nanorlhf.nanoray.scheduler.policies import FIFO, RoundRobin


def _init_single():
    nodes = {"A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}})}
    return init_session(policy=FIFO(), nodes=nodes, default_node_id="A")


def _init_multi(policy):
    nodes = {
        "A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}}),
        "B": (Worker(store=ObjectStore("B")), {"cpus": 1.0, "gpus": 0.0, "resources": {}}),
    }
    return init_session(policy=policy, nodes=nodes, default_node_id="A")


def test_actor_sticky_single_node():
    _ = _init_single()

    @actor
    class Counter:
        def __init__(self):
            self.x = 0

        def inc(self, n=1):
            self.x += n
            return self.x

    h = get_val(Counter.remote())
    r1 = get_val(h.inc.remote(3))
    r2 = get_val(h.inc.remote(5))
    assert (r1, r2) == (3, 8)


def test_actor_pinned_to_specific_node_inprocess():
    _ = _init_multi(RoundRobin())

    @actor
    class Holder:
        def __init__(self, id_):
            self.id_ = id_

        def owner(self):
            return self.id_

    h_ref = Holder.options(pinned_node_id="B").remote("hello")
    h = get_val(h_ref)
    out = get_val(h.owner.remote())
    assert out == "hello"
