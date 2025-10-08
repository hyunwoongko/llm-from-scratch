from nanorlhf.nanoray.api.remote import remote, actor
from nanorlhf.nanoray.api.session import init_session, get_session, get as get_val, drain as drain_queue
from nanorlhf.nanoray.core.object_store import ObjectStore
from nanorlhf.nanoray.core.placement import Bundle, PlacementStrategy
from nanorlhf.nanoray.runtime.worker import Worker
from nanorlhf.nanoray.scheduler.policies import RoundRobin, FIFO


def _init_multi(policy):
    nodes = {
        "A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}}),
        "B": (Worker(store=ObjectStore("B")), {"cpus": 1.0, "gpus": 0.0, "resources": {}}),
    }
    return init_session(policy=policy, nodes=nodes, default_node_id="A")


def test_pg_pack_remote_function_all_on_one_node():
    _ = _init_multi(RoundRobin())
    sess = get_session()

    pg = sess.create_placement_group(
        bundles=[Bundle(cpus=0.0), Bundle(cpus=0.0)],
        strategy=PlacementStrategy.PACK,
    )

    @remote()
    def whoami():
        import os, platform
        return platform.node(), os.getpid()

    refs = [whoami.options(placement_group=pg).remote() for _ in range(6)]
    refs = [r for r in refs if r] + drain_queue()
    owners = [r.owner_node_id for r in refs]
    assert len(set(owners)) == 1


def test_pg_spread_remote_function_alternates_nodes():
    _ = _init_multi(RoundRobin())
    sess = get_session()

    pg = sess.create_placement_group(
        bundles=[Bundle(cpus=0.0), Bundle(cpus=0.0)],
        strategy=PlacementStrategy.SPREAD,
    )

    @remote()
    def whoami():
        import os, platform
        return platform.node(), os.getpid()

    refs = [
        whoami.options(placement_group=pg, bundle_index=(i % 2)).remote()
        for i in range(8)
    ]
    refs = [r for r in refs if r] + drain_queue()
    owners = [r.owner_node_id for r in refs]
    assert set(owners) == {"A", "B"}


def test_pg_pack_actors_same_node():
    _ = _init_multi(FIFO())
    sess = get_session()

    pg = sess.create_placement_group(
        bundles=[Bundle(cpus=0.0), Bundle(cpus=0.0)],
        strategy=PlacementStrategy.PACK,
    )

    @actor
    class Tag:
        def __init__(self, name):
            self.name = name

        def owner(self):
            return self.name

    h1 = get_val(Tag.options(placement_group=pg).remote("x"))
    h2 = get_val(Tag.options(placement_group=pg).remote("y"))

    assert h1.owner_node_id == h2.owner_node_id


def test_pg_spread_actors_across_nodes():
    _ = _init_multi(RoundRobin())
    sess = get_session()

    pg = sess.create_placement_group(
        bundles=[Bundle(cpus=0.0), Bundle(cpus=0.0)],
        strategy=PlacementStrategy.SPREAD,
    )

    @actor
    class Tag:
        def __init__(self, name):
            self.name = name

        def who(self):
            return self.name

    h1 = get_val(Tag.options(placement_group=pg, bundle_index=0).remote("a"))
    h2 = get_val(Tag.options(placement_group=pg, bundle_index=1).remote("b"))

    assert {h1.owner_node_id, h2.owner_node_id} == {"A", "B"}
