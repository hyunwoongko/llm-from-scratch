from nanorlhf.nanoray.api.initialization import init, shutdown, NodeConfig
from nanorlhf.nanoray.api.remote import remote
from nanorlhf.nanoray.api.session import get, drain

cfg = {
    "A": NodeConfig(local=False, address="http://127.0.0.1:8003", remote_token="tA", cpus=1.0),
    "B": NodeConfig(local=False, address="http://127.0.0.1:8004", remote_token="tB", cpus=2.0),
}
init(cfg, default_node_id="A")


@remote(priority=0)
def tag(x): return "p0", x


@remote(priority=5)
def tag_mid(x): return "p5", x


@remote(priority=10)
def tag_hi(x): return "p10", x


for i in range(3):
    _ = tag.remote(i)  # enqueued
    _ = tag_mid.remote(i)  # enqueued
    _ = tag_hi.remote(i)  # enqueued
refs = drain()
print([get(r)[0] for r in refs])
shutdown()


"""
['p10', 'p10', 'p10', 'p5', 'p5', 'p5', 'p0', 'p0', 'p0']
"""