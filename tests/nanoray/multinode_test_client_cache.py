# examples/test_cache_survives_without_networking.py
from nanorlhf.nanoray.api.initialization import init, shutdown, NodeConfig
from nanorlhf.nanoray.api.remote import remote
from nanorlhf.nanoray.api.session import drain, get, get_session
from nanorlhf.nanoray.scheduler.policies import RoundRobin

@remote(priority=0)
def who(i):
    import os, platform
    return {"host": platform.node(), "pid": os.getpid(), "i": i}

CFG = {
    # Give B a GPU so we can force placement with .options(num_gpus=1.0)
    "A": NodeConfig(local=False, address="http://127.0.0.1:8003", remote_token="tA", gpus=0.0),
    "B": NodeConfig(local=False, address="http://127.0.0.1:8004", remote_token="tB", gpus=1.0),
}

def main():
    sess = init(CFG, policy=RoundRobin(), default_node_id="A")
    try:
        # Force this task to run on B (single candidate) using GPU requirement
        r = who.options(num_gpus=1.0).remote(42)
        refs = ([r] if r else []) + drain()
        ref = refs[-1]

        # First get(): goes over RPC, then caches locally (alias map remote_id -> local_id)
        v1 = get(ref)

        # Simulate network outage: disable router/rpc on the session
        s = get_session()
        s._router = None
        s._rpc = None

        # Second get(): must be served from local cache (no networking)
        v2 = get(ref)
        print("values:", v1, v2)
        assert v1 == v2, "cache miss after disabling networking"
        print("OK: alias cache survives without networking.")
    finally:
        shutdown()


if __name__ == "__main__":
    main()

"""
values: {'host': 'kevinnlp-PC.local', 'pid': 4513, 'i': 42} {'host': 'kevinnlp-PC.local', 'pid': 4513, 'i': 42}
OK: alias cache survives without networking.
"""
