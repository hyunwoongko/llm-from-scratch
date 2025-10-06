from nanorlhf.nanoray.api.initialization import init, shutdown, NodeConfig
from nanorlhf.nanoray.api.remote import remote
from nanorlhf.nanoray.api.session import drain, get
from nanorlhf.nanoray.scheduler.policies import RoundRobin, FIFO


@remote(priority=0)
def whoami(i):
    import os, platform
    return platform.node(), os.getpid(), i


CFG = {
    "A": NodeConfig(local=False, address="http://127.0.0.1:8003", remote_token="tA"),
    "B": NodeConfig(local=False, address="http://127.0.0.1:8004", remote_token="tB"),
}


def run(policy):
    init(CFG, policy=policy, default_node_id="A")
    try:
        for i in range(8):
            _ = whoami.remote(i)
        refs = drain()
        vals = [get(r) for r in refs]  # each is (hostname, pid, i)
        owners = [r.owner_node_id for r in refs]
        return owners, vals
    finally:
        shutdown()


owners_rr, vals_rr = run(RoundRobin())
owners_fifo, vals_fifo = run(FIFO())

print("RoundRobin owners:", owners_rr)
print("FIFO owners     :", owners_fifo)

# Distinct PIDs/hosts prove remote execution:
print("RoundRobin PIDs:", [v[1] for v in vals_rr])
print("FIFO PIDs     :", [v[1] for v in vals_fifo])

"""
RoundRobin owners: ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
FIFO owners     : ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A']
RoundRobin PIDs: [4502, 4513, 4502, 4513, 4502, 4513, 4502, 4513]
FIFO PIDs     : [4502, 4502, 4502, 4502, 4502, 4502, 4502, 4502]
"""