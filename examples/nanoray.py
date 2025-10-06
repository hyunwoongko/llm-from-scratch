from collections import Counter
from typing import Tuple

from nanorlhf import nanoray


@nanoray.remote()  # default priority=0
def add(a: int, b: int) -> int:
    """Simple remote function."""
    return a + b


@nanoray.remote(priority=10)
def tag_hi(x: int) -> Tuple[str, int]:
    """High priority task to demonstrate queue ordering."""
    return "p10", x


@nanoray.remote(priority=5)
def tag_mid(x: int) -> Tuple[str, int]:
    """Medium priority task."""
    return "p5", x


@nanoray.remote(priority=0)
def tag_lo(x: int) -> Tuple[str, int]:
    """Low priority task."""
    return "p0", x


def quickstart_single_node():
    """
    Single-node session:
      - init() with defaults
      - put/get convenience
      - submit @remote calls, drain, and fetch results
    """
    print("=== 1) Single-node quickstart ===")
    _ = nanoray.init()  # default: one local worker, default policy

    # put/get
    ref = nanoray.put({"hello": "world"})
    print("put/get ->", nanoray.get(ref))

    # remote calls (enqueue-only), then drain() to execute
    refs = [add.remote(i, i) for i in range(5)]
    refs = [r for r in refs if r] + nanoray.drain()  # if any None returned on submit, drain() completes them
    vals = [nanoray.get(r) for r in refs]
    print("add results:", vals)

    nanoray.shutdown()
    print()


def priority_demo():
    """
    Show that drain() pops by (-priority, seq):
      - all p10 first, then p5, then p0
      - FIFO within the same priority
    """
    print("=== 2) Priority-aware queue (single node) ===")
    _ = nanoray.init()  # single local node

    # Interleave submissions on purpose
    for i in range(3):
        _ = tag_lo.remote(i)
        _ = tag_mid.remote(i)
        _ = tag_hi.remote(i)

    refs = nanoray.drain()
    vals = [nanoray.get(r) for r in refs]
    heads = [v[0] for v in vals]
    print("values:", vals)
    print("priority heads:", heads)  # expected: ['p10','p10','p10','p5','p5','p5','p0','p0','p0']

    nanoray.shutdown()
    print()


@nanoray.remote()
def whoami(i: int):
    """
    Return a small tuple so we can see which node executed the task.
    We inspect ObjectRef.owner_node_id from the returned refs for placement.
    """
    import os, platform
    return platform.node(), os.getpid(), i


def multinode_local_round_robin():
    """
    Bring up two local workers (A, B). With equal priorities, the default policy
    spreads tasks across nodes. We print owner_node_id per result.
    """
    print("=== 3) Multi-node local (RoundRobin default) ===")
    cfg = {
        "A": nanoray.NodeConfig(local=True, cpus=1.0),
        "B": nanoray.NodeConfig(local=True, cpus=1.0),
    }
    _ = nanoray.init(cfg)  # default policy is applied internally

    # Enqueue several identical-priority tasks
    refs = [whoami.remote(i) for i in range(8)]
    refs = [r for r in refs if r] + nanoray.drain()

    owners = [r.owner_node_id for r in refs]  # reading public fields on the ref is fine
    vals = [nanoray.get(r) for r in refs]  # (hostname, pid, i)

    print("owners:", owners)
    print("pid stream (first 6):", [v[1] for v in vals[:6]])
    print("owner counts:", Counter(owners))  # expect near-balanced counts like {'A':4,'B':4}

    nanoray.shutdown()
    print()


def main():
    quickstart_single_node()
    priority_demo()
    multinode_local_round_robin()
    print("Done.")


if __name__ == "__main__":
    main()
