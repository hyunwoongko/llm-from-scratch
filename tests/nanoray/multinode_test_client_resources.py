from nanorlhf.nanoray.api.initialization import init, shutdown, NodeConfig
from nanorlhf.nanoray.api.remote import remote
from nanorlhf.nanoray.api.session import drain, get
from nanorlhf.nanoray.scheduler.policies import RoundRobin, FIFO


@remote(priority=0)
def who(tag):
    """
    Return node/process signature with a tag to track placement.
    """
    import os, platform
    return {"host": platform.node(), "pid": os.getpid(), "tag": tag}


CFG = {
    # A: no GPUs, B: has 1 GPU (from the scheduler's perspective)
    "A": NodeConfig(local=False, address="http://127.0.0.1:8003", remote_token="tA", cpus=1.0, gpus=0.0),
    "B": NodeConfig(local=False, address="http://127.0.0.1:8004", remote_token="tB", cpus=1.0, gpus=1.0),
}


def run(policy):
    """
    Submit interleaved CPU/GPU tasks with equal priority and compare placement.
    """
    sess = init(CFG, policy=policy, default_node_id="A")
    try:
        refs = []
        # Interleave: CPU, GPU, CPU, GPU, ...
        for i in range(8):
            refs.append(who.remote(f"cpu-{i}"))  # CPU-only (eligible: A,B)
            refs.append(who.options(num_gpus=1.0).remote(f"gpu-{i}"))  # needs 1 GPU (eligible: B only)

        refs = [r for r in refs if r] + drain()
        vals = [get(r) for r in refs]
        owners = [r.owner_node_id for r in refs]

        # Split back into cpu/gpu buckets (by tag prefix)
        cpu_idx = [i for i, v in enumerate(vals) if v["tag"].startswith("cpu-")]
        gpu_idx = [i for i, v in enumerate(vals) if v["tag"].startswith("gpu-")]

        owners_cpu = [owners[i] for i in cpu_idx]
        owners_gpu = [owners[i] for i in gpu_idx]

        return owners_cpu, owners_gpu, vals
    finally:
        shutdown()


# Run both policies with identical workload
owners_cpu_rr, owners_gpu_rr, vals_rr = run(RoundRobin())
owners_cpu_fifo, owners_gpu_fifo, vals_fifo = run(FIFO())

print("RoundRobin CPU owners:", owners_cpu_rr)
print("RoundRobin GPU owners:", owners_gpu_rr)
print("FIFO CPU owners     :", owners_cpu_fifo)
print("FIFO GPU owners     :", owners_gpu_fifo)

# Quick assertions (raise if something is off)
# 1) GPU tasks must always land on B due to eligibility
assert set(owners_gpu_rr) == {"B"}, f"RR: GPU tasks should run on B only, got {owners_gpu_rr}"
assert set(owners_gpu_fifo) == {"B"}, f"FIFO: GPU tasks should run on B only, got {owners_gpu_fifo}"

# 2) Policy difference should show up on CPU tasks:
#    - RoundRobin: should use both A and B
#    - FIFO: should prefer the first eligible node (A) only
assert set(owners_cpu_rr) == {"A", "B"}, f"RR: CPU tasks should alternate across nodes, got {owners_cpu_rr}"
assert set(owners_cpu_fifo) in ({"A"}, {"B"}), f"FIFO: CPU tasks should stick to a single node, got {owners_cpu_fifo}"

# Optional: print PID mix to visualize remote processes
print("RR PIDs (first 6):", [v["pid"] for v in vals_rr[:6]])
print("FIFO PIDs (first 6):", [v["pid"] for v in vals_fifo[:6]])


"""
RoundRobin CPU owners: ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
RoundRobin GPU owners: ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']
FIFO CPU owners     : ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A']
FIFO GPU owners     : ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']
RR PIDs (first 6): [4502, 4513, 4513, 4513, 4502, 4513]
FIFO PIDs (first 6): [4502, 4513, 4502, 4513, 4502, 4513]
"""