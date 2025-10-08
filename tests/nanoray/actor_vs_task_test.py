import argparse
import time
from typing import List

import torch

from nanorlhf.nanoray import init, shutdown, get, NodeConfig
from nanorlhf.nanoray.api.remote import remote, actor


@remote()
def stateless_step(x, n: int, dtype_name: str, iters: int = 1):
    import torch
    dtype = getattr(torch, dtype_name)
    W = torch.randn(n, n, dtype=dtype)
    y = x.to(dtype)
    for _ in range(iters):
        y = W @ y
    return float(y.sum().item())


@actor
class MatmulActor:
    def __init__(self, n: int, dtype_name: str, iters: int = 1):
        import torch
        self.n = n
        self.dtype = getattr(torch, dtype_name)
        self.iters = iters
        self.W = torch.randn(n, n, dtype=self.dtype)

    def run(self, x):
        y = x.to(self.dtype)
        for _ in range(self.iters):
            y = self.W @ y
        return float(y.sum().item())


def time_stateless(xs: List[torch.Tensor], n: int, dtype_name: str, iters: int):
    times, vals = [], []
    for x in xs:
        t0 = time.perf_counter()
        r = stateless_step.remote(x, n, dtype_name, iters)
        v = get(r)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        vals.append(v)
    return times, vals


def time_actor(actor_ref, xs: List[torch.Tensor]):
    times, vals = [], []
    for x in xs:
        t0 = time.perf_counter()
        r = actor_ref.run.remote(x)
        v = get(r)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        vals.append(v)
    return times, vals


def owners_stateless_preview(xs: List[torch.Tensor], n: int, dtype_name: str, iters: int, k: int = 8):
    owners = []
    for x in xs[:k]:
        r = stateless_step.remote(x, n, dtype_name, iters)
        owners.append(getattr(r, "owner_node_id", "local") or "local")
        _ = get(r)
    return owners


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["LOCAL", "REMOTE"], default="LOCAL")
    parser.add_argument("--n", type=int, default=2048, help="Matrix size (n x n)")
    parser.add_argument("--calls", type=int, default=16, help="Number of calls")
    parser.add_argument("--dtype", type=str, default="float32", help="torch dtype name")
    parser.add_argument("--iters", type=int, default=2, help="Repeated matmul count per call")
    parser.add_argument("--addrA", type=str, default="http://127.0.0.1:8003")
    parser.add_argument("--addrB", type=str, default="http://127.0.0.1:8004")
    parser.add_argument("--tokA", type=str, default="tA")
    parser.add_argument("--tokB", type=str, default="tB")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(0)

    n = args.n
    calls = args.calls
    dtype_name = args.dtype
    xs = [torch.randn(n, 1, dtype=getattr(torch, dtype_name)) for _ in range(calls)]

    if args.mode == "REMOTE":
        cfg = {
            "A": NodeConfig(local=False, address=args.addrA, remote_token=args.tokA, cpus=1.0),
            "B": NodeConfig(local=False, address=args.addrB, remote_token=args.tokB, cpus=1.0),
        }
        sess = init(cfg, default_node_id="A")
    else:
        sess = init()

    try:
        print(f"=== Actor vs. Task (heavy state) ===")
        print(f"mode={args.mode}, N={n}, calls={calls}, dtype={dtype_name}, iters={args.iters}")

        stateless_times, _ = time_stateless(xs, n, dtype_name, args.iters)
        Aref = get(MatmulActor.remote(n=n, dtype_name=dtype_name, iters=args.iters))
        actor_times, _ = time_actor(Aref, xs)

        mean_stateless = sum(stateless_times) / len(stateless_times)
        mean_actor = sum(actor_times) / len(actor_times)
        speedup = mean_stateless / max(mean_actor, 1e-12)

        print(f"Stateless mean: {mean_stateless:.4f}s  (per call)")
        print(f"Actor     mean: {mean_actor:.4f}s  (per call)")
        print(f"Speedup (stateless/actor): {speedup:.2f}×")

        def fmt_first(ts, k=6):
            return [f"{t:.4f}" for t in ts[:k]]

        print(f"First 6 stateless: {fmt_first(stateless_times)}")
        print(f"First 6 actor    : {fmt_first(actor_times)}")

        owners = owners_stateless_preview(xs, n, dtype_name, args.iters, k=min(8, calls))
        print(f"Owners (stateless short batch): {owners}")

    finally:
        shutdown()


if __name__ == "__main__":
    main()
