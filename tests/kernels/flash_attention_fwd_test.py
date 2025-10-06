import time

import torch

from nanorlhf.kernels.flash_attention.fwd import flash_attn_fwd


def naive_attention_torch(q, k, v, causal=True, softmax_scale=None):
    bsz, num_heads, seq_len_q, dim_head = q.shape
    seq_len_kv = k.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / (dim_head ** 0.5)

    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    if causal:
        mask = torch.triu(torch.ones(seq_len_q, seq_len_kv, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


def sdpa_torch(q, k, v, causal=True, softmax_scale=None):
    bsz, num_heads, seq_len_q, dim_head = q.shape
    seq_len_kv = k.shape[2]
    q_merged = q.reshape(bsz * num_heads, seq_len_q, dim_head)
    k_merged = k.reshape(bsz * num_heads, seq_len_kv, dim_head)
    v_merged = v.reshape(bsz * num_heads, seq_len_kv, dim_head)

    if softmax_scale is None:
        softmax_scale = 1.0 / (dim_head ** 0.5)

    out = torch.nn.functional.scaled_dot_product_attention(
        q_merged, k_merged, v_merged,
        attn_mask=None, dropout_p=0.0, is_causal=causal, scale=softmax_scale,
    )

    return out.reshape(bsz, num_heads, seq_len_q, dim_head)


@torch.inference_mode()
def check_correctness():
    torch.manual_seed(7)
    cases = [
        # (bsz, num_heads, seq_len_q, seq_len_kv, dim_head, causal)
        (2, 8, 5, 5, 64, True),
        (2, 8, 5, 128, 64, True),
        (2, 8, 64, 64, 128, False),
    ]

    for (bsz, num_heads, seq_len_q, seq_len_kv, dim_head, causal) in cases:
        q = torch.randn(bsz, num_heads, seq_len_q, dim_head, device="cuda", dtype=torch.float32)
        k = torch.randn(bsz, num_heads, seq_len_kv, dim_head, device="cuda", dtype=torch.float32)
        v = torch.randn(bsz, num_heads, seq_len_kv, dim_head, device="cuda", dtype=torch.float32)

        out_triton = flash_attn_fwd(q, k, v, causal=causal)
        out_sdpa = sdpa_torch(q, k, v, causal=causal)
        out_naive = naive_attention_torch(q, k, v, causal=causal)

        def stats(a, b):
            diff = (a - b).float().abs()
            return diff.max().item(), diff.mean().item(), (diff / (b.float().abs().clamp_min(1e-6))).mean().item()

        m_ts, mae_ts, rel_ts = stats(out_triton, out_sdpa)
        m_ns, mae_ns, rel_ns = stats(out_naive, out_sdpa)
        m_tn, mae_tn, rel_tn = stats(out_triton, out_naive)

        print(f"[bsz={bsz} heads={num_heads} q_len={seq_len_q} kv_len={seq_len_kv} d_head={dim_head} causal={causal}]")
        print(f"  Triton vs SDPA : max_abs={m_ts:.3e}  mae={mae_ts:.3e}  rel_mean={rel_ts:.3e}")
        print(f"  Naive  vs SDPA : max_abs={m_ns:.3e}  mae={mae_ns:.3e}  rel_mean={rel_ns:.3e}")
        print(f"  Triton vs Naive: max_abs={m_tn:.3e}  mae={mae_tn:.3e}  rel_mean={rel_tn:.3e}")


@torch.inference_mode()
def benchmark(warmup=10, iters=50):
    torch.manual_seed(7)
    configs = [
        ("inference-like", (16, 64, 1, 4096, 128, True)),
        ("training-like", (16, 64, 1024, 1024, 128, True)),
    ]

    for name, (bsz, num_heads, q_len, kv_len, d_head, causal) in configs:
        q = torch.randn(bsz, num_heads, q_len, d_head, device="cuda", dtype=torch.float32)
        k = torch.randn(bsz, num_heads, kv_len, d_head, device="cuda", dtype=torch.float32)
        v = torch.randn(bsz, num_heads, kv_len, d_head, device="cuda", dtype=torch.float32)

        for _ in range(warmup):
            _ = flash_attn_fwd(q, k, v, causal=causal)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = flash_attn_fwd(q, k, v, causal=causal)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - t0) * 1000 / iters

        for _ in range(warmup):
            _ = sdpa_torch(q, k, v, causal=causal)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = sdpa_torch(q, k, v, causal=causal)
        torch.cuda.synchronize()
        sdpa_ms = (time.perf_counter() - t0) * 1000 / iters

        for _ in range(warmup):
            _ = naive_attention_torch(q, k, v, causal=causal)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = naive_attention_torch(q, k, v, causal=causal)
        torch.cuda.synchronize()
        naive_ms = (time.perf_counter() - t0) * 1000 / iters

        print(
            f"[{name}] bsz={bsz} heads={num_heads} q_len={q_len} kv_len={kv_len} d_head={d_head} causal={causal} "
            f"=> Triton: {triton_ms:.2f} ms | SDPA: {sdpa_ms:.2f} ms | Naive: {naive_ms:.2f} ms"
        )


if __name__ == "__main__":
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    print("== Correctness ==")
    check_correctness()

    print("\n== Benchmark ==")
    benchmark(warmup=10, iters=50)

"""
== Correctness ==
[bsz=2 heads=8 q_len=5 kv_len=5 d_head=64 causal=True]
  Triton vs SDPA : max_abs=2.131e-03  mae=3.325e-04  rel_mean=1.219e-03
  Naive  vs SDPA : max_abs=4.768e-07  mae=2.849e-08  rel_mean=3.102e-07
  Triton vs Naive: max_abs=2.131e-03  mae=3.325e-04  rel_mean=1.219e-03
[bsz=2 heads=8 q_len=5 kv_len=128 d_head=64 causal=True]
  Triton vs SDPA : max_abs=3.160e-03  mae=3.466e-04  rel_mean=3.441e-03
  Naive  vs SDPA : max_abs=1.698e-03  mae=1.144e-04  rel_mean=1.970e-03
  Triton vs Naive: max_abs=3.196e-03  mae=3.576e-04  rel_mean=1.923e-03
[bsz=2 heads=8 q_len=64 kv_len=64 d_head=128 causal=False]
  Triton vs SDPA : max_abs=2.542e-03  mae=2.081e-04  rel_mean=6.008e-03
  Naive  vs SDPA : max_abs=7.580e-04  mae=7.453e-05  rel_mean=3.437e-03
  Triton vs Naive: max_abs=2.486e-03  mae=2.132e-04  rel_mean=6.943e-03

== Benchmark ==
[inference-like] bsz=16 heads=64 q_len=1 kv_len=4096 d_head=128 causal=True => Triton: 1.64 ms | SDPA: 2.73 ms | Naive: 1.69 ms
[training-like] bsz=16 heads=64 q_len=1024 kv_len=1024 d_head=128 causal=True => Triton: 9.56 ms | SDPA: 13.67 ms | Naive: 12.11 ms
"""