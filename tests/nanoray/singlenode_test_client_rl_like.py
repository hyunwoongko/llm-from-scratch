import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from nanorlhf.nanoray.api.initialization import init, shutdown, NodeConfig
from nanorlhf.nanoray.api.remote import remote
from nanorlhf.nanoray.api.session import drain, get
from nanorlhf.nanoray.scheduler.policies import RoundRobin


@dataclass(frozen=True)
class Policy:
    """
    Minimal Bernoulli policy over two actions (a ∈ {0,1}).

    Attributes:
        theta (float): Real-valued logit for P(a=1) = sigmoid(theta).

    Discussion:
        Q. Why such a tiny policy?
            This example is pedagogical: we only want to exercise the rollout
            and aggregation pipeline over nanoray. A single-parameter Bernoulli
            policy keeps the math trivial while still allowing REINFORCE-like
            gradients (∇ log π).

        Q. Why not use NumPy / torch here?
            Keeping the example pure-Python avoids GPU/BLAS dependencies and
            reduces serialization overhead over RPC.
    """
    theta: float

    @property
    def pi1(self) -> float:
        """Return π(a=1) = sigmoid(theta)."""
        return 1.0 / (1.0 + math.exp(-self.theta))


def sigmoid(x: float) -> float:
    """Numerically stable-ish sigmoid for small example."""
    return 1.0 / (1.0 + math.exp(-x))


# reward probability for actions 0 and 1
ENV_P: Tuple[float, float] = (0.2, 0.8)


@remote(priority=0)
def rollout_worker(policy: Policy, episodes: int, steps_per_ep: int, seed: int, baseline: float = 0.5) -> Dict[str, float]:
    """
    Simulate rollouts in a 2-armed bandit and return a REINFORCE-like gradient estimate.

    Args:
        policy (Policy): Bernoulli policy where P(a=1) = sigmoid(theta).
        episodes (int): Number of episodes to simulate.
        steps_per_ep (int): Steps per episode.
        seed (int): RNG seed to diversify parallel rollouts.

    Returns:
        Dict[str, float]: Aggregates across all steps:
            {
                "return_sum": float,   # total rewards
                "steps": float,        # total steps
                "grad_theta": float    # ∑ (r - b) * ∂ log π(a) / ∂theta
            }

    Discussion:
        Q. What gradient do we approximate?
            For a Bernoulli policy π(a=1)=σ(θ), log π gradients are:
              - If a=1: ∂ log π / ∂θ = (1 - π)
              - If a=0: ∂ log π / ∂θ = -π
            We sum (reward - baseline) * grad over steps.

        Q. Why return aggregates instead of per-step data?
            Reduces network payload and mirrors real RL systems where rollout
            workers pre-aggregate statistics to minimize transfer.
    """
    rng = random.Random(seed)
    pi1 = policy.pi1

    ret_sum = 0.0
    grad_sum = 0.0
    total_steps = 0

    for _ in range(episodes):
        for _ in range(steps_per_ep):
            # Sample action from policy
            a1 = 1 if (rng.random() < pi1) else 0
            # Environment reward
            reward = 1.0 if (rng.random() < ENV_P[a1]) else 0.0
            ret_sum += reward

            # REINFORCE gradient for Bernoulli(π1)
            grad_log = (1.0 - pi1) if a1 == 1 else (-pi1)
            grad_sum += (reward - baseline) * grad_log
            total_steps += 1

    return {"return_sum": ret_sum, "steps": float(total_steps), "grad_theta": grad_sum}


def learner_update(
    policy: Policy,
    grad_sums: List[float],
    step_counts: List[float],
    *,
    lr: float = 0.2,
    theta_clip: float = 6.0,
    entropy_beta: float = 0.01,
) -> Policy:
    """
    Apply a gentle SGD step with step-normalization and light entropy bonus.

    Args:
        policy: Current policy.
        grad_sums: Per-rollout sum of (reward - baseline) * dlogpi.
        step_counts: Per-rollout step counts (same length as grad_sums).
        lr: Learning rate (smaller than before).
        theta_clip: Clip range for theta to avoid hard saturation.
        entropy_beta: Small entropy bonus strength (0 disables it).

    Returns:
        Updated Policy.

    Discussion:
        Q. Why step-normalize?
            It makes the update scale invariant to the number of workers and steps,
            stabilizing training across different batch sizes.

        Q. Why clip theta?
            Prevents extreme logits from immediately saturating σ(θ) to 0/1,
            which would kill gradients and exploration.

        Q. Why a tiny entropy bonus?
            A small push toward π≈0.5 counteracts premature collapse while
            still letting rewards drive learning.
    """
    total_steps = sum(step_counts)
    if total_steps <= 0:
        return policy

    # average gradient per step
    g = (sum(grad_sums) / total_steps)

    # light entropy bonus toward 0.5
    pi1 = policy.pi1
    g += entropy_beta * (0.5 - pi1)

    new_theta = policy.theta + lr * g
    new_theta = max(-theta_clip, min(theta_clip, new_theta))
    return Policy(theta=new_theta)


def evaluate(policy: Policy, episodes: int = 50, steps_per_ep: int = 50, seed: int = 123) -> float:
    """
    Do a local, deterministic evaluation (no RPC) to estimate average reward.

    Args:
        policy (Policy): Policy to evaluate.
        episodes (int): Episodes to simulate.
        steps_per_ep (int): Steps per episode.
        seed (int): RNG seed.

    Returns:
        float: Average reward per step.
    """
    rng = random.Random(seed)
    pi1 = policy.pi1

    ret = 0.0
    steps = episodes * steps_per_ep
    for _ in range(steps):
        a1 = 1 if (rng.random() < pi1) else 0
        r = 1.0 if (rng.random() < ENV_P[a1]) else 0.0
        ret += r
    return ret / float(steps)


def main() -> None:
    """
    Orchestrate a rollout → aggregate → learner update loop over multiple nodes.

    - Two local nodes (no GPUs needed).
    - RoundRobin policy to spread rollouts across nodes.
    - Mock REINFORCE update on the driver.
    """
    # 1) Start a small multi-node runtime (local workers; CPU-only)
    cfg = {
        "A": NodeConfig(local=True),
        "B": NodeConfig(local=True),
    }
    _ = init(cfg, policy=RoundRobin(), default_node_id="A")

    try:
        policy = Policy(theta=0.0)  # π(a=1)=0.5
        print(f"[init] theta={policy.theta:.3f}  pi1={policy.pi1:.3f}  eval={evaluate(policy):.3f}")

        iters = 100
        rollouts_per_iter = 8
        episodes = 2
        steps_per_ep = 64

        avg_ret_baseline = 0.5
        for it in range(1, iters + 1):
            # 2) Submit rollouts (enqueue-only); seeds diversify workers
            for k in range(rollouts_per_iter):
                seed = 1000 * it + k
                _ = rollout_worker.remote(policy, episodes, steps_per_ep, seed=seed, baseline=avg_ret_baseline)

            # 3) Run and collect
            refs = drain()
            outs = [get(r) for r in refs]
            total_steps = sum(o["steps"] for o in outs)
            avg_ret = (sum(o["return_sum"] for o in outs) / total_steps) if total_steps else 0.0
            grad_sums = [o["grad_theta"] for o in outs]
            step_counts = [o["steps"] for o in outs]

            policy = learner_update(policy, grad_sums, step_counts, lr=0.3, theta_clip=6.0, entropy_beta=0.0)
            avg_ret_baseline = avg_ret

            print(
                f"[iter {it:02d}] avg_ret={avg_ret:.3f}  theta={policy.theta:+.3f}  pi1={policy.pi1:.3f}  "
                f"(rollouts={len(outs)}, episodes={episodes}, steps/ep={steps_per_ep})"
            )

        print(f"[final] theta={policy.theta:+.3f}  pi1={policy.pi1:.3f}  eval={evaluate(policy):.3f}")

    finally:
        shutdown()


if __name__ == "__main__":
    main()

"""
[init] theta=0.000  pi1=0.500  eval=0.510
[iter 01] avg_ret=0.496  theta=+0.046  pi1=0.512  (rollouts=8, episodes=2, steps/ep=64)
[iter 02] avg_ret=0.491  theta=+0.091  pi1=0.523  (rollouts=8, episodes=2, steps/ep=64)
[iter 03] avg_ret=0.487  theta=+0.134  pi1=0.533  (rollouts=8, episodes=2, steps/ep=64)
[iter 04] avg_ret=0.507  theta=+0.177  pi1=0.544  (rollouts=8, episodes=2, steps/ep=64)
[iter 05] avg_ret=0.554  theta=+0.221  pi1=0.555  (rollouts=8, episodes=2, steps/ep=64)
[iter 06] avg_ret=0.548  theta=+0.265  pi1=0.566  (rollouts=8, episodes=2, steps/ep=64)
[iter 07] avg_ret=0.546  theta=+0.311  pi1=0.577  (rollouts=8, episodes=2, steps/ep=64)
[iter 08] avg_ret=0.560  theta=+0.355  pi1=0.588  (rollouts=8, episodes=2, steps/ep=64)
[iter 09] avg_ret=0.543  theta=+0.398  pi1=0.598  (rollouts=8, episodes=2, steps/ep=64)
[iter 10] avg_ret=0.512  theta=+0.444  pi1=0.609  (rollouts=8, episodes=2, steps/ep=64)
[iter 11] avg_ret=0.569  theta=+0.488  pi1=0.620  (rollouts=8, episodes=2, steps/ep=64)
[iter 12] avg_ret=0.583  theta=+0.529  pi1=0.629  (rollouts=8, episodes=2, steps/ep=64)
[iter 13] avg_ret=0.570  theta=+0.572  pi1=0.639  (rollouts=8, episodes=2, steps/ep=64)
[iter 14] avg_ret=0.595  theta=+0.614  pi1=0.649  (rollouts=8, episodes=2, steps/ep=64)
[iter 15] avg_ret=0.590  theta=+0.654  pi1=0.658  (rollouts=8, episodes=2, steps/ep=64)
[iter 16] avg_ret=0.602  theta=+0.695  pi1=0.667  (rollouts=8, episodes=2, steps/ep=64)
[iter 17] avg_ret=0.598  theta=+0.735  pi1=0.676  (rollouts=8, episodes=2, steps/ep=64)
[iter 18] avg_ret=0.622  theta=+0.773  pi1=0.684  (rollouts=8, episodes=2, steps/ep=64)
[iter 19] avg_ret=0.612  theta=+0.810  pi1=0.692  (rollouts=8, episodes=2, steps/ep=64)
[iter 20] avg_ret=0.609  theta=+0.849  pi1=0.700  (rollouts=8, episodes=2, steps/ep=64)
[iter 21] avg_ret=0.635  theta=+0.891  pi1=0.709  (rollouts=8, episodes=2, steps/ep=64)
[iter 22] avg_ret=0.636  theta=+0.927  pi1=0.716  (rollouts=8, episodes=2, steps/ep=64)
[iter 23] avg_ret=0.638  theta=+0.964  pi1=0.724  (rollouts=8, episodes=2, steps/ep=64)
[iter 24] avg_ret=0.629  theta=+1.001  pi1=0.731  (rollouts=8, episodes=2, steps/ep=64)
[iter 25] avg_ret=0.649  theta=+1.037  pi1=0.738  (rollouts=8, episodes=2, steps/ep=64)
[iter 26] avg_ret=0.634  theta=+1.075  pi1=0.746  (rollouts=8, episodes=2, steps/ep=64)
[iter 27] avg_ret=0.652  theta=+1.109  pi1=0.752  (rollouts=8, episodes=2, steps/ep=64)
[iter 28] avg_ret=0.647  theta=+1.145  pi1=0.759  (rollouts=8, episodes=2, steps/ep=64)
[iter 29] avg_ret=0.659  theta=+1.180  pi1=0.765  (rollouts=8, episodes=2, steps/ep=64)
[iter 30] avg_ret=0.673  theta=+1.212  pi1=0.771  (rollouts=8, episodes=2, steps/ep=64)
[iter 31] avg_ret=0.667  theta=+1.245  pi1=0.777  (rollouts=8, episodes=2, steps/ep=64)
[iter 32] avg_ret=0.666  theta=+1.274  pi1=0.781  (rollouts=8, episodes=2, steps/ep=64)
[iter 33] avg_ret=0.652  theta=+1.302  pi1=0.786  (rollouts=8, episodes=2, steps/ep=64)
[iter 34] avg_ret=0.672  theta=+1.332  pi1=0.791  (rollouts=8, episodes=2, steps/ep=64)
[iter 35] avg_ret=0.684  theta=+1.360  pi1=0.796  (rollouts=8, episodes=2, steps/ep=64)
[iter 36] avg_ret=0.687  theta=+1.385  pi1=0.800  (rollouts=8, episodes=2, steps/ep=64)
[iter 37] avg_ret=0.686  theta=+1.414  pi1=0.804  (rollouts=8, episodes=2, steps/ep=64)
[iter 38] avg_ret=0.689  theta=+1.444  pi1=0.809  (rollouts=8, episodes=2, steps/ep=64)
[iter 39] avg_ret=0.704  theta=+1.472  pi1=0.813  (rollouts=8, episodes=2, steps/ep=64)
[iter 40] avg_ret=0.707  theta=+1.498  pi1=0.817  (rollouts=8, episodes=2, steps/ep=64)
[iter 41] avg_ret=0.689  theta=+1.524  pi1=0.821  (rollouts=8, episodes=2, steps/ep=64)
[iter 42] avg_ret=0.727  theta=+1.550  pi1=0.825  (rollouts=8, episodes=2, steps/ep=64)
[iter 43] avg_ret=0.705  theta=+1.571  pi1=0.828  (rollouts=8, episodes=2, steps/ep=64)
[iter 44] avg_ret=0.723  theta=+1.597  pi1=0.832  (rollouts=8, episodes=2, steps/ep=64)
[iter 45] avg_ret=0.702  theta=+1.619  pi1=0.835  (rollouts=8, episodes=2, steps/ep=64)
[iter 46] avg_ret=0.698  theta=+1.645  pi1=0.838  (rollouts=8, episodes=2, steps/ep=64)
[iter 47] avg_ret=0.714  theta=+1.673  pi1=0.842  (rollouts=8, episodes=2, steps/ep=64)
[iter 48] avg_ret=0.721  theta=+1.697  pi1=0.845  (rollouts=8, episodes=2, steps/ep=64)
[iter 49] avg_ret=0.702  theta=+1.721  pi1=0.848  (rollouts=8, episodes=2, steps/ep=64)
[iter 50] avg_ret=0.722  theta=+1.742  pi1=0.851  (rollouts=8, episodes=2, steps/ep=64)
[iter 51] avg_ret=0.710  theta=+1.765  pi1=0.854  (rollouts=8, episodes=2, steps/ep=64)
[iter 52] avg_ret=0.701  theta=+1.788  pi1=0.857  (rollouts=8, episodes=2, steps/ep=64)
[iter 53] avg_ret=0.710  theta=+1.811  pi1=0.859  (rollouts=8, episodes=2, steps/ep=64)
[iter 54] avg_ret=0.704  theta=+1.834  pi1=0.862  (rollouts=8, episodes=2, steps/ep=64)
[iter 55] avg_ret=0.722  theta=+1.859  pi1=0.865  (rollouts=8, episodes=2, steps/ep=64)
[iter 56] avg_ret=0.714  theta=+1.879  pi1=0.867  (rollouts=8, episodes=2, steps/ep=64)
[iter 57] avg_ret=0.739  theta=+1.897  pi1=0.870  (rollouts=8, episodes=2, steps/ep=64)
[iter 58] avg_ret=0.731  theta=+1.918  pi1=0.872  (rollouts=8, episodes=2, steps/ep=64)
[iter 59] avg_ret=0.732  theta=+1.941  pi1=0.874  (rollouts=8, episodes=2, steps/ep=64)
[iter 60] avg_ret=0.703  theta=+1.961  pi1=0.877  (rollouts=8, episodes=2, steps/ep=64)
[iter 61] avg_ret=0.705  theta=+1.977  pi1=0.878  (rollouts=8, episodes=2, steps/ep=64)
[iter 62] avg_ret=0.737  theta=+1.997  pi1=0.880  (rollouts=8, episodes=2, steps/ep=64)
[iter 63] avg_ret=0.739  theta=+2.018  pi1=0.883  (rollouts=8, episodes=2, steps/ep=64)
[iter 64] avg_ret=0.712  theta=+2.042  pi1=0.885  (rollouts=8, episodes=2, steps/ep=64)
[iter 65] avg_ret=0.743  theta=+2.057  pi1=0.887  (rollouts=8, episodes=2, steps/ep=64)
[iter 66] avg_ret=0.722  theta=+2.077  pi1=0.889  (rollouts=8, episodes=2, steps/ep=64)
[iter 67] avg_ret=0.736  theta=+2.093  pi1=0.890  (rollouts=8, episodes=2, steps/ep=64)
[iter 68] avg_ret=0.743  theta=+2.112  pi1=0.892  (rollouts=8, episodes=2, steps/ep=64)
[iter 69] avg_ret=0.756  theta=+2.130  pi1=0.894  (rollouts=8, episodes=2, steps/ep=64)
[iter 70] avg_ret=0.729  theta=+2.148  pi1=0.896  (rollouts=8, episodes=2, steps/ep=64)
[iter 71] avg_ret=0.727  theta=+2.166  pi1=0.897  (rollouts=8, episodes=2, steps/ep=64)
[iter 72] avg_ret=0.759  theta=+2.181  pi1=0.899  (rollouts=8, episodes=2, steps/ep=64)
[iter 73] avg_ret=0.741  theta=+2.196  pi1=0.900  (rollouts=8, episodes=2, steps/ep=64)
[iter 74] avg_ret=0.738  theta=+2.211  pi1=0.901  (rollouts=8, episodes=2, steps/ep=64)
[iter 75] avg_ret=0.736  theta=+2.223  pi1=0.902  (rollouts=8, episodes=2, steps/ep=64)
[iter 76] avg_ret=0.755  theta=+2.239  pi1=0.904  (rollouts=8, episodes=2, steps/ep=64)
[iter 77] avg_ret=0.738  theta=+2.257  pi1=0.905  (rollouts=8, episodes=2, steps/ep=64)
[iter 78] avg_ret=0.738  theta=+2.273  pi1=0.907  (rollouts=8, episodes=2, steps/ep=64)
[iter 79] avg_ret=0.760  theta=+2.289  pi1=0.908  (rollouts=8, episodes=2, steps/ep=64)
[iter 80] avg_ret=0.763  theta=+2.307  pi1=0.909  (rollouts=8, episodes=2, steps/ep=64)
[iter 81] avg_ret=0.735  theta=+2.322  pi1=0.911  (rollouts=8, episodes=2, steps/ep=64)
[iter 82] avg_ret=0.713  theta=+2.337  pi1=0.912  (rollouts=8, episodes=2, steps/ep=64)
[iter 83] avg_ret=0.750  theta=+2.352  pi1=0.913  (rollouts=8, episodes=2, steps/ep=64)
[iter 84] avg_ret=0.736  theta=+2.368  pi1=0.914  (rollouts=8, episodes=2, steps/ep=64)
[iter 85] avg_ret=0.764  theta=+2.382  pi1=0.915  (rollouts=8, episodes=2, steps/ep=64)
[iter 86] avg_ret=0.767  theta=+2.395  pi1=0.916  (rollouts=8, episodes=2, steps/ep=64)
[iter 87] avg_ret=0.750  theta=+2.412  pi1=0.918  (rollouts=8, episodes=2, steps/ep=64)
[iter 88] avg_ret=0.755  theta=+2.425  pi1=0.919  (rollouts=8, episodes=2, steps/ep=64)
[iter 89] avg_ret=0.742  theta=+2.438  pi1=0.920  (rollouts=8, episodes=2, steps/ep=64)
[iter 90] avg_ret=0.743  theta=+2.452  pi1=0.921  (rollouts=8, episodes=2, steps/ep=64)
[iter 91] avg_ret=0.753  theta=+2.465  pi1=0.922  (rollouts=8, episodes=2, steps/ep=64)
[iter 92] avg_ret=0.788  theta=+2.477  pi1=0.923  (rollouts=8, episodes=2, steps/ep=64)
[iter 93] avg_ret=0.744  theta=+2.489  pi1=0.923  (rollouts=8, episodes=2, steps/ep=64)
[iter 94] avg_ret=0.758  theta=+2.500  pi1=0.924  (rollouts=8, episodes=2, steps/ep=64)
[iter 95] avg_ret=0.762  theta=+2.514  pi1=0.925  (rollouts=8, episodes=2, steps/ep=64)
[iter 96] avg_ret=0.754  theta=+2.526  pi1=0.926  (rollouts=8, episodes=2, steps/ep=64)
[iter 97] avg_ret=0.763  theta=+2.538  pi1=0.927  (rollouts=8, episodes=2, steps/ep=64)
[iter 98] avg_ret=0.742  theta=+2.552  pi1=0.928  (rollouts=8, episodes=2, steps/ep=64)
[iter 99] avg_ret=0.757  theta=+2.565  pi1=0.929  (rollouts=8, episodes=2, steps/ep=64)
[iter 100] avg_ret=0.764  theta=+2.575  pi1=0.929  (rollouts=8, episodes=2, steps/ep=64)
[final] theta=+2.575  pi1=0.929  eval=0.760
"""
