from math import exp
from nanorlhf.nanoray import init, shutdown, get
from nanorlhf.nanoray.api.remote import actor


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


@actor
class RolloutWorker:
    def __init__(self, seed: int = 0):
        self.rng = (seed * 1103515245 + 12345) & 0x7FFFFFFF

    def _rand(self) -> float:
        self.rng = (1103515245 * self.rng + 12345) & 0x7FFFFFFF
        return (self.rng / 0x7FFFFFFF)

    def rollout(self, theta: float, steps: int = 64, episodes: int = 2):
        pi = sigmoid(theta)
        sum_grad = 0.0
        sum_r = 0.0
        T = steps * episodes
        for _ in range(T):
            a = 1 if self._rand() < pi else 0
            p_r = 0.8 if a == 1 else 0.2
            r = 1.0 if self._rand() < p_r else 0.0
            sum_grad += (a - pi) * r
            sum_r += r
        avg_ret = sum_r / T
        return {"sum_grad": sum_grad, "sum_r": sum_r, "steps": T, "avg_ret": avg_ret}

    def eval_return(self, theta: float, trials: int = 256):
        pi = sigmoid(theta)
        ret = 0.0
        for _ in range(trials):
            a = 1 if self._rand() < pi else 0
            p_r = 0.8 if a == 1 else 0.2
            r = 1.0 if self._rand() < p_r else 0.0
            ret += r
        return ret / trials


@actor
class Learner:
    def __init__(self, theta: float = 0.0):
        self.theta = theta

    def get_theta(self):
        return self.theta

    def apply(self, batch, lr: float = 0.2, theta_clip: float = 6.0):
        sum_grad = sum(item["sum_grad"] for item in batch)
        total_steps = sum(item["steps"] for item in batch)
        g = (sum_grad / max(1, total_steps))
        self.theta += lr * g
        self.theta = max(-theta_clip, min(theta_clip, self.theta))
        return self.theta


def main():
    _ = init()

    try:
        W1 = get(RolloutWorker.remote(seed=0))
        W2 = get(RolloutWorker.remote(seed=1))
        L = get(Learner.remote(theta=0.0))

        theta0 = get(L.get_theta.remote())
        pi0 = sigmoid(theta0)
        init_eval = get(W1.eval_return.remote(theta0))
        print(f"[init] theta={theta0:+.3f}  pi1={pi0:.3f}  eval={init_eval:.3f}")

        iters = 100
        steps = 64
        episodes = 2

        for it in range(1, iters + 1):
            th = get(L.get_theta.remote())
            r1 = W1.rollout.remote(theta=th, steps=steps, episodes=episodes)
            r2 = W2.rollout.remote(theta=th, steps=steps, episodes=episodes)
            b1, b2 = get(r1), get(r2)
            theta = get(L.apply.remote([b1, b2], lr=0.2, theta_clip=6.0))

            avg_ret = 0.5 * (b1["avg_ret"] + b2["avg_ret"])
            pi = sigmoid(theta)
            print(
                f"[iter {it:02d}] avg_ret={avg_ret:.3f}  "
                f"theta={theta:+.3f}  pi1={pi:.3f}  "
                f"(rollouts=2, episodes={episodes}, steps/ep={steps})"
            )

        theta_f = get(L.get_theta.remote())
        pi_f = sigmoid(theta_f)
        eval_f = get(W1.eval_return.remote(theta_f))
        print(f"[final] theta={theta_f:+.3f}  pi1={pi_f:.3f}  eval={eval_f:.3f}")

    finally:
        shutdown()


if __name__ == "__main__":
    main()
