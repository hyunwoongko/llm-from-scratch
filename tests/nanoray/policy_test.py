from nanorlhf.nanoray.api.remote import remote
from nanorlhf.nanoray.api.session import drain as drain_queue
from nanorlhf.nanoray.api.session import init_session
from nanorlhf.nanoray.core.object_store import ObjectStore
from nanorlhf.nanoray.runtime.worker import Worker
from nanorlhf.nanoray.scheduler.policies import RoundRobin, FIFO


def _init_inprocess(policy):
    # 내부 API로 인프로세스 멀티노드 세션 구성 (A, B)
    nodes = {
        "A": (Worker(store=ObjectStore("A")), {"cpus": 1.0, "gpus": 0.0, "resources": {}}),
        "B": (Worker(store=ObjectStore("B")), {"cpus": 1.0, "gpus": 0.0, "resources": {}}),
    }
    return init_session(policy=policy, nodes=nodes, default_node_id="A")


def test_round_robin_owner_alternation_inprocess():
    _ = _init_inprocess(RoundRobin())

    @remote()
    def whoami(i):
        import os, platform
        return platform.node(), os.getpid(), i

    refs = [whoami.remote(i) for i in range(8)]
    refs = [r for r in refs if r] + drain_queue()
    owners = [f"{r.owner_node_id}" for r in refs]
    # RR이면 A/B 번갈아
    assert owners == ["A", "B", "A", "B", "A", "B", "A", "B"]


def test_fifo_keeps_submissions_on_first_node_inprocess():
    _ = _init_inprocess(FIFO())

    @remote()
    def whoami(i):
        import os, platform
        return platform.node(), os.getpid(), i

    refs = [whoami.remote(i) for i in range(8)]
    refs = [r for r in refs if r] + drain_queue()
    owners = [f"{r.owner_node_id}" for r in refs]
    # FIFO 정책 + 기본 배치 결과(첫 노드 고정)
    assert all(o == "A" for o in owners)
