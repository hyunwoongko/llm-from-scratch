import pytest

from nanorlhf.nanoray import init, shutdown, get
from nanorlhf.nanoray.api.remote import remote


@pytest.fixture(autouse=True)
def _single_node_session():
    # Public API: 싱글노드 초기화
    _ = init()
    try:
        yield
    finally:
        shutdown()


def test_remote_function_basic_single():
    @remote()
    def add(x, y):
        return x + y

    refs = [add.remote(i, i) for i in range(5)]
    vals = [get(r) for r in refs]
    assert vals == [0, 2, 4, 6, 8]


def test_get_drives_scheduling_single():
    # submit/queue 개념 없이 remote() 는 항상 즉시 ObjectRef를 반환하고,
    # get() 이 알아서 스케줄링을 진행해 값을 가져온다.
    @remote()
    def square(x):
        return x * x

    refs = [square.remote(i) for i in range(5)]
    vals = [get(r) for r in refs]
    assert vals == [0, 1, 4, 9, 16]
