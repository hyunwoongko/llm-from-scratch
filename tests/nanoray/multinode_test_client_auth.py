# examples/test_auth_error_surface.py
from nanorlhf.nanoray.api.initialization import init, shutdown, NodeConfig
from nanorlhf.nanoray.api.remote import remote
from nanorlhf.nanoray.api.session import drain
from nanorlhf.nanoray.scheduler.policies import RoundRobin

@remote()
def ping(x): return x

CFG_BAD = {
    "A": NodeConfig(local=False, address="http://127.0.0.1:8003", remote_token="tA"),
    "B": NodeConfig(local=False, address="http://127.0.0.1:8004", remote_token="WRONG"),  # wrong token
}

def main():
    sess = init(CFG_BAD, policy=RoundRobin(), default_node_id="A")
    try:
        # With RoundRobin, one of these should land on B and trigger 401
        for i in range(4):
            _ = ping.remote(i)
        try:
            _ = drain()
            raise AssertionError("Expected RPC auth failure did not occur")
        except RuntimeError as e:
            msg = str(e)
            # Our RpcClient surfaces structured server errors when possible
            print("caught:", msg.splitlines()[0])
            assert ("401" in msg or "AuthError" in msg or "unauthorized" in msg.lower()), \
                "Expected auth-related error surface"
            print("OK: auth error surfaced as expected.")
    finally:
        shutdown()


if __name__ == "__main__":
    main()

"""
caught: RPC request failed to http://127.0.0.1:8004/rpc/execute_task: HTTP Error 401: Unauthorized
OK: auth error surfaced as expected.
"""