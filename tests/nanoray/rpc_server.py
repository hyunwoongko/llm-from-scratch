import argparse
import signal
import sys

from nanorlhf.nanoray import shutdown
from nanorlhf.nanoray.core.object_store import ObjectStore
from nanorlhf.nanoray.network.rpc_server import RpcServer
from nanorlhf.nanoray.runtime.worker import Worker


def main():
    ap = argparse.ArgumentParser("nanoray node")
    ap.add_argument("--node-id", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8003)
    # ap.add_argument("--port", type=int, default=8004)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    shutdown()
    print(f"Running server {args.node_id} on {args.host}:{args.port}")
    srv = RpcServer(args.node_id, Worker(ObjectStore(args.node_id)),
                    host=args.host, port=args.port, token=args.token)

    def stop(sig, frm):
        srv.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    srv.start()  # serve_forever()


if __name__ == "__main__":
    main()
