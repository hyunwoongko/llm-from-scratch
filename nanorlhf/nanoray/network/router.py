from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from nanorlhf.nanoray.core.object_ref import ObjectRef


@dataclass
class NodeRegistry:
    """
    In-memory mapping of `node_id` -> (address, token)

    Notes:
        - `address` should include scheme and port,
            e.g. 'http://127.0.0.1:8001'
        - `token` (optional) is attached as `Authorization: Bearer <token>` header.

    Examples:
          >>> reg = NodeRegistry()
          >>> reg.register("A", "http://127.0.0.1:8001", token="secret")
          >>> reg.get("A")[0].startswith("http")
            True
    """

    _table: Dict[str, Tuple[str, Optional[str]]] = None

    def __post_init__(self):
        if self._table is None:
            self._table = {}

    def register(self, node_id: str, address: str, token: Optional[str] = None):
        """
        Register a node by its ID, address, and optional token.

        Args:
            node_id (str): Unique identifier for the node.
            address (str): Node's network address (including scheme and port).
            token (Optional[str]): Optional auth token for secure communication.
        """
        self._table[node_id] = (address.rstrip("/"), token)

    def get(self, node_id: str) -> Optional[Tuple[str, Optional[str]]]:
        """
        Retrieve the (address, token) tuple for the given node_id.

        Args:
            node_id (str): The unique identifier of the node.

        Returns:
            Optional[Tuple[str, Optional[str]]]: (address, token) if found, else None.
        """
        return self._table.get(node_id, None)


class Router:
    """
    Minimal router that resolves the destination node for an `ObjectRef`.

    Discussion:
        Q. Why is this a separate class from the registry?
            `NodeRegistry` answers "how to reach node X", while `Router` answers
            "which node should we reach for this *ref*". Keeping them separate
            lets us evolve routing logic (e.g., cache hits, replicas) without
            touching endpoint storage.
    """

    def __init__(self, registry: NodeRegistry):
        self._registry = registry

    def route_object(self, ref: ObjectRef) -> Optional[str]:
        """
        Resolve the owner node for a given `ObjectRef`.

        Args:
            ref (ObjectRef): The reference we want to fetch.

        Returns:
            Optional[str]: The owner node id if known, else None.
        """
        return ref.owner_node_id
