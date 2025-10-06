from typing import List, Optional


class SchedulingPolicy:
    """
    Abstract interface for node selection among candidate nodes.

    Discussion:
        Q. Why does the policy get a *stable* node order?
            For reproducibility and teaching clarity. FIFO uses the given order;
            RoundRobin advances a cursor over this order while skipping ineligible nodes.
    """

    def set_node_order(self, node_ids: List[str]) -> None:
        """Called by the scheduler to set the global, stable node order."""
        raise NotImplementedError

    def select(self, candidates: List[str]) -> Optional[str]:
        """Choose one node id from `candidates` or return None if none is acceptable."""
        raise NotImplementedError


class FIFO(SchedulingPolicy):
    """
    First-In-First-Out (FIFO) scheduling policy.

    Attributes:
        _order (List[str]): The global, stable node order set by the scheduler.

    Examples:
        >>> policy = FIFO()
        >>> policy.set_node_order(['node-A', 'node-B', 'node-C'])
        >>> policy.select(['node-B', 'node-C'])
        'node-B'
    """

    _order: List[str] = []

    def set_node_order(self, node_ids: List[str]) -> None:
        """Called by the scheduler to set the global, stable node order."""
        self._order = node_ids.copy()

    def select(self, candidates: List[str]) -> Optional[str]:
        """Choose one node id from `candidates` or return None if none is acceptable."""
        if not candidates or not self._order:
            return None

        cand = set(candidates)
        for nid in self._order:
            if nid in cand:
                return nid
        return None


class RoundRobin(SchedulingPolicy):
    """
    Cycle through the node order and pick the next available candidate.

    Attributes:
        _order (List[str]): The global, stable node order set by the scheduler.
        _cursor (int): The current position in the node order for round-robin selection.

    Examples:
        >>> policy = RoundRobin()
        >>> policy.set_node_order(['node-A', 'node-B', 'node-C'])
        >>> policy.select(['node-B', 'node-C'])
        'node-B'
        >>> policy.select(['node-B', 'node-C'])
        'node-C'
        >>> policy.select(['node-B', 'node-C'])
        'node-B'
        >>> policy.select(['node-A'])
        'node-A'
        >>> policy.select(['node-D'])
        None

    Discussion:
        Q. What if the next node in round-robin is not eligible?
            The policy scans forward (with wrap-around) until it finds an eligible node.
            If none are eligible, it returns None.
    """
    _order: List[str] = []
    _cursor: int = 0

    def set_node_order(self, node_ids: List[str]) -> None:
        """Called by the scheduler to set the global, stable node order."""
        self._order = node_ids.copy()
        self._cursor = 0

    def select(self, candidates: List[str]) -> Optional[str]:
        """Choose one node id from `candidates` or return None if none is acceptable."""
        if not candidates or not self._order:
            return None

        cand = set(candidates)
        n = len(self._order)

        for step in range(n):
            idx = (self._cursor + step) % n
            nid = self._order[idx]
            if nid in cand:
                if len(candidates) > 1:
                    # Advance cursor only if there are multiple candidates
                    self._cursor = (idx + 1) % n
                return nid
        return candidates[0]
