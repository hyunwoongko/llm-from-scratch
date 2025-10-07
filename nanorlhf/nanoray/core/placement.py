from dataclasses import dataclass, field
from typing import Dict, List

from nanorlhf.nanoray.utils import new_placement_group_id


@dataclass(frozen=True)
class Bundle:
    """
    A small resource bundle used inside a PlacementGroup.

    Attributes:
        cpus (float): CPU requirement for this bundle.
        gpus (float): GPU requirement for this bundle.
        resources (Dict[str, float]): Custom resources for this bundle. (e.g., {"ram": 32})

    Examples:
        >>> b = Bundle(cpus=2.0, gpus=1.0, resources={"ram": 32})
    """
    cpus: float = 0.0
    gpus: float = 0.0
    resources: Dict[str, float] = field(default_factory=dict)


class PlacementStrategy:
    """Minimal placement strategies for placement groups."""
    PACK = "PACK"  # Prefer a single node for all bundles / tasks
    SPREAD = "SPREAD"  # Prefer different nodes across bundles


@dataclass(frozen=True)
class PlacementGroup:
    """
    A logical grouping of resource bundles with a placement strategy.

    Attributes:
        pg_id (str): Stable id of this placement group.
        bundles (List[Bundle]): Resource bundles indexed by `bundle_index`.
        strategy (str): Placement strategy, either `PACK` or `SPREAD`.

    Examples:
        >>> from nanorlhf.nanoray.core.placement import PlacementGroup, Bundle, PlacementStrategy
        >>> pg = PlacementGroup(
        ...     pg_id="pg-1234",
        ...     bundles=[
        ...         Bundle(cpus=2.0, gpus=1.0, resources={"ram": 32}),
        ...         Bundle(cpus=4.0, gpus=0.0, resources={"ram": 64}),
        ...     ],
        ...     strategy=PlacementStrategy.SPREAD
        ... )

    Discussion:
        Q. What is a "placement group"?
           A placement group is a higher-level hint telling the scheduler that several tasks/actors
           are related and should be co-scheduled in a certain pattern.

        Q. Why not just use `pinned_node_id` on each task?
            You can, but PG centralizes intent. With PACK, the first successful placement
            determines the node for the group; flowwing tasks inherit it.
            With SPREAD, bundles prefer different nodes without you micromanaging ids.

        Q. How does RLHF benefit?
            - Co-locate components that share big state (model weights, dataset cache).
            - Spread pipeline shards across nodes deterministiacally (bundle index → node).
    """

    bundles: List[Bundle]
    strategy: str = PlacementStrategy.PACK
    pg_id: str = new_placement_group_id()

    def bundle(self, index: int) -> Bundle:
        """
        Return the bundle at the given index.

        Args:
            index (int): Index of the bundle to retrieve.

        Returns:
            Bundle: The bundle at the specified index.
        """
        return self.bundles[index]

    def __len__(self):
        """
        Return the number of bundles in this placement group.

        Returns:
            int: Count of bundles.
        """
        return len(self.bundles)
