from dataclasses import dataclass, field
from typing import Dict

from nanorlhf.nanoray.core.task import Task


@dataclass
class NodeState:
    """
    Internal resource tracker for a node.

    Attributes:
        total_cpus (float): Total CPU capacity of the node.
        total_gpus (float): Total GPU capacity of the node.
        total_custom (Dict[str, float]): Total custom resources of the node.
        used_cpus (float): CPUs currently reserved.
        used_gpus (float): GPUs currently reserved.
        used_custom (Dict[str, float]): Custom resources currently reserved.
    """

    total_cpus: float = 1.0
    total_gpus: float = 0.0
    total_custom: Dict[str, float] = field(default_factory=dict)

    used_cpus: float = 0.0
    used_gpus: float = 0.0
    used_custom: Dict[str, float] = field(default_factory=dict)

    def fit_custom(self, req: Dict[str, float]) -> bool:
        """
        Check if the requested custom resources can fit in the available capacity.

        Args:
            req (Dict[str, float]): Custom resource requirements.

        Returns:
            bool: True if the request can be satisfied, False otherwise.
        """
        for kind, requirement in req.items():
            total = self.total_custom.get(kind, 0.0)
            used = self.used_custom.get(kind, 0.0)
            if used + float(requirement) > total:
                return False
        return True

    def can_run(self, task: Task) -> bool:
        """
        Check if the node has enough available resources to run the given task.

        Args:
            task (Task): The task specification to check.

        Returns:
            bool: True if the node can run the task, False otherwise.
        """
        if self.used_cpus + task.num_cpus > self.total_cpus:
            return False
        if self.used_gpus + task.num_gpus > self.total_gpus:
            return False
        if task.resources and not self.fit_custom(task.resources):
            return False
        return True

    def allocate(self, task: Task):
        """
        Reserve resources for the given task.

        Args:
            task (Task): The task specification whose resources to allocate.

        Notes:
            `release(task)` must be called afterward to free them.
        """

        self.used_cpus += task.num_cpus
        self.used_gpus += task.num_gpus
        if task.resources:
            for k, v in task.resources.items():
                self.used_custom[k] = self.used_custom.get(k, 0.0) + float(v)

    def release(self, task: Task):
        """
        Free resources previously reserved for the given task.

        Args:
            task (Task): The task specification whose resources to release.
        """
        self.used_cpus -= task.num_cpus
        self.used_gpus -= task.num_gpus
        if task.resources:
            for k, v in task.resources.items():
                self.used_custom[k] = self.used_custom.get(k, 0.0) - float(v)
                if self.used_custom[k] <= 0.0:
                    self.used_custom.pop(k)
