from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import nn


class LossWeightingStrategy(nn.Module, ABC):
    """Abstract base class for multi-task loss weighting strategies."""

    def __init__(
        self,
        task_names: list[str],
        task_specs: dict[str, dict[str, Any]],
    ) -> None:
        super().__init__()
        self.task_names = list(task_names)
        self.task_specs = dict(task_specs)
        self.requires_post_backward = False

    @abstractmethod
    def compute_weighted_loss(
        self,
        task_losses,
        step_context: dict[str, Any],
    ):
        """Return total loss tensor and structured logging statistics."""

    def on_after_backward(self, step_context: dict[str, Any]) -> dict[str, Any]:
        return {}

    def on_after_optimizer_step(
        self, step_context: dict[str, Any]
    ) -> dict[str, Any]:
        return {}

    def extra_optimizer_param_groups(self) -> list[dict[str, Any]]:
        return []
