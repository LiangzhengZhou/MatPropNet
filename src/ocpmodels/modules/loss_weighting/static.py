from __future__ import annotations

from typing import Any

import torch

from .base import LossWeightingStrategy


class StaticLossWeighting(LossWeightingStrategy):
    """Backward-compatible static/manual task weighting."""

    def compute_weighted_loss(
        self,
        task_losses: dict[str, torch.Tensor],
        step_context: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del step_context
        if not task_losses:
            raise ValueError("Static loss weighting received no active task losses.")

        device = next(iter(task_losses.values())).device
        total_loss = torch.zeros((), device=device)
        stats: dict[str, Any] = {"loss_weighting/mode": "static"}

        for task_name, loss_value in task_losses.items():
            weight = float(self.task_specs[task_name].get("weight", 1.0))
            weighted_loss = weight * loss_value
            total_loss = total_loss + weighted_loss
            stats[f"loss_weighting/raw_loss/{task_name}"] = float(
                loss_value.detach().cpu()
            )
            stats[f"loss_weighting/weight/{task_name}"] = weight
            stats[f"loss_weighting/weighted_loss/{task_name}"] = float(
                weighted_loss.detach().cpu()
            )

        stats["loss_weighting/total_loss"] = float(total_loss.detach().cpu())
        return total_loss, stats
