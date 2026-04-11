from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import LossWeightingStrategy


class UncertaintyLossWeighting(LossWeightingStrategy):
    """Homoscedastic uncertainty-based weighting with per-task log variances."""

    def __init__(
        self,
        task_names: list[str],
        task_specs: dict[str, dict[str, Any]],
        init_log_var: float = 0.0,
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
        learn_rate: float | None = None,
    ) -> None:
        super().__init__(task_names, task_specs)
        self.log_vars = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.tensor(float(init_log_var), dtype=torch.float32)
                )
                for name in task_names
            }
        )
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.learn_rate = learn_rate

    def compute_weighted_loss(
        self,
        task_losses: dict[str, torch.Tensor],
        step_context: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del step_context
        if not task_losses:
            raise ValueError(
                "Uncertainty loss weighting received no active task losses."
            )

        device = next(iter(task_losses.values())).device
        total_loss = torch.zeros((), device=device)
        stats: dict[str, Any] = {"loss_weighting/mode": "uncertainty"}

        for task_name, raw_loss in task_losses.items():
            log_var = torch.clamp(
                self.log_vars[task_name], self.clamp_min, self.clamp_max
            )
            task_type = self.task_specs[task_name].get("type", "regression")
            precision = torch.exp(-log_var)

            if task_type == "regression":
                weighted_loss = 0.5 * precision * raw_loss + 0.5 * log_var
                effective_weight = 0.5 * precision
            else:
                weighted_loss = precision * raw_loss + log_var
                effective_weight = precision

            total_loss = total_loss + weighted_loss
            sigma = torch.exp(0.5 * log_var)
            stats[f"loss_weighting/raw_loss/{task_name}"] = float(
                raw_loss.detach().cpu()
            )
            stats[f"loss_weighting/log_var/{task_name}"] = float(
                log_var.detach().cpu()
            )
            stats[f"loss_weighting/sigma/{task_name}"] = float(
                sigma.detach().cpu()
            )
            stats[f"loss_weighting/weight/{task_name}"] = float(
                effective_weight.detach().cpu()
            )
            stats[f"loss_weighting/weighted_loss/{task_name}"] = float(
                weighted_loss.detach().cpu()
            )

        stats["loss_weighting/total_loss"] = float(total_loss.detach().cpu())
        return total_loss, stats

    def extra_optimizer_param_groups(self) -> list[dict[str, Any]]:
        if self.learn_rate is None:
            return []
        return [
            {
                "params": list(self.log_vars.parameters()),
                "lr": self.learn_rate,
                "name": "loss_weighting_uncertainty",
            }
        ]
