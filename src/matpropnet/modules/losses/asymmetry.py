from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_asymmetry


class BaseAsymmetry(nn.Module):
    mode = "none"

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def describe(self) -> dict[str, object]:
        return {"asymmetry enabled": False}


@register_asymmetry("none")
class NoAsymmetry(BaseAsymmetry):
    mode = "none"

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        del pred, target
        return weight


@register_asymmetry("underestimation")
class UnderestimationAsymmetry(BaseAsymmetry):
    mode = "underestimation"

    def __init__(self, *, factor: float = 2.0):
        super().__init__()
        self.factor = float(factor)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        under_mask = pred < target
        factor = torch.where(
            under_mask,
            torch.full_like(weight, self.factor),
            torch.ones_like(weight),
        )
        return weight * factor

    def describe(self) -> dict[str, object]:
        return {
            "asymmetry enabled": True,
            "asymmetry mode": self.mode,
            "asymmetry factor": self.factor,
        }


@register_asymmetry("threshold_underestimation")
class ThresholdUnderestimationAsymmetry(BaseAsymmetry):
    mode = "threshold_underestimation"

    def __init__(self, *, threshold: float, factor: float = 2.0):
        super().__init__()
        self.threshold = float(threshold)
        self.factor = float(factor)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        under_mask = (pred < target) & (target >= self.threshold)
        factor = torch.where(
            under_mask,
            torch.full_like(weight, self.factor),
            torch.ones_like(weight),
        )
        return weight * factor

    def describe(self) -> dict[str, object]:
        return {
            "asymmetry enabled": True,
            "asymmetry mode": self.mode,
            "asymmetry threshold": self.threshold,
            "asymmetry factor": self.factor,
        }

