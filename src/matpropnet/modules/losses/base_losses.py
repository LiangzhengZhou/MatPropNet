from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_base_loss


def _flatten_regression_tensors(
    pred: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = pred.reshape(-1).float()
    target = target.reshape(-1).float()
    if pred.shape != target.shape:
        raise ValueError(
            f"Regression loss expects matching shapes, got {pred.shape} and {target.shape}."
        )
    return pred, target


class BaseRegressionLoss(nn.Module):
    loss_name = "base"

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr_dict(self) -> dict[str, float]:
        return {}


@register_base_loss("mae")
@register_base_loss("l1")
class MAELoss(BaseRegressionLoss):
    loss_name = "mae"

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = _flatten_regression_tensors(pred, target)
        return torch.abs(pred - target)


@register_base_loss("mse")
class MSELoss(BaseRegressionLoss):
    loss_name = "mse"

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = _flatten_regression_tensors(pred, target)
        error = pred - target
        return error * error


@register_base_loss("huber")
class HuberLoss(BaseRegressionLoss):
    loss_name = "huber"

    def __init__(self, delta: float = 1.0):
        super().__init__()
        if delta <= 0:
            raise ValueError("Huber loss requires delta > 0.")
        self.delta = float(delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = _flatten_regression_tensors(pred, target)
        error = pred - target
        abs_error = torch.abs(error)
        quadratic = 0.5 * error * error
        linear = self.delta * (abs_error - 0.5 * self.delta)
        return torch.where(abs_error <= self.delta, quadratic, linear)

    def extra_repr_dict(self) -> dict[str, float]:
        return {"delta": self.delta}


@register_base_loss("smooth_l1")
class SmoothL1Loss(BaseRegressionLoss):
    loss_name = "smooth_l1"

    def __init__(self, beta: float = 1.0):
        super().__init__()
        if beta <= 0:
            raise ValueError("SmoothL1 loss requires beta > 0.")
        self.beta = float(beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = _flatten_regression_tensors(pred, target)
        return F.smooth_l1_loss(pred, target, reduction="none", beta=self.beta)

    def extra_repr_dict(self) -> dict[str, float]:
        return {"beta": self.beta}


@register_base_loss("log_cosh")
class LogCoshLoss(BaseRegressionLoss):
    loss_name = "log_cosh"

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = _flatten_regression_tensors(pred, target)
        error = pred - target
        return error + F.softplus(-2.0 * error) - math.log(2.0)

