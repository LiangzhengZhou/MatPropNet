from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from . import asymmetry as _asymmetry  # noqa: F401
from . import base_losses as _base_losses  # noqa: F401
from . import sample_weights as _sample_weights  # noqa: F401
from .registry import ASYMMETRY_REGISTRY, BASE_LOSS_REGISTRY, WEIGHT_REGISTRY


DEFAULT_LOSS_CONFIG = {
    "name": "mae",
    "weight": {"type": "none"},
    "asymmetry": {"enabled": False},
    "epsilon": 1.0e-8,
}


def normalize_loss_config(raw_config: str | dict | None) -> dict[str, Any]:
    if raw_config is None:
        config: dict[str, Any] = {}
    elif isinstance(raw_config, str):
        config = {"name": raw_config}
    elif isinstance(raw_config, dict):
        config = copy.deepcopy(raw_config)
    else:
        raise TypeError("Loss config must be a string, mapping, or None.")

    normalized = copy.deepcopy(DEFAULT_LOSS_CONFIG)
    normalized.update({k: v for k, v in config.items() if k not in {"weight", "asymmetry"}})
    normalized["weight"] = copy.deepcopy(DEFAULT_LOSS_CONFIG["weight"])
    normalized["weight"].update(config.get("weight") or {})
    normalized["asymmetry"] = copy.deepcopy(DEFAULT_LOSS_CONFIG["asymmetry"])
    normalized["asymmetry"].update(config.get("asymmetry") or {})
    return normalized


class ConfigurableRegressionLoss(nn.Module):
    def __init__(
        self,
        *,
        task_name: str,
        config: dict[str, Any],
        training_stats: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.task_name = task_name
        self.config = normalize_loss_config(config)
        self.training_stats = training_stats or {}
        self.epsilon = float(self.config.get("epsilon", 1.0e-8))

        base_name = str(self.config["name"]).lower()
        if base_name not in BASE_LOSS_REGISTRY:
            raise NotImplementedError(
                f"Unsupported regression loss '{base_name}' for task '{task_name}'."
            )
        base_cls = BASE_LOSS_REGISTRY[base_name]
        base_kwargs = {
            key: value
            for key, value in self.config.items()
            if key not in {"name", "weight", "asymmetry", "epsilon"}
        }
        self.base_loss = base_cls(**base_kwargs)

        self.sample_weight = self._build_weight_module(self.config["weight"])
        self.asymmetry = self._build_asymmetry_module(self.config["asymmetry"])

    def _build_weight_module(self, config: dict[str, Any]) -> nn.Module:
        weight_type = str(config.get("type", "none")).lower()
        if weight_type not in WEIGHT_REGISTRY:
            raise NotImplementedError(
                f"Unsupported sample-weight strategy '{weight_type}' for task '{self.task_name}'."
            )
        weight_cls = WEIGHT_REGISTRY[weight_type]
        kwargs = {key: value for key, value in config.items() if key != "type"}
        if weight_type == "target_power":
            ref_name = str(kwargs.get("ref", "p95"))
            kwargs["ref_value"] = self.training_stats.get(ref_name)
        elif weight_type == "bin_balanced":
            bins = kwargs.get("bins")
            if bins is None:
                raise ValueError("bin_balanced weights require explicit 'bins'.")
            values = self.training_stats.get("values")
            if values is None:
                raise ValueError(
                    f"Training stats for task '{self.task_name}' do not contain values needed for bin balancing."
                )
            parsed_bins = [
                float("inf") if isinstance(item, str) and item.lower() == "inf" else float(item)
                for item in bins
            ]
            counts, _ = np.histogram(np.asarray(values, dtype=np.float32), bins=parsed_bins)
            kwargs["bin_counts"] = counts.tolist()
        return weight_cls(**kwargs)

    def _build_asymmetry_module(self, config: dict[str, Any]) -> nn.Module:
        if not config.get("enabled", False):
            return ASYMMETRY_REGISTRY["none"]()
        mode = str(config.get("mode", "underestimation")).lower()
        if mode not in ASYMMETRY_REGISTRY:
            raise NotImplementedError(
                f"Unsupported asymmetry mode '{mode}' for task '{self.task_name}'."
            )
        asym_cls = ASYMMETRY_REGISTRY[mode]
        kwargs = {
            key: value
            for key, value in config.items()
            if key not in {"enabled", "mode"}
        }
        return asym_cls(**kwargs)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        log_var: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred = pred.reshape(-1).float()
        target = target.reshape(-1).float()
        if self.base_loss.loss_name == "gaussian_nll":
            per_sample_loss = self.base_loss(pred, target, log_var=log_var)
        else:
            per_sample_loss = self.base_loss(pred, target)
        sample_weight = self.sample_weight(target)
        effective_weight = self.asymmetry(pred, target, sample_weight)
        effective_weight = torch.nan_to_num(
            effective_weight, nan=1.0, posinf=1.0, neginf=1.0
        ).clamp_min(0.0)
        weighted_sum = torch.sum(effective_weight * per_sample_loss)
        normalizer = effective_weight.sum().clamp_min(self.epsilon)
        weighted_loss = weighted_sum / normalizer
        stats = {
            f"loss/base_loss/{self.task_name}": float(per_sample_loss.mean().detach().cpu()),
            f"loss/weighted_loss/{self.task_name}": float(weighted_loss.detach().cpu()),
            f"loss/mean_weight/{self.task_name}": float(effective_weight.mean().detach().cpu()),
            f"loss/max_weight/{self.task_name}": float(effective_weight.max().detach().cpu()),
            f"loss/min_weight/{self.task_name}": float(effective_weight.min().detach().cpu()),
        }
        if log_var is not None:
            log_var_for_stats = log_var.reshape(-1).float()
            if self.base_loss.loss_name == "gaussian_nll":
                log_var_for_stats = torch.clamp(
                    log_var_for_stats,
                    min=self.base_loss.min_log_var,
                    max=self.base_loss.max_log_var,
                )
            stats[f"loss/log_var_mean/{self.task_name}"] = float(
                log_var_for_stats.mean().detach().cpu()
            )
            stats[f"loss/sigma_mean/{self.task_name}"] = float(
                torch.exp(0.5 * log_var_for_stats).mean().detach().cpu()
            )
        return weighted_loss, stats

    def describe(self) -> OrderedDict[str, object]:
        description: OrderedDict[str, object] = OrderedDict()
        description["base loss"] = self.base_loss.loss_name
        for key, value in self.base_loss.extra_repr_dict().items():
            description[key] = value
        for key, value in self.sample_weight.describe().items():
            description[key] = value
        for key, value in self.asymmetry.describe().items():
            description[key] = value
        return description


def build_regression_loss(
    *,
    task_name: str,
    config: str | dict | None,
    training_stats: dict[str, Any] | None = None,
) -> ConfigurableRegressionLoss:
    return ConfigurableRegressionLoss(
        task_name=task_name,
        config=normalize_loss_config(config),
        training_stats=training_stats,
    )
