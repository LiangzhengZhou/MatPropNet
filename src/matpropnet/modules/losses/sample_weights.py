from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from .registry import register_weight


class BaseSampleWeight(nn.Module):
    weight_type = "none"

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def describe(self) -> dict[str, object]:
        return {"weight type": self.weight_type}


@register_weight("none")
class NoSampleWeight(BaseSampleWeight):
    weight_type = "none"

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(target.reshape(-1).float())


@register_weight("target_power")
class TargetPowerWeight(BaseSampleWeight):
    weight_type = "target_power"

    def __init__(
        self,
        *,
        alpha: float = 2.0,
        gamma: float = 1.0,
        ref: str = "p95",
        max_weight: float = 5.0,
        ref_value: float | None = None,
        epsilon: float = 1.0e-8,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.ref = ref
        self.max_weight = float(max_weight)
        self.ref_value = None if ref_value is None else float(ref_value)
        self.epsilon = float(epsilon)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        target = target.reshape(-1).float()
        if self.ref_value is None or not math.isfinite(self.ref_value):
            return torch.ones_like(target)
        ref_value = max(abs(self.ref_value), self.epsilon)
        target_scale = torch.abs(target).clamp_min(self.epsilon)
        ratio = target_scale / ref_value
        weight = 1.0 + self.alpha * torch.pow(ratio, self.gamma)
        return torch.clamp(weight, max=self.max_weight)

    def describe(self) -> dict[str, object]:
        return {
            "weight type": self.weight_type,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "ref": self.ref,
            "ref value": self.ref_value,
            "max weight": self.max_weight,
        }


@register_weight("threshold")
class ThresholdWeight(BaseSampleWeight):
    weight_type = "threshold"

    def __init__(self, *, threshold: float, high_weight: float = 2.0):
        super().__init__()
        self.threshold = float(threshold)
        self.high_weight = float(high_weight)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        target = target.reshape(-1).float()
        return torch.where(
            target >= self.threshold,
            torch.full_like(target, self.high_weight),
            torch.ones_like(target),
        )

    def describe(self) -> dict[str, object]:
        return {
            "weight type": self.weight_type,
            "threshold": self.threshold,
            "high weight": self.high_weight,
        }


@register_weight("bin_balanced")
class BinBalancedWeight(BaseSampleWeight):
    weight_type = "bin_balanced"

    def __init__(
        self,
        *,
        bins: Sequence[float | str],
        bin_weights: Sequence[float] | None = None,
        bin_counts: Sequence[int] | None = None,
        min_weight: float = 0.5,
        max_weight: float = 5.0,
    ):
        super().__init__()
        if len(bins) < 2:
            raise ValueError("Bin-balanced weights require at least two bin edges.")
        parsed_bins = [self._parse_edge(item) for item in bins]
        if any(right <= left for left, right in zip(parsed_bins[:-1], parsed_bins[1:])):
            raise ValueError("Bin edges must be strictly increasing.")
        if bin_weights is None:
            if bin_counts is None:
                raise ValueError(
                    "Bin-balanced weights require either bin_weights or bin_counts."
                )
            bin_weights = self._build_bin_weights(
                bin_counts=bin_counts,
                min_weight=min_weight,
                max_weight=max_weight,
            )
        if len(bin_weights) != len(parsed_bins) - 1:
            raise ValueError("Number of bin weights must match number of intervals.")
        self.register_buffer(
            "bin_edges", torch.tensor(parsed_bins, dtype=torch.float32)
        )
        self.register_buffer(
            "bin_weights", torch.tensor(bin_weights, dtype=torch.float32)
        )
        self.bin_counts = None if bin_counts is None else [int(item) for item in bin_counts]
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

    @staticmethod
    def _parse_edge(value: float | str) -> float:
        if isinstance(value, str) and value.lower() == "inf":
            return float("inf")
        return float(value)

    @staticmethod
    def _build_bin_weights(
        *,
        bin_counts: Sequence[int],
        min_weight: float,
        max_weight: float,
    ) -> list[float]:
        counts = np.asarray(bin_counts, dtype=np.float64)
        if counts.ndim != 1:
            raise ValueError("bin_counts must be a 1D sequence.")
        raw_weights = np.ones_like(counts, dtype=np.float64)
        valid = counts > 0
        raw_weights[valid] = 1.0 / counts[valid]
        total = counts.sum()
        mean_weight = 1.0
        if total > 0 and valid.any():
            mean_weight = float((raw_weights[valid] * counts[valid]).sum() / total)
            if mean_weight <= 0 or not math.isfinite(mean_weight):
                mean_weight = 1.0
        normalized = raw_weights / mean_weight
        clipped = np.clip(normalized, min_weight, max_weight)
        clipped[~valid] = 1.0
        return clipped.astype(np.float32).tolist()

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        target = target.reshape(-1).float()
        bin_ids = torch.bucketize(target, self.bin_edges[1:-1], right=False)
        return self.bin_weights[bin_ids]

    def describe(self) -> dict[str, object]:
        intervals = []
        edges = self.bin_edges.detach().cpu().tolist()
        weights = self.bin_weights.detach().cpu().tolist()
        counts = self.bin_counts or [0] * len(weights)
        for idx, weight in enumerate(weights):
            left = edges[idx]
            right = edges[idx + 1]
            intervals.append(
                {
                    "range": f"[{left:g}, {right if math.isinf(right) else f'{right:g}'}"
                    + (")" if not math.isinf(right) else ")"),
                    "count": counts[idx] if idx < len(counts) else 0,
                    "weight": float(weight),
                }
            )
        return {
            "weight type": self.weight_type,
            "min weight": self.min_weight,
            "max weight": self.max_weight,
            "bins": intervals,
        }

