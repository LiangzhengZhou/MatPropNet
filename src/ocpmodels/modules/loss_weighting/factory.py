from __future__ import annotations

from typing import Any

from .base import LossWeightingStrategy
from .gradnorm import GradNormLossWeighting
from .static import StaticLossWeighting
from .uncertainty import UncertaintyLossWeighting


LOSS_WEIGHTING_REGISTRY = {
    "static": StaticLossWeighting,
    "gradnorm": GradNormLossWeighting,
    "uncertainty": UncertaintyLossWeighting,
}


def build_loss_weighting_strategy(
    config: dict[str, Any],
    task_specs: dict[str, dict[str, Any]],
) -> LossWeightingStrategy:
    strategy_cfg = dict(config.get("loss_weighting") or {})
    mode = strategy_cfg.pop("mode", "static")
    if mode not in LOSS_WEIGHTING_REGISTRY:
        raise ValueError(
            f"Unsupported loss weighting mode '{mode}'. "
            f"Expected one of {sorted(LOSS_WEIGHTING_REGISTRY)}."
        )
    return LOSS_WEIGHTING_REGISTRY[mode](
        task_names=list(task_specs.keys()),
        task_specs=task_specs,
        **strategy_cfg,
    )
