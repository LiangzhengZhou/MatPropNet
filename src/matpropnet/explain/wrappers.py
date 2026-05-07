"""Model wrappers used by MatPropNet explainers."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


class GraphRegressionModelWrapper(nn.Module):
    """Expose a MatPropNet property model as a graph-level raw predictor."""

    def __init__(
        self,
        model: nn.Module,
        *,
        task_name: str | None = None,
        target_index: int = 0,
        normalizer=None,
    ):
        super().__init__()
        self.model = model
        self.task_name = task_name
        self.target_index = target_index
        self.normalizer = normalizer

    def forward(
        self, *args, edge_mask: torch.Tensor | None = None, **kwargs
    ):
        data = kwargs.get("data")
        if data is None:
            data = args[0] if args else None
        if data is None:
            raise ValueError("GraphRegressionModelWrapper requires a PyG data object.")
        try:
            output = self.model(data, edge_mask=edge_mask, explain_mode=True)
        except TypeError:
            output = self.model(data, edge_mask=edge_mask)
        pred = output["pred"] if isinstance(output, Mapping) else output
        if isinstance(pred, Mapping):
            if self.task_name is None:
                pred = next(iter(pred.values()))
            else:
                pred = pred[self.task_name]
        if pred.ndim > 1:
            pred = pred[:, self.target_index]
        pred = pred.view(-1)
        if self.normalizer is not None and self.task_name is not None:
            pred = self.normalizer(self.task_name, pred)
        return pred


def unwrap_model(model: nn.Module) -> nn.Module:
    """Remove DataParallel/DDP wrappers."""

    while hasattr(model, "module"):
        model = model.module
    return model
