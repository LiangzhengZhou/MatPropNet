"""Custom PyG-style explainer algorithms for MatPropNet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

try:  # pragma: no cover - exercised when torch_geometric is installed.
    from torch_geometric.explain import Explanation
    from torch_geometric.explain.algorithm import ExplainerAlgorithm
except ModuleNotFoundError:  # pragma: no cover
    Explanation = None

    class ExplainerAlgorithm(nn.Module):  # type: ignore[no-redef]
        pass


@dataclass
class ExplainerResult:
    edge_mask: torch.Tensor
    prediction: torch.Tensor
    target_prediction: torch.Tensor
    losses: list[float]
    algorithm: str


def _edge_mask_entropy(mask: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-8
    mask = mask.clamp(eps, 1.0 - eps)
    return -mask * torch.log(mask) - (1.0 - mask) * torch.log(1.0 - mask)


class MatPropNetEdgeMaskExplainer(ExplainerAlgorithm):
    """Learn a continuous edge mask for graph-level MatPropNet predictions."""

    def __init__(
        self,
        *,
        epochs: int = 100,
        lr: float = 0.01,
        edge_size_weight: float = 0.005,
        edge_entropy_weight: float = 0.001,
        task: str = "regression",
    ):
        super().__init__()
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.edge_size_weight = float(edge_size_weight)
        self.edge_entropy_weight = float(edge_entropy_weight)
        self.task = task

    def supports(self) -> bool:
        return True

    def explain_data(
        self,
        model: nn.Module,
        data,
        *,
        target: torch.Tensor | None = None,
    ) -> ExplainerResult:
        model.eval()
        num_edges = int(data.edge_index.shape[1])
        device = data.edge_index.device
        with torch.no_grad():
            full_prediction = model(data)
        target_prediction = (
            full_prediction.detach()
            if target is None
            else target.detach().to(device=full_prediction.device)
        )
        logits = nn.Parameter(torch.zeros(num_edges, device=device))
        optimizer = torch.optim.Adam([logits], lr=self.lr)
        losses: list[float] = []

        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            edge_mask = torch.sigmoid(logits)
            masked_prediction = model(data, edge_mask=edge_mask)
            pred_loss = torch.mean(torch.abs(masked_prediction - target_prediction))
            size_loss = edge_mask.mean()
            entropy_loss = _edge_mask_entropy(edge_mask).mean()
            loss = (
                pred_loss
                + self.edge_size_weight * size_loss
                + self.edge_entropy_weight * entropy_loss
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        edge_mask = torch.sigmoid(logits).detach()
        with torch.no_grad():
            masked_prediction = model(data, edge_mask=edge_mask)
        return ExplainerResult(
            edge_mask=edge_mask,
            prediction=masked_prediction.detach(),
            target_prediction=target_prediction.detach(),
            losses=losses,
            algorithm="matpropnet_edge_mask",
        )

    def forward(self, model: nn.Module, x=None, edge_index=None, **kwargs: Any):
        data = kwargs.get("data")
        if data is None:
            raise ValueError("MatPropNetEdgeMaskExplainer requires data=...")
        result = self.explain_data(model, data, target=kwargs.get("target"))
        if Explanation is None:
            return {
                "edge_index": data.edge_index,
                "edge_mask": result.edge_mask,
            }
        return Explanation(edge_index=data.edge_index, edge_mask=result.edge_mask)
