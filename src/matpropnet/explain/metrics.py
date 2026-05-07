"""Regression fidelity metrics for graph-level explanations."""

from __future__ import annotations

import copy

import torch


def topk_hard_mask(edge_mask: torch.Tensor, top_k: int) -> torch.Tensor:
    mask = torch.zeros_like(edge_mask.view(-1))
    if mask.numel() == 0:
        return mask
    top = torch.topk(edge_mask.view(-1), k=min(top_k, mask.numel())).indices
    mask[top] = 1.0
    return mask


def regression_fidelity_metrics(model, data, edge_mask: torch.Tensor, top_k: int):
    """Compute keep/remove prediction deltas for a top-k explanation mask."""

    hard_keep = topk_hard_mask(edge_mask, top_k).to(edge_mask.device)
    hard_remove = 1.0 - hard_keep
    with torch.no_grad():
        y_full = model(data)
        y_keep = model(copy.copy(data), edge_mask=hard_keep)
        y_remove = model(copy.copy(data), edge_mask=hard_remove)
    sufficiency_error = torch.abs(y_keep - y_full)
    necessity_drop = torch.abs(y_remove - y_full)
    signed_remove_drop = y_full - y_remove
    relative_drop = necessity_drop / (torch.abs(y_full) + 1.0e-8)
    return {
        "y_full": y_full.detach().cpu(),
        "y_keep_topk": y_keep.detach().cpu(),
        "y_remove_topk": y_remove.detach().cpu(),
        "sufficiency_error": sufficiency_error.detach().cpu(),
        "necessity_drop": necessity_drop.detach().cpu(),
        "signed_remove_drop": signed_remove_drop.detach().cpu(),
        "relative_remove_drop": relative_drop.detach().cpu(),
    }
