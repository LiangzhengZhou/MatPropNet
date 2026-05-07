"""Writers and table builders for explainability outputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from .materials import cell_volume, edge_metadata


def _jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def build_edge_table(
    data,
    edge_mask: torch.Tensor,
    *,
    sample_id: str,
    top_k: int,
    ensemble_columns: dict[str, list[float]] | None = None,
) -> list[dict[str, Any]]:
    metadata = edge_metadata(data)
    mask = edge_mask.detach().cpu().view(-1)
    order = torch.argsort(mask, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, mask.numel() + 1)
    top_ids = set(order[: min(top_k, mask.numel())].tolist())
    ensemble_columns = ensemble_columns or {}

    rows = []
    for edge_id, row in enumerate(metadata):
        out = {
            "sample_id": sample_id,
            **row,
            "edge_mask": float(mask[edge_id].item()),
            "edge_mask_rank": int(ranks[edge_id].item()),
            "is_topk": edge_id in top_ids,
        }
        for key, values in ensemble_columns.items():
            out[key] = values[edge_id]
        rows.append(out)
    return rows


def aggregate_by_bond_type(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["bond_type"]].append(row)
    out = []
    for bond_type, items in sorted(grouped.items()):
        masks = [float(item["edge_mask"]) for item in items]
        distances = [
            float(item["distance"])
            for item in items
            if item.get("distance") not in (None, "")
        ]
        topk_count = sum(1 for item in items if item.get("is_topk"))
        record = {
            "sample_id": items[0]["sample_id"],
            "bond_type": bond_type,
            "count": len(items),
            "mean_mask": sum(masks) / len(masks),
            "max_mask": max(masks),
            "sum_mask": sum(masks),
            "mean_distance": (
                sum(distances) / len(distances) if distances else None
            ),
            "topk_count": topk_count,
            "topk_fraction": topk_count / len(items),
        }
        for key in ("ensemble_std_mask", "confidence"):
            values = [
                float(item[key])
                for item in items
                if item.get(key) not in (None, "")
            ]
            if values:
                record[f"{key}_mean"] = sum(values) / len(values)
        out.append(record)
    return out


def write_csv(rows: list[dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_sample_outputs(
    *,
    output_dir: Path,
    data,
    edge_mask: torch.Tensor,
    sample_id: str,
    top_k: int,
    summary: dict[str, Any],
    extra_masks: dict[str, Any] | None = None,
    ensemble_columns: dict[str, list[float]] | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_rows = build_edge_table(
        data,
        edge_mask,
        sample_id=sample_id,
        top_k=top_k,
        ensemble_columns=ensemble_columns,
    )
    bond_rows = aggregate_by_bond_type(edge_rows)
    write_csv(edge_rows, output_dir / "explanation_edges.csv")
    write_csv(bond_rows, output_dir / "bond_type_importance.csv")
    torch.save(
        {"edge_mask": edge_mask.detach().cpu(), **(extra_masks or {})},
        output_dir / "masks.pt",
    )
    summary = dict(summary)
    summary.setdefault("cell_volume", cell_volume(data))
    with (output_dir / "explanation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_jsonable(summary), handle, indent=2)


def write_resolved_config(config: dict[str, Any], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_resolved.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_jsonable(config), handle, sort_keys=False)
