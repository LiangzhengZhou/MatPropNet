from __future__ import annotations

import csv
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


def _read_prediction_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _infer_tasks(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    tasks = []
    for key in rows[0].keys():
        if not key.startswith("pred_"):
            continue
        name = key.removeprefix("pred_")
        if name.endswith("_sigma") or name.endswith("_log_var"):
            continue
        tasks.append(name)
    return tasks


def _as_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def _validate_member_rows(member_rows: list[list[dict[str, str]]]):
    if not member_rows:
        raise ValueError("At least one member prediction CSV is required.")
    reference_ids = [row["id"] for row in member_rows[0]]
    for member_idx, rows in enumerate(member_rows[1:], start=1):
        ids = [row["id"] for row in rows]
        if ids != reference_ids:
            raise ValueError(
                "Prediction CSV ids must match exactly across ensemble members; "
                f"member {member_idx} differs from member 0."
            )


def _write_csv(rows: list[dict[str, Any]], output_path: str | Path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_uncertainty_calibration(
    rows: list[dict[str, Any]],
    tasks: list[str],
    *,
    min_variance: float = 1.0e-12,
    min_scale: float = 1.0,
    max_scale: float = 100.0,
) -> dict[str, dict[str, float]]:
    """Fit a scalar post-hoc variance calibration from rows with targets.

    For each task, this estimates a scalar ``s`` such that
    ``s * pred_var_total`` better matches squared residuals on a calibration
    split, usually validation. ``min_scale=1`` keeps this conservative by only
    inflating over-confident uncertainties.
    """

    calibration: dict[str, dict[str, float]] = {}
    for task_name in tasks:
        target_key = f"target_{task_name}"
        mean_key = f"pred_{task_name}_mean"
        var_key = f"pred_{task_name}_var_total"
        ratios = []
        for row in rows:
            if (
                row.get(target_key) in (None, "")
                or row.get(mean_key) in (None, "")
                or row.get(var_key) in (None, "")
            ):
                continue
            target = _as_float(row[target_key])
            pred = _as_float(row[mean_key])
            variance = max(_as_float(row[var_key]), min_variance)
            if not all(math.isfinite(value) for value in (target, pred, variance)):
                continue
            ratios.append((target - pred) ** 2 / variance)
        if not ratios:
            continue
        scale = float(np.mean(np.asarray(ratios, dtype=np.float64)))
        scale = min(max(scale, min_scale), max_scale)
        calibration[task_name] = {
            "variance_scale": scale,
            "std_scale": math.sqrt(scale),
            "num_samples": float(len(ratios)),
        }
    return calibration


def apply_uncertainty_calibration(
    rows: list[dict[str, Any]],
    calibration: dict[str, dict[str, float]],
    *,
    min_variance: float = 1.0e-12,
) -> list[dict[str, Any]]:
    for row in rows:
        for task_name, task_calibration in calibration.items():
            var_key = f"pred_{task_name}_var_total"
            if row.get(var_key) in (None, ""):
                continue
            scale = float(task_calibration["variance_scale"])
            variance = max(_as_float(row[var_key]), min_variance)
            calibrated_variance = variance * scale
            row[f"pred_{task_name}_var_total_calibrated"] = calibrated_variance
            row[f"pred_{task_name}_std_total_calibrated"] = math.sqrt(
                calibrated_variance
            )
            row[f"pred_{task_name}_uncertainty_scale"] = scale
    return rows


def aggregate_ensemble_predictions(
    prediction_files: list[str | Path],
    *,
    output_path: str | Path | None = None,
    tasks: list[str] | None = None,
    include_members: bool = True,
    calibration: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    member_rows = [_read_prediction_csv(path) for path in prediction_files]
    _validate_member_rows(member_rows)
    tasks = tasks or _infer_tasks(member_rows[0])
    if not tasks:
        raise ValueError("No prediction task columns were found.")

    aggregated_rows: list[dict[str, Any]] = []
    num_members = len(member_rows)
    num_samples = len(member_rows[0])
    for row_idx in range(num_samples):
        out_row: OrderedDict[str, Any] = OrderedDict()
        out_row["id"] = member_rows[0][row_idx]["id"]
        for task_name in tasks:
            pred_key = f"pred_{task_name}"
            sigma_key = f"pred_{task_name}_sigma"
            target_key = f"target_{task_name}"
            mus = np.asarray(
                [
                    _as_float(member_rows[member_idx][row_idx].get(pred_key))
                    for member_idx in range(num_members)
                ],
                dtype=np.float64,
            )
            if not np.isfinite(mus).all():
                raise ValueError(f"Non-finite prediction found for task '{task_name}'.")

            mean = float(mus.mean())
            var_epistemic = float(np.mean(mus * mus) - mean * mean)
            var_epistemic = max(var_epistemic, 0.0)
            sigma_values = [
                _as_float(member_rows[member_idx][row_idx].get(sigma_key))
                for member_idx in range(num_members)
                if sigma_key in member_rows[member_idx][row_idx]
            ]
            has_aleatoric = len(sigma_values) == num_members and all(
                math.isfinite(value) for value in sigma_values
            )
            var_aleatoric = (
                float(np.mean(np.asarray(sigma_values, dtype=np.float64) ** 2))
                if has_aleatoric
                else 0.0
            )
            var_total = var_epistemic + var_aleatoric

            if target_key in member_rows[0][row_idx]:
                out_row[target_key] = member_rows[0][row_idx][target_key]
            out_row[f"pred_{task_name}_mean"] = mean
            out_row[f"pred_{task_name}_var_epistemic"] = var_epistemic
            out_row[f"pred_{task_name}_std_epistemic"] = math.sqrt(var_epistemic)
            out_row[f"pred_{task_name}_var_aleatoric"] = var_aleatoric
            out_row[f"pred_{task_name}_std_aleatoric"] = math.sqrt(var_aleatoric)
            out_row[f"pred_{task_name}_var_total"] = var_total
            out_row[f"pred_{task_name}_std_total"] = math.sqrt(var_total)
            if include_members:
                for member_idx, value in enumerate(mus.tolist()):
                    out_row[f"pred_{task_name}_member_{member_idx}"] = value
                if has_aleatoric:
                    for member_idx, value in enumerate(sigma_values):
                        out_row[f"pred_{task_name}_sigma_member_{member_idx}"] = value
        aggregated_rows.append(out_row)

    if calibration:
        apply_uncertainty_calibration(aggregated_rows, calibration)

    if output_path is not None:
        _write_csv(aggregated_rows, output_path)
    return aggregated_rows
