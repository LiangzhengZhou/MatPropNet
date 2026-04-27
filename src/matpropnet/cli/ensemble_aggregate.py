from __future__ import annotations

import argparse
import csv

from matpropnet.ensemble import aggregate_ensemble_predictions, fit_uncertainty_calibration


def _read_rows(path: str):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _infer_aggregate_tasks(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    tasks = []
    for key in rows[0]:
        if key.startswith("pred_") and key.endswith("_mean"):
            tasks.append(key.removeprefix("pred_").removesuffix("_mean"))
    return tasks


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate MatPropNet ensemble prediction CSV files."
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="Prediction CSV files from ensemble members.",
    )
    parser.add_argument("--out", required=True, help="Output aggregate CSV path.")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Optional comma-separated task names. Defaults to inferred tasks.",
    )
    parser.add_argument(
        "--no-members",
        action="store_true",
        help="Do not include per-member prediction columns in the output.",
    )
    parser.add_argument(
        "--calibrate-from",
        default=None,
        help=(
            "Optional aggregate CSV with targets, usually val_ensemble.csv, "
            "used to fit a scalar total-variance calibration."
        ),
    )
    parser.add_argument("--min-scale", type=float, default=1.0)
    parser.add_argument("--max-scale", type=float, default=100.0)
    args = parser.parse_args(argv)
    tasks = (
        [task.strip() for task in args.tasks.split(",") if task.strip()]
        if args.tasks
        else None
    )
    calibration = None
    if args.calibrate_from:
        calibration_rows = _read_rows(args.calibrate_from)
        calibration = fit_uncertainty_calibration(
            calibration_rows,
            tasks or _infer_aggregate_tasks(calibration_rows),
            min_scale=args.min_scale,
            max_scale=args.max_scale,
        )
    aggregate_ensemble_predictions(
        args.predictions,
        output_path=args.out,
        tasks=tasks,
        include_members=not args.no_members,
        calibration=calibration,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
