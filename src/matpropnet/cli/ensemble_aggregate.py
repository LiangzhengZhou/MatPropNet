from __future__ import annotations

import argparse

from matpropnet.ensemble import aggregate_ensemble_predictions


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
    args = parser.parse_args(argv)
    tasks = (
        [task.strip() for task in args.tasks.split(",") if task.strip()]
        if args.tasks
        else None
    )
    aggregate_ensemble_predictions(
        args.predictions,
        output_path=args.out,
        tasks=tasks,
        include_members=not args.no_members,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
