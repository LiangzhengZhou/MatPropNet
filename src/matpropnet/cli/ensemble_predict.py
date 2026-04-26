from __future__ import annotations

import argparse

from matpropnet.ensemble import run_ensemble_predict


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a trained MatPropNet Deep Ensemble on a new LMDB."
    )
    parser.add_argument("--manifest", required=True, help="ensemble_manifest.json path.")
    parser.add_argument("--lmdb", required=True, help="LMDB data path to predict.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Optional comma-separated task names. Defaults to manifest tasks.",
    )
    parser.add_argument(
        "--no-members",
        action="store_true",
        help="Do not include per-member columns in the aggregate CSV.",
    )
    parser.add_argument(
        "--hide-progress",
        action="store_true",
        help="Hide prediction progress bars.",
    )
    args = parser.parse_args(argv)
    tasks = (
        [task.strip() for task in args.tasks.split(",") if task.strip()]
        if args.tasks
        else None
    )
    run_ensemble_predict(
        manifest_path=args.manifest,
        lmdb=args.lmdb,
        output_dir=args.out_dir,
        tasks=tasks,
        include_members=not args.no_members,
        hide_eval_progressbar=args.hide_progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

