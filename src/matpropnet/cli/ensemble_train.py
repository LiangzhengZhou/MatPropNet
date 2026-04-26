from __future__ import annotations

import argparse

from matpropnet.ensemble import run_ensemble_train


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train and evaluate a MatPropNet Deep Ensemble."
    )
    parser.add_argument("--config", required=True, help="Ensemble YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and expand the ensemble plan without training.",
    )
    args = parser.parse_args(argv)
    run_ensemble_train(args.config, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

