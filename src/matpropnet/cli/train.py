from __future__ import annotations

import argparse
import json

from matpropnet.tasks import run_train
from matpropnet.utils.runtime import setup_runtime_logging


def build_parser():
    parser = argparse.ArgumentParser(description="Train a MatPropNet model.")
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--identifier", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args, overrides = parser.parse_known_args(argv)
    setup_runtime_logging(level=args.log_level, log_file=args.log_file, force=True)
    result = run_train(
        args.config,
        overrides=overrides,
        checkpoint=args.checkpoint,
        run_dir=args.run_dir,
        identifier=args.identifier,
        seed=args.seed,
        print_every=args.print_every,
        amp=args.amp if args.amp else None,
        cpu=args.cpu if args.cpu else None,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(result, indent=2, ensure_ascii=False))
