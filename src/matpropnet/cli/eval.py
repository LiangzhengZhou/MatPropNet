from __future__ import annotations

import argparse
import json

from matpropnet.tasks import run_eval
from matpropnet.utils.runtime import setup_runtime_logging


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a MatPropNet checkpoint.")
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args, overrides = parser.parse_known_args(argv)
    setup_runtime_logging(level=args.log_level, log_file=args.log_file, force=True)
    result = run_eval(
        args.config,
        checkpoint=args.checkpoint,
        overrides=overrides,
        run_dir=args.run_dir,
        cpu=args.cpu if args.cpu else None,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(result, indent=2, ensure_ascii=False))
