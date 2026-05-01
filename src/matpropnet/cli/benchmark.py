from __future__ import annotations

import argparse
import json

from matpropnet.benchmark import run_benchmark
from matpropnet.benchmark.workflow import run_benchmark_worker
from matpropnet.utils.runtime import setup_runtime_logging


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and compare multiple MatPropNet model configs."
    )
    parser.add_argument("--config", default=None, help="Benchmark YAML config.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and expand the benchmark plan without training.",
    )
    parser.add_argument(
        "--worker-payload",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker_payload:
        run_benchmark_worker(args.worker_payload)
        return 0
    if not args.config:
        parser.error("--config is required unless --worker-payload is used.")
    setup_runtime_logging(level=args.log_level, log_file=args.log_file, force=True)
    result = run_benchmark(args.config, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
