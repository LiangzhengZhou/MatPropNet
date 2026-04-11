from __future__ import annotations

import argparse

from matpropnet.preprocessing import build_parser, run_preprocess
from matpropnet.utils.runtime import setup_runtime_logging


def main(argv: list[str] | None = None):
    parser = build_parser()
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args(argv)
    setup_runtime_logging(level=args.log_level, log_file=args.log_file, force=True)
    run_preprocess(args)
