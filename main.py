"""Compatibility wrapper for legacy `python main.py --mode ...` usage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from matpropnet.cli.eval import main as eval_main
from matpropnet.cli.predict import main as predict_main
from matpropnet.cli.train import main as train_main


def _dispatch():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", required=True, choices=["train", "validate", "predict"])
    parser.add_argument("--config-yml", required=True)
    args, remaining = parser.parse_known_args()

    translated = ["--config", args.config_yml] + remaining
    if args.mode == "train":
        train_main(translated)
    elif args.mode == "validate":
        eval_main(translated)
    else:
        predict_main(translated)


if __name__ == "__main__":
    _dispatch()
