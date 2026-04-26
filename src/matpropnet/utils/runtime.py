"""Runtime helpers shared by CLI entrypoints and compatibility wrappers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def ensure_src_on_path():
    repo_root = Path(__file__).resolve().parents[3]
    src_dir = repo_root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir


def parse_log_level(value: str | int | None):
    if value is None:
        return logging.INFO
    if isinstance(value, int):
        return value
    normalized = str(value).strip().upper()
    if not hasattr(logging, normalized):
        raise ValueError(f"Unsupported log level '{value}'.")
    return getattr(logging, normalized)


def setup_runtime_logging(
    level: int | str = logging.INFO,
    log_file: str | None = None,
    force: bool = False,
):
    level = parse_log_level(level)
    root = logging.getLogger()
    if root.handlers and not force:
        return
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s (%(levelname)s): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)
