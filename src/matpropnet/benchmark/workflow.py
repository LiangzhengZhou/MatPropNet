from __future__ import annotations

import copy
import csv
import datetime as _dt
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from matpropnet.config import load_config
from matpropnet.tasks import run_train
from matpropnet.utils.runtime import setup_runtime_logging


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(payload: dict[str, Any], path: str | Path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_json(payload: dict[str, Any], path: str | Path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(rows: list[dict[str, Any]], path: str | Path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_benchmark_config(path: str | Path) -> dict[str, Any]:
    raw = _read_yaml(path)
    config = raw.get("benchmark", raw)
    if "output_dir" not in config:
        raise ValueError("Benchmark config requires 'output_dir'.")
    if not config.get("models"):
        raise ValueError("Benchmark config requires a non-empty 'models' list.")
    if not isinstance(config["models"], list):
        raise TypeError("'models' must be a list.")
    config.setdefault("name", Path(config["output_dir"]).name)
    config.setdefault("seed", 0)
    config.setdefault("print_every", None)
    config.setdefault("train", {})
    config.setdefault("evaluate", {"splits": ["val", "test"]})
    config.setdefault("stop_on_error", False)
    return config


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "model"


def _model_config_path(model: dict[str, Any], benchmark_path: Path) -> Path:
    raw_path = model.get("config") or model.get("base_config")
    if not raw_path:
        raise ValueError("Each benchmark model requires 'config'.")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = benchmark_path.parent / path
    return path.resolve()


def _log_file_for_run(run_dir: Path, train_cfg: dict[str, Any]) -> str | None:
    log_file_name = train_cfg.get("log_file_name", "train.log")
    if not log_file_name:
        return None
    log_file = Path(log_file_name).expanduser()
    if not log_file.is_absolute():
        log_file = run_dir / log_file
    return str(log_file)


def _checkpoint_path_for_trainer(trainer, checkpoint_name: str) -> str | None:
    checkpoint_dir = trainer.config.get("cmd", {}).get("checkpoint_dir")
    if not checkpoint_dir:
        return None
    checkpoint = Path(checkpoint_dir) / checkpoint_name
    return str(checkpoint) if checkpoint.exists() else None


def _prediction_csv_for_trainer(trainer, split: str) -> str | None:
    results_dir = trainer.config.get("cmd", {}).get("results_dir")
    if not results_dir:
        return None
    prediction_csv = Path(results_dir) / "property_predictions" / f"{split}.csv"
    return str(prediction_csv) if prediction_csv.exists() else None


def _metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "metric" in value:
            return value["metric"]
        if "value" in value:
            return value["value"]
    return value


def _flatten_metrics(metrics: dict[str, Any], split: str) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for metric_name, metric_value in (metrics or {}).items():
        value = _metric_value(metric_value)
        try:
            flattened[f"{split}_{metric_name}"] = float(value)
        except (TypeError, ValueError):
            flattened[f"{split}_{metric_name}"] = value
    return flattened


def _evaluate_trainer(trainer, splits: list[str]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for split in splits:
        results[split] = trainer.validate(split=split, disable_tqdm=True)
    return results


def _close_trainer(trainer):
    close = getattr(trainer, "close_datasets", None)
    if callable(close):
        close()


def _model_run_plan(
    *,
    benchmark_path: Path,
    output_dir: Path,
    model: dict[str, Any],
    seen_names: set[str],
) -> dict[str, Any]:
    display_name = str(model.get("name") or model.get("id") or "model")
    run_name = _safe_name(str(model.get("run_name") or display_name))
    if run_name in seen_names:
        raise ValueError(f"Duplicate benchmark model/run name: {run_name}")
    seen_names.add(run_name)
    config_path = _model_config_path(model, benchmark_path)
    return {
        "name": display_name,
        "run_name": run_name,
        "config": str(config_path),
        "run_dir": str(output_dir / "runs" / run_name),
    }


def run_benchmark(
    benchmark_config_path: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    benchmark_path = Path(benchmark_config_path).expanduser().resolve()
    benchmark_cfg = _load_benchmark_config(benchmark_path)
    output_dir = Path(benchmark_cfg["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = benchmark_path.parent / output_dir
    output_dir = output_dir.resolve()
    runs_dir = output_dir / "runs"
    summary_dir = output_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = benchmark_cfg["train"]
    evaluate_cfg = benchmark_cfg["evaluate"]
    checkpoint_name = train_cfg.get("checkpoint_name", "best_checkpoint.pt")
    evaluate_splits = list(evaluate_cfg.get("splits") or [])
    seed = int(benchmark_cfg.get("seed", 0))

    seen_names: set[str] = set()
    planned_models = [
        _model_run_plan(
            benchmark_path=benchmark_path,
            output_dir=output_dir,
            model=model,
            seen_names=seen_names,
        )
        for model in benchmark_cfg["models"]
    ]

    manifest: dict[str, Any] = {
        "name": benchmark_cfg["name"],
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "benchmark_config": str(benchmark_path),
        "output_dir": str(output_dir),
        "seed": seed,
        "evaluate_splits": evaluate_splits,
        "models": [],
        "summary": {
            "csv": str(summary_dir / "benchmark_summary.csv"),
            "json": str(summary_dir / "benchmark_summary.json"),
        },
    }

    _write_yaml({"benchmark": benchmark_cfg}, output_dir / "benchmark_config.yml")
    if dry_run:
        manifest["dry_run"] = True
        manifest["planned_models"] = [
            {
                **model,
                "log_file": _log_file_for_run(Path(model["run_dir"]), train_cfg),
            }
            for model in planned_models
        ]
        _write_json(manifest, output_dir / "benchmark_manifest.json")
        return manifest

    summary_rows: list[dict[str, Any]] = []
    for model_plan in planned_models:
        run_dir = Path(model_plan["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {
            "model": model_plan["name"],
            "run_name": model_plan["run_name"],
            "status": "running",
            "seed": seed,
            "config": model_plan["config"],
            "run_dir": str(run_dir),
            "log_file": _log_file_for_run(run_dir, train_cfg),
            "checkpoint": None,
            "error": None,
        }
        trainer = None
        try:
            model_config = load_config(model_plan["config"])
            model_config = copy.deepcopy(model_config)
            model_config["run_dir"] = str(run_dir)
            model_config["seed"] = seed
            model_config["identifier"] = model_plan["run_name"]
            _write_yaml(model_config, run_dir / "config.resolved.yml")
            setup_runtime_logging(
                level=train_cfg.get("log_level", "INFO"),
                log_file=row["log_file"],
                force=True,
            )
            logging.info("Starting benchmark model '%s'.", model_plan["name"])
            trainer = run_train(
                model_config,
                run_dir=str(run_dir),
                identifier=model_plan["run_name"],
                seed=seed,
                print_every=(
                    train_cfg.get("print_every")
                    if train_cfg.get("print_every") is not None
                    else benchmark_cfg.get("print_every")
                ),
                amp=train_cfg.get("amp"),
                cpu=train_cfg.get("cpu"),
            )
            row["checkpoint"] = _checkpoint_path_for_trainer(
                trainer, checkpoint_name
            )
            if evaluate_splits:
                metrics_by_split = _evaluate_trainer(trainer, evaluate_splits)
                for split, metrics in metrics_by_split.items():
                    row.update(_flatten_metrics(metrics, split))
                    prediction_csv = _prediction_csv_for_trainer(trainer, split)
                    if prediction_csv:
                        row[f"{split}_predictions"] = prediction_csv
            row["status"] = "completed"
        except Exception as exc:  # pragma: no cover - exercised through tests
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            logging.exception("Benchmark model '%s' failed.", model_plan["name"])
            if benchmark_cfg.get("stop_on_error", False):
                summary_rows.append(row)
                manifest["models"] = summary_rows
                _write_csv(summary_rows, summary_dir / "benchmark_summary.csv")
                _write_json(summary_rows, summary_dir / "benchmark_summary.json")
                _write_json(manifest, output_dir / "benchmark_manifest.json")
                raise
        finally:
            if trainer is not None:
                _close_trainer(trainer)
        summary_rows.append(row)
        manifest["models"] = summary_rows
        _write_csv(summary_rows, summary_dir / "benchmark_summary.csv")
        _write_json(summary_rows, summary_dir / "benchmark_summary.json")
        _write_json(manifest, output_dir / "benchmark_manifest.json")

    return manifest
