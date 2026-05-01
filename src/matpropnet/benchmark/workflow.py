from __future__ import annotations

import copy
import csv
import datetime as _dt
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from matpropnet.config import load_config
from matpropnet.tasks import run_train
from matpropnet.tasks.core import _build_task, _build_trainer, _load_runtime_config
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
    config.setdefault("execution", {})
    config["execution"].setdefault("mode", "subprocess")
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


def _evaluate_checkpoint(
    *,
    model_config: dict[str, Any],
    checkpoint: str,
    run_dir: Path,
    run_name: str,
    seed: int,
    splits: list[str],
    train_cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    eval_config = _load_runtime_config(
        model_config,
        mode="validate",
        checkpoint=checkpoint,
        run_dir=str(run_dir / "eval"),
        identifier=f"{run_name}_eval",
        seed=seed,
        amp=train_cfg.get("amp"),
        cpu=train_cfg.get("cpu"),
    )
    eval_config["hide_eval_progressbar"] = True
    trainer = _build_trainer(eval_config)
    _build_task(eval_config, trainer)
    results: dict[str, dict[str, Any]] = {}
    try:
        for split in splits:
            results[split] = trainer.validate(split=split, disable_tqdm=True)
    finally:
        _close_trainer(trainer)
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


def _initial_summary_row(
    model_plan: dict[str, Any],
    run_dir: Path,
    train_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
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


def _run_single_model(
    *,
    model_plan: dict[str, Any],
    train_cfg: dict[str, Any],
    evaluate_splits: list[str],
    checkpoint_name: str,
    seed: int,
    print_every: int | None,
) -> dict[str, Any]:
    run_dir = Path(model_plan["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    row = _initial_summary_row(model_plan, run_dir, train_cfg, seed)
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
            print_every=print_every,
            amp=train_cfg.get("amp"),
            cpu=train_cfg.get("cpu"),
        )
        row["checkpoint"] = _checkpoint_path_for_trainer(trainer, checkpoint_name)
        if evaluate_splits:
            if not row["checkpoint"]:
                raise FileNotFoundError(
                    f"Expected checkpoint was not found for {model_plan['name']}."
                )
            metrics_by_split = _evaluate_checkpoint(
                model_config=model_config,
                checkpoint=row["checkpoint"],
                run_dir=run_dir,
                run_name=model_plan["run_name"],
                seed=seed,
                splits=evaluate_splits,
                train_cfg=train_cfg,
            )
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
    finally:
        if trainer is not None:
            _close_trainer(trainer)
    return row


def _run_model_subprocess(payload: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(payload["model_plan"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    payload_path = run_dir / "benchmark_worker_payload.json"
    result_path = run_dir / "benchmark_worker_result.json"
    worker_log_path = run_dir / "benchmark_worker.log"
    payload["result_path"] = str(result_path)
    _write_json(payload, payload_path)
    cmd = [
        sys.executable,
        "-m",
        "matpropnet.cli.benchmark",
        "--worker-payload",
        str(payload_path),
    ]
    with worker_log_path.open("w", encoding="utf-8") as worker_log:
        completed = subprocess.run(
            cmd,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result_path.exists():
        row = _read_yaml(result_path)
    else:
        row = _initial_summary_row(
            payload["model_plan"],
            run_dir,
            payload["train_cfg"],
            int(payload["seed"]),
        )
        row["status"] = "failed"
        row["error"] = (
            "Benchmark worker exited without writing a result "
            f"(returncode={completed.returncode}). See {worker_log_path}."
        )
    row["worker_log"] = str(worker_log_path)
    if completed.returncode != 0 and row.get("status") != "failed":
        row["status"] = "failed"
        row["error"] = (
            f"Benchmark worker exited with returncode={completed.returncode}. "
            f"See {worker_log_path}."
        )
    return row


def run_benchmark_worker(payload_path: str | Path) -> dict[str, Any]:
    payload = _read_yaml(payload_path)
    row = _run_single_model(
        model_plan=payload["model_plan"],
        train_cfg=payload["train_cfg"],
        evaluate_splits=list(payload.get("evaluate_splits") or []),
        checkpoint_name=payload.get("checkpoint_name", "best_checkpoint.pt"),
        seed=int(payload.get("seed", 0)),
        print_every=payload.get("print_every"),
    )
    result_path = payload.get("result_path")
    if result_path:
        _write_json(row, result_path)
    return row


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
    execution_cfg = benchmark_cfg["execution"]
    checkpoint_name = train_cfg.get("checkpoint_name", "best_checkpoint.pt")
    evaluate_splits = list(evaluate_cfg.get("splits") or [])
    seed = int(benchmark_cfg.get("seed", 0))
    execution_mode = str(execution_cfg.get("mode", "subprocess")).lower()
    if execution_mode not in {"subprocess", "in_process"}:
        raise ValueError(
            "benchmark.execution.mode must be 'subprocess' or 'in_process'."
        )

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
        try:
            print_every = (
                train_cfg.get("print_every")
                if train_cfg.get("print_every") is not None
                else benchmark_cfg.get("print_every")
            )
            if execution_mode == "in_process":
                row = _run_single_model(
                    model_plan=model_plan,
                    train_cfg=train_cfg,
                    evaluate_splits=evaluate_splits,
                    checkpoint_name=checkpoint_name,
                    seed=seed,
                    print_every=print_every,
                )
            else:
                logging.info(
                    "Starting benchmark model '%s' in an isolated subprocess.",
                    model_plan["name"],
                )
                row = _run_model_subprocess(
                    {
                        "model_plan": model_plan,
                        "train_cfg": train_cfg,
                        "evaluate_splits": evaluate_splits,
                        "checkpoint_name": checkpoint_name,
                        "seed": seed,
                        "print_every": print_every,
                    }
                )
                if row.get("status") == "completed":
                    logging.info(
                        "Benchmark model '%s' completed.", model_plan["name"]
                    )
                else:
                    logging.error(
                        "Benchmark model '%s' failed. See %s",
                        model_plan["name"],
                        row.get("worker_log"),
                    )
        except Exception as exc:  # pragma: no cover - exercised through tests
            row = _initial_summary_row(model_plan, run_dir, train_cfg, seed)
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            logging.exception("Benchmark model '%s' failed.", model_plan["name"])
        summary_rows.append(row)
        manifest["models"] = summary_rows
        _write_csv(summary_rows, summary_dir / "benchmark_summary.csv")
        _write_json(summary_rows, summary_dir / "benchmark_summary.json")
        _write_json(manifest, output_dir / "benchmark_manifest.json")
        if row.get("status") == "failed" and benchmark_cfg.get(
            "stop_on_error", False
        ):
            raise RuntimeError(
                f"Benchmark model '{model_plan['name']}' failed: {row.get('error')}"
            )

    return manifest
