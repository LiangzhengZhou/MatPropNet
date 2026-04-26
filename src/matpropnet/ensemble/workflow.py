from __future__ import annotations

import copy
import datetime as _dt
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from matpropnet.config import load_config
from matpropnet.ensemble.aggregate import aggregate_ensemble_predictions
from matpropnet.tasks.core import (
    _build_task,
    _build_trainer,
    _load_runtime_config,
    _write_prediction_csv,
    run_train,
)


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


def _load_ensemble_config(path: str | Path) -> dict[str, Any]:
    raw = _read_yaml(path)
    config = raw.get("ensemble", raw)
    if "base_config" not in config:
        raise ValueError("Ensemble config requires 'base_config'.")
    if "output_dir" not in config:
        raise ValueError("Ensemble config requires 'output_dir'.")
    num_members = int(config.get("num_members", 0))
    seeds = list(config.get("seeds") or [])
    if not seeds and num_members <= 0:
        raise ValueError("Provide either 'num_members' or explicit 'seeds'.")
    if not seeds:
        seeds = list(range(num_members))
    if num_members and len(seeds) != num_members:
        raise ValueError("'num_members' must match the number of seeds.")
    config["num_members"] = len(seeds)
    config["seeds"] = [int(seed) for seed in seeds]
    config.setdefault("name", Path(config["output_dir"]).name)
    config.setdefault("train", {})
    config.setdefault("evaluate", {"splits": ["test"]})
    config.setdefault("aggregate", {})
    return config


def _dataset_split_sources(base_config: dict[str, Any]) -> dict[str, str]:
    dataset = base_config.get("dataset")
    sources: dict[str, str] = {}
    if isinstance(dataset, list):
        names = ["train", "val", "test"]
        for idx, entry in enumerate(dataset[:3]):
            if isinstance(entry, dict) and entry.get("src"):
                sources[names[idx]] = entry["src"]
    elif isinstance(dataset, dict):
        if dataset.get("src"):
            sources["train"] = dataset["src"]
        for split in ("train", "val", "test"):
            value = dataset.get(split)
            if isinstance(value, list) and value:
                value = value[0]
            if isinstance(value, dict) and value.get("src"):
                sources[split] = value["src"]
    for config_key, split in (("val_dataset", "val"), ("test_dataset", "test")):
        value = base_config.get(config_key)
        if isinstance(value, dict) and value.get("src"):
            sources[split] = value["src"]
    return sources


def _task_names(base_config: dict[str, Any]) -> list[str]:
    return list(base_config.get("task", {}).get("tasks", {}).keys())


def _carry_target_normalization(config: dict[str, Any], base_config: dict[str, Any]):
    task_cfg = config.setdefault("task", {})
    if task_cfg.get("target_mean") is not None and task_cfg.get("target_std") is not None:
        return
    dataset = base_config.get("dataset")
    first_entry = None
    if isinstance(dataset, list) and dataset:
        first_entry = dataset[0]
    elif isinstance(dataset, dict) and dataset.get("src"):
        first_entry = dataset
    elif isinstance(dataset, dict) and isinstance(dataset.get("train"), dict):
        first_entry = dataset["train"]
    if not isinstance(first_entry, dict):
        return
    if first_entry.get("target_mean") is not None and first_entry.get("target_std") is not None:
        task_cfg.setdefault("target_mean", first_entry["target_mean"])
        task_cfg.setdefault("target_std", first_entry["target_std"])


def _checkpoint_path_for_trainer(trainer, checkpoint_name: str) -> str:
    checkpoint = Path(trainer.config["cmd"]["checkpoint_dir"]) / checkpoint_name
    if not checkpoint.exists():
        raise FileNotFoundError(f"Expected checkpoint was not found: {checkpoint}")
    return str(checkpoint)


def _predict_checkpoint_on_lmdb(
    *,
    base_config: dict[str, Any],
    checkpoint: str,
    lmdb: str,
    output_csv: str | Path,
    run_dir: str | Path,
    seed: int | None = None,
    cpu: bool | None = None,
    amp: bool | None = None,
    hide_eval_progressbar: bool | None = None,
) -> dict[str, list[Any]]:
    config = _load_runtime_config(
        base_config,
        mode="predict",
        checkpoint=checkpoint,
        run_dir=str(run_dir),
        seed=seed,
        cpu=cpu,
        amp=amp,
    )
    config["dataset"] = {"test": {"src": lmdb}}
    _carry_target_normalization(config, base_config)
    trainer = _build_trainer(config)
    _build_task(config, trainer)
    try:
        predictions = trainer.predict(
            trainer.test_loader,
            results_file=None,
            disable_tqdm=(
                hide_eval_progressbar
                if hide_eval_progressbar is not None
                else config.get("hide_eval_progressbar", False)
            ),
        )
    finally:
        trainer.close_datasets()
    _write_prediction_csv(predictions, str(output_csv))
    return predictions


def _ensemble_metrics(rows: list[dict[str, Any]], tasks: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for task_name in tasks:
        target_key = f"target_{task_name}"
        pred_key = f"pred_{task_name}_mean"
        var_key = f"pred_{task_name}_var_total"
        usable = [
            row
            for row in rows
            if row.get(target_key) not in (None, "")
            and row.get(pred_key) not in (None, "")
        ]
        if not usable:
            continue
        y = np.asarray([float(row[target_key]) for row in usable], dtype=np.float64)
        pred = np.asarray([float(row[pred_key]) for row in usable], dtype=np.float64)
        err = pred - y
        mse = float(np.mean(err * err))
        mae = float(np.mean(np.abs(err)))
        ss_res = float(np.sum(err * err))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        task_metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": 0.0 if ss_tot <= 1.0e-12 else 1.0 - ss_res / ss_tot,
        }
        if all(row.get(var_key) not in (None, "") for row in usable):
            var_total = np.asarray(
                [max(float(row[var_key]), 1.0e-12) for row in usable],
                dtype=np.float64,
            )
            nll = 0.5 * np.log(2.0 * np.pi * var_total) + 0.5 * err * err / var_total
            std_total = np.sqrt(var_total)
            task_metrics.update(
                {
                    "nll": float(np.mean(nll)),
                    "coverage_1sigma": float(np.mean(np.abs(err) <= std_total)),
                    "coverage_2sigma": float(np.mean(np.abs(err) <= 2.0 * std_total)),
                    "mean_total_std": float(np.mean(std_total)),
                }
            )
        metrics[task_name] = task_metrics
    return metrics


def run_ensemble_train(
    ensemble_config_path: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    ensemble_cfg = _load_ensemble_config(ensemble_config_path)
    base_config_path = Path(ensemble_cfg["base_config"]).expanduser()
    if not base_config_path.is_absolute():
        base_config_path = Path(ensemble_config_path).parent / base_config_path
    base_config = load_config(base_config_path)
    output_dir = Path(ensemble_cfg["output_dir"]).expanduser().resolve()
    members_dir = output_dir / "members"
    aggregate_dir = output_dir / "aggregate"
    checkpoint_name = ensemble_cfg["train"].get("checkpoint_name", "best_checkpoint.pt")
    overwrite = bool(ensemble_cfg["train"].get("overwrite", False))
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    members_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    split_sources = _dataset_split_sources(base_config)
    tasks = ensemble_cfg.get("aggregate", {}).get("tasks") or _task_names(base_config)
    evaluate_splits = list(ensemble_cfg.get("evaluate", {}).get("splits") or [])
    include_members = bool(ensemble_cfg.get("aggregate", {}).get("include_members", True))

    _write_yaml({"ensemble": ensemble_cfg}, output_dir / "ensemble_config.yml")
    _write_yaml(base_config, output_dir / "base_config.resolved.yml")

    manifest: dict[str, Any] = {
        "name": ensemble_cfg["name"],
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "base_config": str(base_config_path.resolve()),
        "output_dir": str(output_dir),
        "num_members": ensemble_cfg["num_members"],
        "tasks": tasks,
        "members": [],
        "aggregate": {},
    }

    if dry_run:
        manifest["dry_run"] = True
        manifest["planned_members"] = [
            {
                "index": idx,
                "seed": seed,
                "run_dir": str(members_dir / f"member_{idx:03d}"),
            }
            for idx, seed in enumerate(ensemble_cfg["seeds"])
        ]
        return manifest

    for idx, seed in enumerate(ensemble_cfg["seeds"]):
        member_dir = members_dir / f"member_{idx:03d}"
        member_dir.mkdir(parents=True, exist_ok=True)
        member_config = copy.deepcopy(base_config)
        member_config["seed"] = seed
        member_config["run_dir"] = str(member_dir)
        member_config["identifier"] = f"member_{idx:03d}"
        _write_yaml(member_config, member_dir / "config.resolved.yml")
        trainer = run_train(
            member_config,
            run_dir=str(member_dir),
            seed=seed,
            identifier=f"member_{idx:03d}",
            print_every=ensemble_cfg["train"].get("print_every"),
            amp=ensemble_cfg["train"].get("amp"),
            cpu=ensemble_cfg["train"].get("cpu"),
        )
        checkpoint = _checkpoint_path_for_trainer(trainer, checkpoint_name)
        member_record = {
            "index": idx,
            "seed": seed,
            "run_dir": str(member_dir),
            "config": str(member_dir / "config.resolved.yml"),
            "checkpoint": checkpoint,
            "predictions": {},
        }
        for split in evaluate_splits:
            if split not in split_sources:
                continue
            split_csv = member_dir / "ensemble_predictions" / f"{split}.csv"
            _predict_checkpoint_on_lmdb(
                base_config=base_config,
                checkpoint=checkpoint,
                lmdb=split_sources[split],
                output_csv=split_csv,
                run_dir=member_dir / "predict_runs" / split,
                seed=seed,
                cpu=ensemble_cfg["train"].get("cpu"),
                amp=ensemble_cfg["train"].get("amp"),
                hide_eval_progressbar=ensemble_cfg.get("hide_eval_progressbar", False),
            )
            member_record["predictions"][split] = str(split_csv)
        manifest["members"].append(member_record)
        _write_json(manifest, output_dir / "ensemble_manifest.json")

    metrics: dict[str, Any] = {}
    for split in evaluate_splits:
        prediction_files = [
            member["predictions"][split]
            for member in manifest["members"]
            if split in member["predictions"]
        ]
        if len(prediction_files) != len(manifest["members"]):
            continue
        aggregate_csv = aggregate_dir / f"{split}_ensemble.csv"
        rows = aggregate_ensemble_predictions(
            prediction_files,
            output_path=aggregate_csv,
            tasks=tasks,
            include_members=include_members,
        )
        manifest["aggregate"][split] = str(aggregate_csv)
        split_metrics = _ensemble_metrics(rows, tasks)
        if split_metrics:
            metrics[split] = split_metrics
    if metrics:
        _write_json(metrics, aggregate_dir / "ensemble_metrics.json")
        manifest["ensemble_metrics"] = str(aggregate_dir / "ensemble_metrics.json")
    _write_json(manifest, output_dir / "ensemble_manifest.json")
    return manifest


def run_ensemble_predict(
    *,
    manifest_path: str | Path,
    lmdb: str,
    output_dir: str | Path,
    tasks: list[str] | None = None,
    include_members: bool = True,
    hide_eval_progressbar: bool = False,
) -> dict[str, Any]:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    base_config = load_config(manifest["base_config"])
    output_dir = Path(output_dir).expanduser().resolve()
    members_out = output_dir / "members"
    members_out.mkdir(parents=True, exist_ok=True)
    tasks = tasks or manifest.get("tasks") or _task_names(base_config)

    prediction_files = []
    for member in manifest["members"]:
        member_csv = members_out / f"member_{int(member['index']):03d}.csv"
        _predict_checkpoint_on_lmdb(
            base_config=base_config,
            checkpoint=member["checkpoint"],
            lmdb=lmdb,
            output_csv=member_csv,
            run_dir=output_dir / "predict_runs" / f"member_{int(member['index']):03d}",
            seed=member.get("seed"),
            hide_eval_progressbar=hide_eval_progressbar,
        )
        prediction_files.append(str(member_csv))

    aggregate_csv = output_dir / "ensemble_predictions.csv"
    rows = aggregate_ensemble_predictions(
        prediction_files,
        output_path=aggregate_csv,
        tasks=tasks,
        include_members=include_members,
    )
    metrics = _ensemble_metrics(rows, tasks)
    if metrics:
        _write_json(metrics, output_dir / "ensemble_metrics.json")
    summary = {
        "manifest": str(Path(manifest_path).resolve()),
        "lmdb": lmdb,
        "output_dir": str(output_dir),
        "member_predictions": prediction_files,
        "ensemble_predictions": str(aggregate_csv),
        "metrics": str(output_dir / "ensemble_metrics.json") if metrics else None,
    }
    _write_json(summary, output_dir / "prediction_manifest.json")
    return summary

