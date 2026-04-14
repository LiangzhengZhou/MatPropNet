"""Python API task entrypoints for MatPropNet."""

from __future__ import annotations

import copy
import csv
import tempfile
from argparse import Namespace
from pathlib import Path

from matpropnet.config import load_config
from matpropnet.preprocessing import run_preprocess as run_preprocess_job
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import setup_imports


def _finalize_runtime_config(
    config: dict,
    *,
    mode: str,
    checkpoint: str | None = None,
    run_dir: str | None = None,
    identifier: str | None = None,
    seed: int | None = None,
    print_every: int | None = None,
    amp: bool | None = None,
    cpu: bool | None = None,
):
    cfg = copy.deepcopy(config)
    cfg["mode"] = mode
    cfg["checkpoint"] = checkpoint or cfg.get("checkpoint")
    cfg["run_dir"] = run_dir or cfg.get("run_dir") or cfg.get("config_dir")
    cfg["identifier"] = (
        identifier if identifier is not None else cfg.get("identifier", "")
    )
    cfg["seed"] = seed if seed is not None else cfg.get("seed", 0)
    cfg["print_every"] = (
        print_every if print_every is not None else cfg.get("print_every", 10)
    )
    cfg["amp"] = amp if amp is not None else cfg.get("amp", False)
    cfg["cpu"] = cpu if cpu is not None else cfg.get("cpu", False)
    cfg["submit"] = False
    cfg["summit"] = False
    cfg["local_rank"] = 0
    cfg["distributed_port"] = cfg.get("distributed_port", 13356)
    cfg["world_size"] = 1
    cfg["distributed_backend"] = cfg.get("distributed_backend", "nccl")
    return cfg


def _build_trainer(config: dict):
    setup_imports()
    trainer_cls = registry.get_trainer_class(config.get("trainer", "property"))
    return trainer_cls(
        task=config["task"],
        model=config["model"],
        dataset=config.get("dataset"),
        optimizer=config["optim"],
        loss=config.get("loss"),
        identifier=config.get("identifier", ""),
        timestamp_id=config.get("timestamp_id"),
        run_dir=config.get("run_dir"),
        is_debug=config.get("is_debug", False),
        print_every=config.get("print_every", 10),
        seed=config.get("seed", 0),
        logger=config.get("logger", "tensorboard"),
        local_rank=config.get("local_rank", 0),
        amp=config.get("amp", False),
        cpu=config.get("cpu", False),
        slurm=config.get("slurm", {}),
        extra_config=config,
    )


def _build_task(config: dict, trainer):
    task = registry.get_task_class(config["mode"])(config)
    task.setup(trainer)
    return task


def _load_runtime_config(config_or_path, overrides=None, **runtime_kwargs):
    if isinstance(config_or_path, (str, Path)):
        config = load_config(config_or_path, overrides=overrides)
    else:
        config = copy.deepcopy(config_or_path)
    return _finalize_runtime_config(config, **runtime_kwargs)


def run_train(
    config_or_path,
    overrides=None,
    dry_run: bool = False,
    **runtime_kwargs,
):
    config = _load_runtime_config(
        config_or_path, overrides=overrides, mode="train", **runtime_kwargs
    )
    if dry_run:
        return config
    trainer = _build_trainer(config)
    task = _build_task(config, trainer)
    task.run()
    return trainer


def run_eval(
    config_or_path,
    checkpoint: str,
    overrides=None,
    dry_run: bool = False,
    **runtime_kwargs,
):
    config = _load_runtime_config(
        config_or_path,
        overrides=overrides,
        mode="validate",
        checkpoint=checkpoint,
        **runtime_kwargs,
    )
    if dry_run:
        return config
    trainer = _build_trainer(config)
    task = _build_task(config, trainer)
    task.run()
    return trainer


def _infer_task_columns(config: dict):
    return list(config.get("task", {}).get("tasks", {}).keys())


def _ensure_prediction_csv_schema(
    input_csv: str, output_csv: str, required_columns: list[str]
):
    with open(input_csv, "r", encoding="utf-8", newline="") as src_handle:
        reader = csv.DictReader(src_handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [column for column in required_columns if column not in fieldnames]
    if not missing:
        return input_csv
    patched_fieldnames = fieldnames + missing
    with open(output_csv, "w", encoding="utf-8", newline="") as dst_handle:
        writer = csv.DictWriter(dst_handle, fieldnames=patched_fieldnames)
        writer.writeheader()
        for row in rows:
            for column in missing:
                row[column] = ""
            writer.writerow(row)
    return output_csv


def _write_prediction_csv(predictions: dict, output_path: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(predictions.keys())
    row_count = len(predictions["id"])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(row_count):
            writer.writerow({key: predictions[key][idx] for key in fieldnames})


def run_predict(
    config_or_path,
    checkpoint: str,
    input_csv: str | None = None,
    output: str | None = None,
    overrides=None,
    dry_run: bool = False,
    **runtime_kwargs,
):
    config = _load_runtime_config(
        config_or_path,
        overrides=overrides,
        mode="predict",
        checkpoint=checkpoint,
        **runtime_kwargs,
    )

    temp_dir = None
    try:
        if input_csv is not None:
            temp_dir = tempfile.TemporaryDirectory(prefix="matpropnet_predict_")
            working_input = _ensure_prediction_csv_schema(
                input_csv,
                str(Path(temp_dir.name) / "predict_input.csv"),
                _infer_task_columns(config),
            )
            task_specs = list(config.get("task", {}).get("tasks", {}).values())
            preprocess_args = Namespace(
                csv=working_input,
                out_path=str(Path(temp_dir.name) / "predict"),
                out_root=None,
                id_column="id",
                cif_column="cif",
                target_columns=",".join(_infer_task_columns(config)),
                task_types=",".join(
                    spec.get("type", "regression") for spec in task_specs
                ),
                num_classes=",".join(
                    str(spec.get("num_classes", 2))
                    for spec in task_specs
                    if spec.get("type") == "classification"
                )
                or None,
                cif_mode="auto",
                cif_root=None,
                radius=config.get("model", {}).get("backbone", {}).get("cutoff", 6.0),
                max_neigh=config.get("model", {}).get("backbone", {}).get(
                    "max_neighbors", 50
                ),
                get_edges=not config.get("model", {}).get("backbone", {}).get(
                    "otf_graph", False
                ),
                skip_failed=False,
                map_size_gb=1,
                split=None,
                split_seed=0,
                split_column=None,
                kfolds=0,
                fold_val_ratio=0.1,
            )
            run_preprocess_job(preprocess_args)
            config["dataset"] = {
                "test": {
                    "src": str(Path(temp_dir.name) / "predict" / "data.lmdb")
                }
            }

        if dry_run:
            return config

        trainer = _build_trainer(config)
        task = _build_task(config, trainer)
        if input_csv is None:
            task.run()
            return trainer

        predictions = trainer.predict(
            trainer.test_loader,
            results_file=None,
            disable_tqdm=config.get("hide_eval_progressbar", False),
        )
        trainer.close_datasets()
        if output:
            _write_prediction_csv(predictions, output)
        return predictions
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def run_preprocess(**kwargs):
    args = Namespace(**kwargs)
    return run_preprocess_job(args)
