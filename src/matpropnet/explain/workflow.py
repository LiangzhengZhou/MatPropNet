"""Single-checkpoint explainability workflow."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import torch

from matpropnet.tasks.core import _build_task, _build_trainer, _load_runtime_config
from .algorithms import MatPropNetEdgeMaskExplainer
from .metrics import regression_fidelity_metrics
from .outputs import save_sample_outputs, write_resolved_config
from .wrappers import GraphRegressionModelWrapper, unwrap_model


def _batch_one(data, device):
    try:
        from torch_geometric.data import Batch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "MatPropNet explainability requires torch-geometric."
        ) from exc
    batch = Batch.from_data_list([copy.copy(data)])
    if hasattr(data, "sample_id"):
        batch.sample_id = [str(data.sample_id)]
    if not hasattr(batch, "batch"):
        batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long)
    if not hasattr(batch, "neighbors") and hasattr(batch, "edge_index"):
        batch.neighbors = torch.tensor([batch.edge_index.shape[1]], dtype=torch.long)
    return batch.to(device)


def _sample_id(data, index: int) -> str:
    if hasattr(data, "sample_id"):
        return str(data.sample_id)
    if hasattr(data, "sid"):
        return str(data.sid)
    return str(index)


def _task_name(config: dict[str, Any], target_index: int) -> str | None:
    tasks = list(config.get("task", {}).get("tasks", {}).keys())
    if not tasks:
        return None
    return tasks[target_index] if target_index < len(tasks) else tasks[0]


def _normalizer_for_trainer(trainer):
    def denormalize(task_name: str, pred: torch.Tensor) -> torch.Tensor:
        if hasattr(trainer, "_denormalize_prediction"):
            return trainer._denormalize_prediction(task_name, pred)
        return pred

    return denormalize


def explain_checkpoint(
    config_or_path,
    *,
    checkpoint: str,
    lmdb: str,
    output_dir: str | Path,
    algorithm: str = "matpropnet_edge_mask",
    target_index: int = 0,
    num_samples: int = 20,
    top_k: int = 20,
    epochs: int = 100,
    lr: float = 0.01,
    edge_size_weight: float = 0.005,
    edge_entropy_weight: float = 0.001,
    cpu: bool | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if algorithm not in {"matpropnet_edge_mask", "auto"}:
        raise ValueError(f"Unsupported explain algorithm '{algorithm}'.")
    config = _load_runtime_config(
        config_or_path,
        mode="predict",
        checkpoint=checkpoint,
        cpu=cpu,
    )
    config["dataset"] = {"test": {"src": lmdb}}
    output_dir = Path(output_dir).expanduser().resolve()
    task_name = _task_name(config, target_index)
    if dry_run:
        return {
            "checkpoint": checkpoint,
            "lmdb": lmdb,
            "output_dir": str(output_dir),
            "algorithm": algorithm,
            "task_name": task_name,
            "num_samples": num_samples,
            "top_k": top_k,
        }

    trainer = _build_trainer(config)
    _build_task(config, trainer)
    summaries = []
    try:
        model = unwrap_model(trainer.model)
        wrapper = GraphRegressionModelWrapper(
            model,
            task_name=task_name,
            target_index=target_index,
            normalizer=_normalizer_for_trainer(trainer),
        ).to(trainer.device)
        explainer = MatPropNetEdgeMaskExplainer(
            epochs=epochs,
            lr=lr,
            edge_size_weight=edge_size_weight,
            edge_entropy_weight=edge_entropy_weight,
        )
        write_resolved_config(config, output_dir)
        dataset = trainer.test_dataset
        for sample_index in range(min(num_samples, len(dataset))):
            start = time.time()
            raw_data = dataset[sample_index]
            sample_id = _sample_id(raw_data, sample_index)
            data = _batch_one(raw_data, trainer.device)
            result = explainer.explain_data(wrapper, data)
            fidelity = regression_fidelity_metrics(
                wrapper, data, result.edge_mask, top_k
            )
            summary = {
                "sample_id": sample_id,
                "checkpoint": checkpoint,
                "lmdb": lmdb,
                "algorithm": result.algorithm,
                "task_name": task_name,
                "target_index": target_index,
                "top_k": top_k,
                "number_of_nodes": int(data.num_nodes),
                "number_of_edges": int(data.edge_index.shape[1]),
                "model_prediction": result.target_prediction,
                "masked_prediction": result.prediction,
                "fidelity": fidelity,
                "runtime_seconds": time.time() - start,
                "warnings": [],
            }
            save_sample_outputs(
                output_dir=output_dir / sample_id,
                data=data,
                edge_mask=result.edge_mask,
                sample_id=sample_id,
                top_k=top_k,
                summary=summary,
            )
            summaries.append(summary)
    finally:
        trainer.close_datasets()

    manifest = {
        "checkpoint": checkpoint,
        "lmdb": lmdb,
        "output_dir": str(output_dir),
        "algorithm": algorithm,
        "num_samples": len(summaries),
        "samples": [item["sample_id"] for item in summaries],
    }
    with (output_dir / "explain_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
