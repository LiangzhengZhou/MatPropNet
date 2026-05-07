"""Deep ensemble explanation workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from matpropnet.config import load_config
from .outputs import save_sample_outputs
from .workflow import _batch_one, _normalizer_for_trainer, _sample_id, _task_name
from .algorithms import MatPropNetEdgeMaskExplainer
from .metrics import regression_fidelity_metrics
from .wrappers import GraphRegressionModelWrapper, unwrap_model
from matpropnet.tasks.core import _build_task, _build_trainer, _load_runtime_config


def _load_member_wrapper(base_config, member, lmdb, target_index, cpu):
    config = _load_runtime_config(
        base_config,
        mode="predict",
        checkpoint=member["checkpoint"],
        seed=member.get("seed"),
        cpu=cpu,
    )
    config["dataset"] = {"test": {"src": lmdb}}
    trainer = _build_trainer(config)
    _build_task(config, trainer)
    task_name = _task_name(config, target_index)
    wrapper = GraphRegressionModelWrapper(
        unwrap_model(trainer.model),
        task_name=task_name,
        target_index=target_index,
        normalizer=_normalizer_for_trainer(trainer),
    ).to(trainer.device)
    return trainer, wrapper, task_name


def explain_ensemble(
    *,
    manifest_path: str | Path,
    lmdb: str,
    output_dir: str | Path,
    algorithm: str = "matpropnet_edge_mask",
    target_index: int = 0,
    num_samples: int = 20,
    top_k: int = 20,
    epochs: int = 100,
    lr: float = 0.01,
    repeat: int = 1,
    cpu: bool | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if algorithm not in {"matpropnet_edge_mask", "auto"}:
        raise ValueError(f"Unsupported explain algorithm '{algorithm}'.")
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    base_config = load_config(manifest["base_config"])
    output_dir = Path(output_dir).expanduser().resolve()
    if dry_run:
        return {
            "manifest": str(Path(manifest_path).resolve()),
            "lmdb": lmdb,
            "output_dir": str(output_dir),
            "num_members": len(manifest["members"]),
            "num_samples": num_samples,
            "top_k": top_k,
        }

    member_contexts = []
    try:
        for member in manifest["members"]:
            member_contexts.append(
                (member, *_load_member_wrapper(base_config, member, lmdb, target_index, cpu))
            )
        first_trainer = member_contexts[0][1]
        dataset = first_trainer.test_dataset
        explainer = MatPropNetEdgeMaskExplainer(epochs=epochs, lr=lr)
        sample_ids = []
        for sample_index in range(min(num_samples, len(dataset))):
            raw_data = dataset[sample_index]
            sample_id = _sample_id(raw_data, sample_index)
            sample_ids.append(sample_id)
            member_masks = []
            member_predictions = []
            data_for_output = None
            task_name = None
            for member, trainer, wrapper, member_task in member_contexts:
                task_name = member_task
                data = _batch_one(raw_data, trainer.device)
                data_for_output = data if data_for_output is None else data_for_output
                repeated_masks = []
                for _ in range(max(1, repeat)):
                    result = explainer.explain_data(wrapper, data)
                    repeated_masks.append(result.edge_mask.detach().cpu())
                member_mask = torch.stack(repeated_masks).mean(dim=0)
                member_masks.append(member_mask)
                with torch.no_grad():
                    member_predictions.append(wrapper(data).detach().cpu())
            masks = torch.stack(member_masks, dim=0)
            mean_mask = masks.mean(dim=0)
            std_mask = masks.std(dim=0, unbiased=False)
            relative = std_mask / (mean_mask + 1.0e-8)
            confidence = mean_mask / (std_mask + 1.0e-8)
            ensemble_columns = {
                "ensemble_mean_mask": mean_mask.tolist(),
                "ensemble_std_mask": std_mask.tolist(),
                "relative_uncertainty": relative.tolist(),
                "confidence": confidence.tolist(),
            }
            for idx, member_mask in enumerate(member_masks):
                ensemble_columns[f"member_{idx:03d}_mask"] = member_mask.tolist()
            fidelity = regression_fidelity_metrics(
                member_contexts[0][2],
                data_for_output,
                mean_mask.to(data_for_output.edge_index.device),
                top_k,
            )
            summary = {
                "sample_id": sample_id,
                "ensemble_manifest": str(Path(manifest_path).resolve()),
                "algorithm": algorithm,
                "task_name": task_name,
                "target_index": target_index,
                "top_k": top_k,
                "number_of_members": len(member_contexts),
                "number_of_nodes": int(data_for_output.num_nodes),
                "number_of_edges": int(data_for_output.edge_index.shape[1]),
                "member_predictions": member_predictions,
                "ensemble_prediction_mean": torch.stack(member_predictions).mean(dim=0),
                "ensemble_prediction_std": torch.stack(member_predictions).std(
                    dim=0, unbiased=False
                ),
                "fidelity": fidelity,
                "warnings": [],
            }
            save_sample_outputs(
                output_dir=output_dir / sample_id,
                data=data_for_output.cpu(),
                edge_mask=mean_mask,
                sample_id=sample_id,
                top_k=top_k,
                summary=summary,
                extra_masks={"member_masks": masks, "ensemble_std_mask": std_mask},
                ensemble_columns=ensemble_columns,
            )
    finally:
        for _, trainer, _, _ in member_contexts:
            trainer.close_datasets()

    out_manifest = {
        "manifest": str(Path(manifest_path).resolve()),
        "lmdb": lmdb,
        "output_dir": str(output_dir),
        "algorithm": algorithm,
        "num_members": len(manifest["members"]),
        "num_samples": len(sample_ids),
        "samples": sample_ids,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ensemble_explain_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(out_manifest, handle, indent=2)
    return out_manifest
