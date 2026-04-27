from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from matpropnet.ensemble import run_ensemble_predict, run_ensemble_train


def _write_yaml(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _base_config(tmp_path: Path) -> dict:
    return {
        "trainer": "property",
        "logger": "tensorboard",
        "dataset": [
            {"src": str(tmp_path / "train" / "data.lmdb"), "target_mean": [0.0], "target_std": [1.0]},
            {"src": str(tmp_path / "val" / "data.lmdb")},
            {"src": str(tmp_path / "test" / "data.lmdb")},
        ],
        "model": {
            "name": "property_model",
            "backbone": {"name": "cgcnn", "hidden_dim": 16, "num_graph_conv_layers": 1},
            "pooling": {"name": "mean"},
            "latent": {"hidden_dim": 16, "out_dim": 8, "num_layers": 1},
        },
        "task": {
            "dataset": "property_lmdb",
            "primary_metric": "H_mae",
            "tasks": {
                "H": {
                    "type": "regression",
                    "loss": {"name": "gaussian_nll"},
                }
            },
        },
        "optim": {
            "batch_size": 1,
            "eval_batch_size": 1,
            "num_workers": 0,
            "lr_initial": 1.0e-3,
            "max_epochs": 1,
        },
    }


def _ensemble_config(tmp_path: Path, base_config_path: Path) -> dict:
    return {
        "ensemble": {
            "name": "test_ensemble",
            "base_config": str(base_config_path),
            "output_dir": str(tmp_path / "ensemble_run"),
            "num_members": 2,
            "seeds": [11, 23],
            "train": {"checkpoint_name": "best_checkpoint.pt"},
            "evaluate": {"splits": ["test"]},
            "aggregate": {"tasks": ["H"], "include_members": True},
        }
    }


def test_ensemble_train_dry_run_expands_members(tmp_path):
    base_path = tmp_path / "base.yml"
    ensemble_path = tmp_path / "ensemble.yml"
    _write_yaml(base_path, _base_config(tmp_path))
    _write_yaml(ensemble_path, _ensemble_config(tmp_path, base_path))

    plan = run_ensemble_train(ensemble_path, dry_run=True)

    assert plan["dry_run"] is True
    assert plan["num_members"] == 2
    assert [member["seed"] for member in plan["planned_members"]] == [11, 23]
    assert Path(plan["planned_members"][0]["log_file"]).parts[-2:] == (
        "member_000",
        "train.log",
    )


def test_ensemble_train_writes_manifest_and_aggregate(monkeypatch, tmp_path):
    base_path = tmp_path / "base.yml"
    ensemble_path = tmp_path / "ensemble.yml"
    _write_yaml(base_path, _base_config(tmp_path))
    _write_yaml(ensemble_path, _ensemble_config(tmp_path, base_path))

    class DummyTrainer:
        def __init__(self, checkpoint_dir):
            self.config = {"cmd": {"checkpoint_dir": str(checkpoint_dir)}}

    def fake_run_train(config, **kwargs):
        checkpoint_dir = Path(kwargs["run_dir"]) / "checkpoints" / f"seed_{kwargs['seed']}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "best_checkpoint.pt").write_text("checkpoint", encoding="utf-8")
        return DummyTrainer(checkpoint_dir)

    def fake_predict(**kwargs):
        output_csv = Path(kwargs["output_csv"])
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        member_value = 1.0 if "member_000" in str(output_csv) else 3.0
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["id", "pred_H", "pred_H_sigma", "target_H"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "id": "sample-1",
                    "pred_H": member_value,
                    "pred_H_sigma": 0.5,
                    "target_H": 2.0,
                }
            )
        return {}

    monkeypatch.setattr("matpropnet.ensemble.workflow.run_train", fake_run_train)
    monkeypatch.setattr(
        "matpropnet.ensemble.workflow._predict_checkpoint_on_lmdb", fake_predict
    )

    manifest = run_ensemble_train(ensemble_path)

    manifest_path = tmp_path / "ensemble_run" / "ensemble_manifest.json"
    aggregate_path = tmp_path / "ensemble_run" / "aggregate" / "test_ensemble.csv"
    assert manifest_path.exists()
    assert aggregate_path.exists()
    assert manifest["num_members"] == 2
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(loaded["members"]) == 2
    assert loaded["aggregate"]["test"] == str(aggregate_path)
    assert Path(loaded["members"][0]["log_file"]).parts[-2:] == (
        "member_000",
        "train.log",
    )
    assert (tmp_path / "ensemble_run" / "members" / "member_000" / "train.log").exists()


def test_ensemble_train_can_calibrate_uncertainty_from_val(monkeypatch, tmp_path):
    base_path = tmp_path / "base.yml"
    ensemble_path = tmp_path / "ensemble.yml"
    ensemble_cfg = _ensemble_config(tmp_path, base_path)
    ensemble_cfg["ensemble"]["evaluate"]["splits"] = ["val", "test"]
    ensemble_cfg["ensemble"]["calibration"] = {"enabled": True, "source_split": "val"}
    _write_yaml(base_path, _base_config(tmp_path))
    _write_yaml(ensemble_path, ensemble_cfg)

    class DummyTrainer:
        def __init__(self, checkpoint_dir):
            self.config = {"cmd": {"checkpoint_dir": str(checkpoint_dir)}}

    def fake_run_train(config, **kwargs):
        checkpoint_dir = Path(kwargs["run_dir"]) / "checkpoints" / f"seed_{kwargs['seed']}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "best_checkpoint.pt").write_text("checkpoint", encoding="utf-8")
        return DummyTrainer(checkpoint_dir)

    def fake_predict(**kwargs):
        output_csv = Path(kwargs["output_csv"])
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        target = 3.0 if output_csv.name == "val.csv" else 1.0
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["id", "pred_H", "pred_H_sigma", "target_H"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "id": "sample-1",
                    "pred_H": 1.0,
                    "pred_H_sigma": 1.0,
                    "target_H": target,
                }
            )
        return {}

    monkeypatch.setattr("matpropnet.ensemble.workflow.run_train", fake_run_train)
    monkeypatch.setattr(
        "matpropnet.ensemble.workflow._predict_checkpoint_on_lmdb", fake_predict
    )

    manifest = run_ensemble_train(ensemble_path)

    assert manifest["uncertainty_calibration"]["tasks"]["H"]["variance_scale"] == 4.0
    with (tmp_path / "ensemble_run" / "aggregate" / "test_ensemble.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))
    assert float(row["pred_H_var_total_calibrated"]) == 4.0


def test_ensemble_train_can_disable_member_log_file(monkeypatch, tmp_path):
    base_path = tmp_path / "base.yml"
    ensemble_path = tmp_path / "ensemble.yml"
    ensemble_cfg = _ensemble_config(tmp_path, base_path)
    ensemble_cfg["ensemble"]["train"]["log_file_name"] = None
    _write_yaml(base_path, _base_config(tmp_path))
    _write_yaml(ensemble_path, ensemble_cfg)

    class DummyTrainer:
        def __init__(self, checkpoint_dir):
            self.config = {"cmd": {"checkpoint_dir": str(checkpoint_dir)}}

    def fake_run_train(config, **kwargs):
        checkpoint_dir = Path(kwargs["run_dir"]) / "checkpoints" / f"seed_{kwargs['seed']}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "best_checkpoint.pt").write_text("checkpoint", encoding="utf-8")
        return DummyTrainer(checkpoint_dir)

    def fake_predict(**kwargs):
        output_csv = Path(kwargs["output_csv"])
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "pred_H"])
            writer.writeheader()
            writer.writerow({"id": "sample-1", "pred_H": 1.0})
        return {}

    monkeypatch.setattr("matpropnet.ensemble.workflow.run_train", fake_run_train)
    monkeypatch.setattr(
        "matpropnet.ensemble.workflow._predict_checkpoint_on_lmdb", fake_predict
    )

    manifest = run_ensemble_train(ensemble_path)

    assert manifest["members"][0]["log_file"] is None
    assert not (tmp_path / "ensemble_run" / "members" / "member_000" / "train.log").exists()


def test_ensemble_predict_uses_manifest_members(monkeypatch, tmp_path):
    base_path = tmp_path / "base.yml"
    _write_yaml(base_path, _base_config(tmp_path))
    checkpoint0 = tmp_path / "member0.pt"
    checkpoint1 = tmp_path / "member1.pt"
    checkpoint0.write_text("checkpoint", encoding="utf-8")
    checkpoint1.write_text("checkpoint", encoding="utf-8")
    manifest_path = tmp_path / "ensemble_manifest.json"
    manifest = {
        "base_config": str(base_path),
        "tasks": ["H"],
        "members": [
            {"index": 0, "seed": 11, "checkpoint": str(checkpoint0)},
            {"index": 1, "seed": 23, "checkpoint": str(checkpoint1)},
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_predict(**kwargs):
        output_csv = Path(kwargs["output_csv"])
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        member_value = 1.0 if output_csv.name == "member_000.csv" else 3.0
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "pred_H"])
            writer.writeheader()
            writer.writerow({"id": "sample-1", "pred_H": member_value})
        return {}

    monkeypatch.setattr(
        "matpropnet.ensemble.workflow._predict_checkpoint_on_lmdb", fake_predict
    )

    summary = run_ensemble_predict(
        manifest_path=manifest_path,
        lmdb=str(tmp_path / "new" / "data.lmdb"),
        output_dir=tmp_path / "predict_out",
    )

    assert Path(summary["ensemble_predictions"]).exists()
    assert (tmp_path / "predict_out" / "prediction_manifest.json").exists()
