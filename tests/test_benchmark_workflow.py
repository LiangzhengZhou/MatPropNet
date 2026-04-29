from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from matpropnet.benchmark import run_benchmark


def _write_yaml(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _base_config(tmp_path: Path, name: str = "cgcnn") -> dict:
    return {
        "trainer": "property",
        "logger": "tensorboard",
        "dataset": [
            {"src": str(tmp_path / "train" / "data.lmdb")},
            {"src": str(tmp_path / "val" / "data.lmdb")},
            {"src": str(tmp_path / "test" / "data.lmdb")},
        ],
        "model": {
            "name": "property_model",
            "backbone": {
                "name": name,
                "hidden_dim": 16,
                "num_graph_conv_layers": 1,
            },
            "pooling": {"name": "mean"},
            "latent": {"hidden_dim": 16, "out_dim": 8, "num_layers": 1},
        },
        "task": {
            "dataset": "property_lmdb",
            "primary_metric": "H_mae",
            "tasks": {"H": {"type": "regression", "loss": "mae"}},
        },
        "optim": {
            "batch_size": 1,
            "eval_batch_size": 1,
            "num_workers": 0,
            "lr_initial": 1.0e-3,
            "max_epochs": 1,
        },
    }


def _benchmark_config(tmp_path: Path, config_paths: list[Path]) -> dict:
    return {
        "benchmark": {
            "name": "test_benchmark",
            "output_dir": str(tmp_path / "benchmark_run"),
            "seed": 42,
            "train": {"checkpoint_name": "best_checkpoint.pt"},
            "evaluate": {"splits": ["val", "test"]},
            "models": [
                {"name": f"model_{idx}", "config": str(path)}
                for idx, path in enumerate(config_paths)
            ],
        }
    }


def test_benchmark_dry_run_expands_models(tmp_path):
    model_config = tmp_path / "model.yml"
    benchmark_config = tmp_path / "benchmark.yml"
    _write_yaml(model_config, _base_config(tmp_path))
    _write_yaml(benchmark_config, _benchmark_config(tmp_path, [model_config]))

    plan = run_benchmark(benchmark_config, dry_run=True)

    assert plan["dry_run"] is True
    assert plan["seed"] == 42
    assert plan["planned_models"][0]["run_name"] == "model_0"
    assert Path(plan["planned_models"][0]["log_file"]).parts[-2:] == (
        "model_0",
        "train.log",
    )


def test_benchmark_writes_summary_and_manifest(monkeypatch, tmp_path):
    config0 = tmp_path / "model0.yml"
    config1 = tmp_path / "model1.yml"
    benchmark_config = tmp_path / "benchmark.yml"
    _write_yaml(config0, _base_config(tmp_path, name="cgcnn"))
    _write_yaml(config1, _base_config(tmp_path, name="schnet"))
    _write_yaml(benchmark_config, _benchmark_config(tmp_path, [config0, config1]))

    class DummyTrainer:
        def __init__(self, checkpoint_dir: Path, results_dir: Path):
            self.config = {
                "cmd": {
                    "checkpoint_dir": str(checkpoint_dir),
                    "results_dir": str(results_dir),
                }
            }
            prediction_dir = results_dir / "property_predictions"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            (prediction_dir / "test.csv").write_text(
                "sample_id,H_target,H_pred\nsample-1,2.0,1.5\n",
                encoding="utf-8",
            )
            self.closed = False

        def validate(self, split="val", disable_tqdm=True):
            raise AssertionError("benchmark must not reuse the closed train trainer")

        def close_datasets(self):
            self.closed = True

    def fake_run_train(config, **kwargs):
        checkpoint_dir = Path(kwargs["run_dir"]) / "checkpoints" / "fake"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "best_checkpoint.pt").write_text(
            "checkpoint", encoding="utf-8"
        )
        return DummyTrainer(checkpoint_dir, Path(kwargs["run_dir"]) / "results")

    def fake_evaluate_checkpoint(**kwargs):
        assert kwargs["checkpoint"].endswith("best_checkpoint.pt")
        assert kwargs["run_name"] in {"model_0", "model_1"}
        return {
            "val": {
                "H_mae": {"metric": 1.0},
                "H_rmse": {"metric": 1.5},
                "loss": {"metric": 2.0},
            },
            "test": {
                "H_mae": {"metric": 2.0},
                "H_rmse": {"metric": 2.5},
                "loss": {"metric": 3.0},
            },
        }

    monkeypatch.setattr("matpropnet.benchmark.workflow.run_train", fake_run_train)
    monkeypatch.setattr(
        "matpropnet.benchmark.workflow._evaluate_checkpoint",
        fake_evaluate_checkpoint,
    )

    manifest = run_benchmark(benchmark_config)

    summary_path = tmp_path / "benchmark_run" / "summary" / "benchmark_summary.csv"
    manifest_path = tmp_path / "benchmark_run" / "benchmark_manifest.json"
    assert summary_path.exists()
    assert manifest_path.exists()
    assert len(manifest["models"]) == 2
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["models"][0]["status"] == "completed"
    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "completed"
    assert float(rows[0]["val_H_mae"]) == 1.0
    assert float(rows[0]["test_H_rmse"]) == 2.5
    assert Path(rows[0]["test_predictions"]).parts[-3:] == (
        "results",
        "property_predictions",
        "test.csv",
    )


def test_benchmark_records_failed_model_when_stop_on_error_false(
    monkeypatch, tmp_path
):
    config0 = tmp_path / "model0.yml"
    config1 = tmp_path / "model1.yml"
    benchmark_config = tmp_path / "benchmark.yml"
    _write_yaml(config0, _base_config(tmp_path, name="cgcnn"))
    _write_yaml(config1, _base_config(tmp_path, name="schnet"))
    payload = _benchmark_config(tmp_path, [config0, config1])
    payload["benchmark"]["models"][1]["name"] = "will_fail"
    _write_yaml(benchmark_config, payload)

    class DummyTrainer:
        def __init__(self, checkpoint_dir: Path):
            self.config = {"cmd": {"checkpoint_dir": str(checkpoint_dir)}}

        def validate(self, split="val", disable_tqdm=True):
            raise AssertionError("benchmark must not reuse the closed train trainer")

        def close_datasets(self):
            pass

    def fake_run_train(config, **kwargs):
        if kwargs["identifier"] == "will_fail":
            raise RuntimeError("boom")
        checkpoint_dir = Path(kwargs["run_dir"]) / "checkpoints" / "fake"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "best_checkpoint.pt").write_text(
            "checkpoint", encoding="utf-8"
        )
        return DummyTrainer(checkpoint_dir)

    monkeypatch.setattr("matpropnet.benchmark.workflow.run_train", fake_run_train)
    monkeypatch.setattr(
        "matpropnet.benchmark.workflow._evaluate_checkpoint",
        lambda **kwargs: {"val": {"H_mae": {"metric": 1.0}}},
    )

    manifest = run_benchmark(benchmark_config)

    assert [row["status"] for row in manifest["models"]] == [
        "completed",
        "failed",
    ]
    assert "RuntimeError: boom" in manifest["models"][1]["error"]
