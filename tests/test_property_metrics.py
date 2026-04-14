from collections import OrderedDict
from pathlib import Path
import tempfile

import torch

from matpropnet.tasks.core import _write_prediction_csv
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.trainers.property_trainer import PropertyTrainer


class DummyBatch:
    def __init__(self, y):
        self.y = y
        self.target_mask = torch.ones_like(y, dtype=torch.bool)

    def to(self, device):
        self.y = self.y.to(device)
        self.target_mask = self.target_mask.to(device)
        return self


def build_dummy_trainer():
    trainer = PropertyTrainer.__new__(PropertyTrainer)
    trainer.device = torch.device("cpu")
    trainer.task_specs = OrderedDict(
        {"target1": {"type": "regression", "weight": 1.0}}
    )
    trainer.task_name_to_idx = {"target1": 0}
    trainer.task_names = ["target1"]
    trainer.num_targets = 1
    trainer.normalizers = {}
    return trainer


def test_property_metrics_include_r2():
    trainer = build_dummy_trainer()
    batch = DummyBatch(torch.tensor([[1.0], [2.0], [3.0]]))
    out = {
        "task_preds": {"target1": torch.tensor([1.0, 2.0, 3.0])},
        "per_task_loss": {},
    }
    metrics = trainer._compute_metrics(out, [batch], metrics={})
    aggregated = trainer._aggregate_metrics(metrics)
    assert "target1_r2" in aggregated
    assert aggregated["target1_r2"]["metric"] == 1.0


def test_property_metrics_r2_handles_nonperfect_predictions():
    trainer = build_dummy_trainer()
    batch = DummyBatch(torch.tensor([[1.0], [2.0], [3.0]]))
    out = {
        "task_preds": {"target1": torch.tensor([1.0, 2.5, 2.0])},
        "per_task_loss": {},
    }
    metrics = trainer._compute_metrics(out, [batch], metrics={})
    aggregated = trainer._aggregate_metrics(metrics)
    assert aggregated["target1_r2"]["metric"] < 1.0


def test_evaluator_eval_uses_fresh_default_metrics():
    evaluator = Evaluator(task="is2re")
    prediction = {"energy": torch.tensor([1.0])}
    target = {"energy": torch.tensor([2.0])}

    first = evaluator.eval(prediction, target)
    second = evaluator.eval(prediction, target)

    assert first["energy_mae"]["numel"] == 1
    assert second["energy_mae"]["numel"] == 1


def test_predict_csv_uses_threshold_for_single_logit_binary_classification():
    predictions = {
        "id": ["sample-1", "sample-2"],
        "pred_label": [1, 0],
        "target_label": [1.0, 0.0],
        "prob_label": [0.8, 0.2],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "predictions.csv"
        _write_prediction_csv(predictions, str(csv_path))
        contents = csv_path.read_text(encoding="utf-8")

    assert "pred_label" in contents
    assert "prob_label" in contents
    assert "0.8" in contents
