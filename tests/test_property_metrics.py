from collections import OrderedDict

import torch

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
