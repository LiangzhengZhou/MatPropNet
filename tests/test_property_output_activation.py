from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ocpmodels.trainers.property_trainer import PropertyTrainer


def _trainer_for_activation(activation: str | None = None, normalizer=None):
    trainer = object.__new__(PropertyTrainer)
    task_spec = {"type": "regression"}
    if activation is not None:
        task_spec["output_activation"] = activation
    trainer.task_specs = {"H": task_spec}
    trainer.task_name_to_idx = {"H": 0}
    trainer.normalizers = {} if normalizer is None else {"target": normalizer}
    return trainer


def test_softplus_output_activation_without_normalizer():
    trainer = _trainer_for_activation("softplus")
    pred = torch.tensor([-4.0, 0.0, 2.0])
    out = {"task_preds": OrderedDict({"H": pred.clone()}), "pred": pred.clone()}

    activated = trainer._apply_output_activations(out)

    assert torch.allclose(activated["task_preds"]["H"], F.softplus(pred))
    assert torch.all(activated["pred"] >= 0.0)


def test_softplus_output_activation_preserves_normalized_training_space():
    normalizer = SimpleNamespace(
        mean=torch.tensor([10.0]),
        std=torch.tensor([2.0]),
    )
    trainer = _trainer_for_activation("softplus", normalizer=normalizer)
    pred_norm = torch.tensor([-10.0, -5.0, 0.0, 2.0])
    out = {"task_preds": OrderedDict({"H": pred_norm.clone()}), "pred": pred_norm.clone()}

    activated = trainer._apply_output_activations(out)
    activated_physical = activated["task_preds"]["H"] * normalizer.std[0] + normalizer.mean[0]
    expected_physical = F.softplus(pred_norm * normalizer.std[0] + normalizer.mean[0])

    assert torch.allclose(activated_physical, expected_physical, atol=1e-6, rtol=1e-5)
    assert torch.all(activated_physical >= 0.0)


def test_linear_output_activation_is_backward_compatible():
    trainer = _trainer_for_activation()
    pred = torch.tensor([-4.0, 0.0, 2.0])
    out = {"task_preds": OrderedDict({"H": pred.clone()}), "pred": pred.clone()}

    activated = trainer._apply_output_activations(out)

    assert torch.equal(activated["task_preds"]["H"], pred)


def test_softplus_output_activation_rejects_gaussian_uncertainty_head():
    trainer = _trainer_for_activation("softplus")
    pred = torch.tensor([1.0, 2.0])
    out = {
        "task_preds": OrderedDict({"H": pred}),
        "task_log_vars": OrderedDict({"H": torch.zeros_like(pred)}),
    }

    with pytest.raises(ValueError, match="Gaussian NLL"):
        trainer._apply_output_activations(out)
