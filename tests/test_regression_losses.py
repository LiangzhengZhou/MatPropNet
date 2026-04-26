from __future__ import annotations

import math

import torch

from matpropnet.modules.losses import build_regression_loss, normalize_loss_config


def test_normalize_loss_config_injects_defaults():
    config = normalize_loss_config({"name": "mae"})

    assert config["name"] == "mae"
    assert config["weight"]["type"] == "none"
    assert config["asymmetry"]["enabled"] is False


def test_huber_loss_matches_manual_definition():
    loss_fn = build_regression_loss(
        task_name="target",
        config={"name": "huber", "delta": 2.0},
        training_stats={},
    )
    pred = torch.tensor([0.0, 3.0], dtype=torch.float32)
    target = torch.tensor([1.0, 0.0], dtype=torch.float32)

    loss, stats = loss_fn(pred, target)

    expected = (0.5 * 1.0**2 + (2.0 * (3.0 - 1.0))) / 2.0
    assert torch.isclose(loss, torch.tensor(expected, dtype=torch.float32))
    assert math.isclose(stats["loss/base_loss/target"], expected, rel_tol=1e-6)


def test_smooth_l1_loss_matches_pytorch_none_reduction():
    loss_fn = build_regression_loss(
        task_name="target",
        config={"name": "smooth_l1", "beta": 0.5},
        training_stats={},
    )
    pred = torch.tensor([[0.0], [1.0], [3.0]], dtype=torch.float32)
    target = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32)

    loss, _ = loss_fn(pred, target)

    expected = torch.nn.functional.smooth_l1_loss(
        pred.reshape(-1), target, reduction="none", beta=0.5
    ).mean()
    assert torch.isclose(loss, expected)


def test_weighted_loss_applies_target_power_and_underestimation_penalty():
    loss_fn = build_regression_loss(
        task_name="H",
        config={
            "name": "mae",
            "weight": {
                "type": "target_power",
                "alpha": 2.0,
                "gamma": 1.0,
                "ref": "p95",
                "max_weight": 5.0,
            },
            "asymmetry": {
                "enabled": True,
                "mode": "underestimation",
                "factor": 2.0,
            },
        },
        training_stats={"p95": 10.0},
    )
    pred = torch.tensor([8.0, 12.0], dtype=torch.float32)
    target = torch.tensor([10.0, 10.0], dtype=torch.float32)

    loss, stats = loss_fn(pred, target)

    expected_weights = torch.tensor([6.0, 3.0], dtype=torch.float32)
    expected_weights = torch.clamp(expected_weights, max=5.0)
    expected_weights[0] = expected_weights[0] * 2.0
    expected_loss = (expected_weights * torch.tensor([2.0, 2.0])).sum() / expected_weights.sum()
    assert torch.isclose(loss, expected_loss)
    assert stats["loss/max_weight/H"] >= stats["loss/min_weight/H"]


def test_bin_balanced_loss_supports_batch_size_one():
    loss_fn = build_regression_loss(
        task_name="target",
        config={
            "name": "mse",
            "weight": {
                "type": "bin_balanced",
                "bins": [0, 5, 10, "inf"],
                "min_weight": 0.5,
                "max_weight": 5.0,
            },
        },
        training_stats={"values": [1.0, 2.0, 3.0, 8.0, 12.0]},
    )
    pred = torch.tensor([[4.0]], dtype=torch.float32)
    target = torch.tensor([[2.0]], dtype=torch.float32)

    loss, stats = loss_fn(pred, target)

    assert torch.isfinite(loss)
    assert loss.shape == ()
    assert "loss/weighted_loss/target" in stats


def test_log_cosh_loss_is_finite():
    loss_fn = build_regression_loss(
        task_name="target",
        config={"name": "log_cosh"},
        training_stats={},
    )
    pred = torch.tensor([100.0, -100.0], dtype=torch.float32)
    target = torch.tensor([0.0, 0.0], dtype=torch.float32)

    loss, _ = loss_fn(pred, target)

    assert torch.isfinite(loss)


def test_gaussian_nll_uses_log_variance():
    loss_fn = build_regression_loss(
        task_name="H",
        config={"name": "gaussian_nll", "min_log_var": -5.0, "max_log_var": 5.0},
        training_stats={},
    )
    pred = torch.tensor([1.0, 3.0], dtype=torch.float32)
    target = torch.tensor([0.0, 1.0], dtype=torch.float32)
    log_var = torch.tensor([0.0, 1.0], dtype=torch.float32)

    loss, stats = loss_fn(pred, target, log_var=log_var)

    expected = 0.5 * torch.exp(-log_var) * (pred - target).pow(2) + 0.5 * log_var
    assert torch.isclose(loss, expected.mean())
    assert "loss/log_var_mean/H" in stats
    assert "loss/sigma_mean/H" in stats
