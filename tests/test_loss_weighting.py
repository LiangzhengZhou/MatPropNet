from __future__ import annotations

import torch

from ocpmodels.modules.loss_weighting import (
    GradNormLossWeighting,
    StaticLossWeighting,
    UncertaintyLossWeighting,
    build_loss_weighting_strategy,
)


TASK_SPECS = {
    "target1": {"type": "regression", "weight": 2.0},
    "target2": {"type": "regression", "weight": 0.5},
}


def test_static_loss_weighting_matches_manual_weights():
    strategy = StaticLossWeighting(list(TASK_SPECS), TASK_SPECS)
    task_losses = {
        "target1": torch.tensor(2.0),
        "target2": torch.tensor(4.0),
    }
    total_loss, stats = strategy.compute_weighted_loss(task_losses, {})
    assert torch.isclose(total_loss, torch.tensor(6.0))
    assert stats["loss_weighting/weight/target1"] == 2.0
    assert stats["loss_weighting/weight/target2"] == 0.5


def test_uncertainty_loss_weighting_state_roundtrip():
    strategy = UncertaintyLossWeighting(
        list(TASK_SPECS), TASK_SPECS, init_log_var=0.2
    )
    with torch.no_grad():
        strategy.log_vars["target1"].copy_(torch.tensor(0.7))
        strategy.log_vars["target2"].copy_(torch.tensor(-0.3))

    restored = UncertaintyLossWeighting(
        list(TASK_SPECS), TASK_SPECS, init_log_var=0.0
    )
    restored.load_state_dict(strategy.state_dict())

    assert torch.allclose(
        restored.log_vars["target1"].detach(),
        strategy.log_vars["target1"].detach(),
    )
    assert torch.allclose(
        restored.log_vars["target2"].detach(),
        strategy.log_vars["target2"].detach(),
    )


def test_gradnorm_runs_and_updates_weight_state():
    shared = torch.nn.Linear(4, 2)
    strategy = GradNormLossWeighting(list(TASK_SPECS), TASK_SPECS, lr=1.0e-2)
    inputs = torch.randn(6, 4)
    outputs = shared(inputs)
    task_losses = {
        "target1": outputs.pow(2).mean(),
        "target2": (outputs[:, 0] - 1.0).abs().mean(),
    }

    total_loss, stats = strategy.compute_weighted_loss(
        task_losses,
        {
            "is_training": True,
            "global_step": 2,
            "shared_params": [param for param in shared.parameters()],
        },
    )
    total_loss.backward(retain_graph=True)
    extra = strategy.on_after_backward({"is_training": True, "global_step": 2})

    assert torch.isfinite(total_loss)
    assert "loss_weighting/gradnorm_loss" in stats
    assert "loss_weighting/mean_grad_norm" in extra
    assert torch.isfinite(strategy.logits).all()


def test_gradnorm_skips_auxiliary_update_for_single_active_task():
    shared = torch.nn.Linear(4, 2)
    strategy = GradNormLossWeighting(list(TASK_SPECS), TASK_SPECS)
    outputs = shared(torch.randn(4, 4))
    total_loss, stats = strategy.compute_weighted_loss(
        {"target1": outputs.pow(2).mean()},
        {
            "is_training": True,
            "global_step": 5,
            "shared_params": [param for param in shared.parameters()],
        },
    )
    assert torch.isfinite(total_loss)
    assert "loss_weighting/gradnorm_loss" not in stats


def test_factory_builds_default_static_mode():
    strategy = build_loss_weighting_strategy({}, TASK_SPECS)
    assert isinstance(strategy, StaticLossWeighting)
