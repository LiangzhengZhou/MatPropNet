from __future__ import annotations

from collections import OrderedDict

import torch

from ocpmodels.models.property_model import PropertyModel


def test_property_model_builds_gaussian_regression_head():
    model = PropertyModel(
        None,
        bond_feat_dim=50,
        num_targets=1,
        backbone={"name": "cgcnn", "hidden_dim": 16, "num_graph_conv_layers": 1},
        pooling={"name": "mean"},
        latent={"hidden_dim": 16, "out_dim": 8, "num_layers": 1},
        tasks=OrderedDict(
            {
                "H": {
                    "type": "regression",
                    "output": {"distribution": "gaussian"},
                }
            }
        ),
    )

    mu, log_var = model.heads["H"](torch.randn(4, 8))

    assert mu.shape == (4,)
    assert log_var.shape == (4,)
