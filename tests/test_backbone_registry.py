from __future__ import annotations

import pytest

from ocpmodels.models.property_model import build_backbone


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("cgcnn", {}),
        ("schnet", {}),
        (
            "gemnet_t",
            {
                "num_spherical": 3,
                "num_radial": 16,
                "num_blocks": 2,
                "emb_size_atom": 64,
                "emb_size_edge": 128,
                "emb_size_trip": 32,
                "emb_size_rbf": 16,
                "emb_size_cbf": 16,
                "emb_size_bil_trip": 32,
                "num_before_skip": 1,
                "num_after_skip": 1,
                "num_concat": 1,
                "num_atom": 2,
            },
        ),
        ("dimenet", {"hidden_dim": 64}),
        ("dimenetplusplus", {"hidden_dim": 64}),
        ("forcenet", {"hidden_channels": 64, "decoder_hidden_channels": 64}),
        ("spinconv", {"hidden_channels": 16, "mid_hidden_channels": 32, "num_basis_functions": 16}),
    ],
)
def test_build_backbone_supports_expected_names(name, extra):
    config = {"name": name, "use_pbc": True, "otf_graph": False}
    config.update(extra)
    backbone = build_backbone(config, bond_feat_dim=50)
    assert hasattr(backbone, "forward")
    assert hasattr(backbone, "blocks")
    assert hasattr(backbone, "hidden_dim")


def test_gemnet_backbone_accepts_non_default_max_neighbors():
    config = {
        "name": "gemnet_t",
        "use_pbc": True,
        "otf_graph": False,
        "max_neighbors": 30,
        "num_spherical": 3,
        "num_radial": 16,
        "num_blocks": 2,
        "emb_size_atom": 64,
        "emb_size_edge": 128,
        "emb_size_trip": 32,
        "emb_size_rbf": 16,
        "emb_size_cbf": 16,
        "emb_size_bil_trip": 32,
        "num_before_skip": 1,
        "num_after_skip": 1,
        "num_concat": 1,
        "num_atom": 2,
    }

    backbone = build_backbone(config, bond_feat_dim=50)

    assert backbone.model.max_neighbors == 30


def test_gemnet_backbone_accepts_preprocessed_cutoff_above_default():
    config = {
        "name": "gemnet_t",
        "use_pbc": True,
        "otf_graph": False,
        "cutoff": 8.0,
        "max_neighbors": 80,
        "num_spherical": 3,
        "num_radial": 16,
        "num_blocks": 2,
        "emb_size_atom": 64,
        "emb_size_edge": 128,
        "emb_size_trip": 32,
        "emb_size_rbf": 16,
        "emb_size_cbf": 16,
        "emb_size_bil_trip": 32,
        "num_before_skip": 1,
        "num_after_skip": 1,
        "num_concat": 1,
        "num_atom": 2,
    }

    backbone = build_backbone(config, bond_feat_dim=50)

    assert backbone.model.cutoff == 8.0
    assert backbone.model.max_neighbors == 80
