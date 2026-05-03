from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from ocpmodels.models.property_model import SchNetBackbone, build_backbone
from ocpmodels.models.spinconv import (
    ProjectLatLongSphere,
    _element_table_size,
    _sort_edges_by_target,
)


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


def test_original_dimenet_points_users_to_dimenetplusplus():
    with pytest.raises(ValueError, match="dimenetplusplus"):
        build_backbone({"name": "dimenet"}, bond_feat_dim=50)


def test_schnet_backbone_returns_hidden_node_embeddings(monkeypatch):
    monkeypatch.setattr(
        "ocpmodels.models.property_model.radius_graph",
        lambda pos, r, batch: torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]],
            dtype=torch.long,
            device=pos.device,
        ),
    )
    backbone = SchNetBackbone(
        hidden_dim=16,
        num_filters=16,
        num_interactions=1,
        num_gaussians=8,
        cutoff=5.0,
        use_pbc=False,
    )
    data = Data(
        atomic_numbers=torch.tensor([6, 8, 14], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2], [1.1, 0.0, 0.0]],
            dtype=torch.float,
        ),
        batch=torch.zeros(3, dtype=torch.long),
    )

    out = backbone(data)

    assert out["node_emb"].shape == (3, 16)


def test_spinconv_sorts_edges_by_target_then_source():
    edge_index = torch.tensor([[2, 0, 1, 0], [1, 2, 1, 1]])
    edge_distance = torch.tensor([2.0, 3.0, 1.0, 4.0])
    edge_vec = torch.arange(12, dtype=torch.float).view(4, 3)

    sorted_index, sorted_distance, sorted_vec = _sort_edges_by_target(
        edge_index, edge_distance, edge_vec
    )

    assert sorted_index.tolist() == [[0, 1, 2, 0], [1, 1, 1, 2]]
    assert sorted_distance.tolist() == [4.0, 1.0, 2.0, 3.0]
    assert sorted_vec.tolist() == edge_vec[[3, 2, 0, 1]].tolist()


def test_spinconv_projection_raises_before_cuda_index_assert():
    projector = ProjectLatLongSphere(sphere_size_lat=2, sphere_size_long=2)
    x = torch.randn(2, 8)
    index = torch.tensor([[0, 8], [1, 2], [2, 3], [3, 4]])
    delta = torch.ones(4, 2)
    source_edge_index = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="out of bounds"):
        projector(x, 2, index, delta, source_edge_index)


def test_spinconv_element_table_size_supports_real_atomic_numbers():
    assert _element_table_size(90) >= 91
    assert _element_table_size(90) >= 119
