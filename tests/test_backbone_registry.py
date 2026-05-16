from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from ocpmodels.models.property_model import (
    DimeNetPlusPlusBackbone,
    GemNetBackbone,
    SchNetBackbone,
    build_backbone,
)
from ocpmodels.models.gemnet.gemnet import GemNetT
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


@pytest.mark.parametrize("backbone_cls", [GemNetBackbone, DimeNetPlusPlusBackbone])
def test_mask_aware_backbone_wrappers_forward_masks(backbone_cls):
    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.kwargs = None

        def forward_features(self, data, **kwargs):
            self.kwargs = kwargs
            return {"node_emb": torch.ones(1, 4)}

    backbone = backbone_cls.__new__(backbone_cls)
    nn.Module.__init__(backbone)
    backbone.model = Recorder()
    data = Data()
    edge_mask = torch.tensor([1.0])
    node_mask = torch.tensor([[1.0]])

    out = backbone(
        data,
        edge_mask=edge_mask,
        node_mask=node_mask,
        explain_mode=True,
    )

    assert out["node_emb"].shape == (1, 4)
    assert backbone.model.kwargs["edge_mask"] is edge_mask
    assert backbone.model.kwargs["node_mask"] is node_mask
    assert backbone.model.kwargs["explain_mode"] is True


def test_gemnet_interaction_states_accept_returned_edge_mask():
    model = GemNetT.__new__(GemNetT)
    model.regress_forces = False
    model.direct_forces = False
    model.num_blocks = 0
    model.int_blocks = []

    edge_mask = torch.tensor([1.0, 0.5])

    def generate_interaction_graph(data, edge_mask=None):
        return (
            torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            torch.tensor([2]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            torch.tensor([1, 0]),
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
            torch.tensor([0, 0]),
            edge_mask,
        )

    model.generate_interaction_graph = generate_interaction_graph
    model.atom_emb_attention = lambda atomic_numbers: torch.ones(
        atomic_numbers.numel(), 4
    )
    model.cbf_basis3 = lambda *args: (torch.ones(2, 3), torch.ones(2, 3))
    model.radial_basis_attn = lambda distances: torch.ones(distances.numel(), 4)
    model.me_block = lambda rbf_attn, h_atomic_data, idx_s, idx_t: rbf_attn
    model.atom_emb = lambda atomic_numbers: torch.ones(atomic_numbers.numel(), 4)
    model.edge_emb = lambda h, me_block, idx_s, idx_t: torch.ones(idx_s.numel(), 4)
    model.mlp_rbf3 = lambda me_block: torch.ones(me_block.shape[0], 3)
    model.mlp_cbf3 = lambda rad_cbf3, cbf3, id3_ca, id3_ragged_idx: torch.ones(
        2, 3
    )
    model.mlp_rbf_h = lambda me_block: torch.ones(me_block.shape[0], 4)
    model.mlp_rbf_out = lambda me_block: torch.ones(me_block.shape[0], 4)

    data = Data(
        pos=torch.zeros(2, 3),
        batch=torch.zeros(2, dtype=torch.long),
        atomic_numbers=torch.tensor([6, 8], dtype=torch.long),
    )

    features = model._compute_interaction_states(data, edge_mask=edge_mask)

    assert torch.equal(features["edge_mask"], edge_mask)
    assert features["states"][-1][0].shape == (2, 4)


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
