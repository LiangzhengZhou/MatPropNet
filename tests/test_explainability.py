from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data

from matpropnet.cli.explain import main as explain_cli_main
from matpropnet.cli.ensemble_explain import main as ensemble_explain_cli_main
from matpropnet.explain.algorithms import MatPropNetEdgeMaskExplainer
from matpropnet.explain.outputs import (
    aggregate_by_bond_type,
    build_edge_table,
    save_sample_outputs,
)
from matpropnet.explain.wrappers import GraphRegressionModelWrapper
from ocpmodels.models.dimenet_plus_plus import InteractionPPBlock, OutputPPBlock
from ocpmodels.models.gemnet.layers.interaction_block import TripletInteraction


class MaskAwareToyModel(nn.Module):
    def forward(self, data, edge_mask=None):
        edge_score = data.edge_attr.view(-1)
        if edge_mask is not None:
            edge_score = edge_score * edge_mask
        return edge_score.sum().view(1)


def _toy_data():
    return Data(
        atomic_numbers=torch.tensor([5, 5, 74]),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [1.7, 0.0, 0.0], [0.0, 2.2, 0.0]],
            dtype=torch.float,
        ),
        cell=torch.eye(3).view(1, 3, 3) * 5.0,
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_attr=torch.tensor([[1.0], [0.5], [0.1]], dtype=torch.float),
        cell_offsets=torch.zeros(3, 3, dtype=torch.long),
        batch=torch.zeros(3, dtype=torch.long),
        y=torch.tensor([1.0]),
    )


def test_matpropnet_edge_mask_explainer_returns_edge_mask():
    data = _toy_data()
    explainer = MatPropNetEdgeMaskExplainer(epochs=3, lr=0.05)

    result = explainer.explain_data(MaskAwareToyModel(), data)

    assert result.edge_mask.shape == (data.edge_index.shape[1],)
    assert torch.all(result.edge_mask >= 0)
    assert torch.all(result.edge_mask <= 1)


def test_matpropnet_explainer_can_be_used_with_pyg_explainer():
    from torch_geometric.explain import Explainer

    data = _toy_data()
    wrapped = GraphRegressionModelWrapper(MaskAwareToyModel())
    explainer = Explainer(
        model=wrapped,
        algorithm=MatPropNetEdgeMaskExplainer(epochs=2, lr=0.05),
        explanation_type="model",
        model_config={
            "mode": "regression",
            "task_level": "graph",
            "return_type": "raw",
        },
        node_mask_type=None,
        edge_mask_type="object",
    )

    explanation = explainer(data.x, data.edge_index, data=data)

    assert explanation.edge_mask.shape == (data.edge_index.shape[1],)


def test_edge_table_and_bond_type_aggregation():
    data = _toy_data()
    mask = torch.tensor([0.9, 0.2, 0.7])

    rows = build_edge_table(data, mask, sample_id="s1", top_k=2)
    grouped = aggregate_by_bond_type(rows)

    assert rows[0]["bond_type"] == "B-B"
    assert rows[0]["is_topk"] is True
    by_type = {row["bond_type"]: row for row in grouped}
    assert by_type["B-B"]["count"] == 1
    assert by_type["B-W"]["count"] == 2
    assert by_type["B-W"]["topk_count"] == 1


def test_dimenetplusplus_blocks_apply_deep_edge_and_triplet_masks():
    torch.manual_seed(7)
    x = torch.randn(5, 8)
    rbf = torch.randn(5, 4)
    sbf = torch.randn(6, 8)
    idx_kj = torch.tensor([0, 1, 2, 3, 1, 4])
    idx_ji = torch.tensor([1, 2, 3, 4, 0, 2])
    edge_mask_ones = torch.ones(5)
    edge_mask_partial = torch.tensor([1.0, 0.0, 0.5, 1.0, 0.25])

    interaction = InteractionPPBlock(
        hidden_channels=8,
        int_emb_size=6,
        basis_emb_size=4,
        num_spherical=2,
        num_radial=4,
        num_before_skip=1,
        num_after_skip=1,
    )
    out_plain = interaction(x, rbf, sbf, idx_kj, idx_ji)
    out_ones = interaction(x, rbf, sbf, idx_kj, idx_ji, edge_mask=edge_mask_ones)
    out_masked = interaction(
        x, rbf, sbf, idx_kj, idx_ji, edge_mask=edge_mask_partial
    )

    assert torch.allclose(out_plain, out_ones, atol=1.0e-6)
    assert not torch.allclose(out_plain, out_masked)

    output = OutputPPBlock(
        num_radial=4,
        hidden_channels=8,
        out_emb_channels=8,
        out_channels=2,
        num_layers=1,
    )
    node_plain = output(x, rbf, torch.tensor([0, 0, 1, 1, 2]), num_nodes=3)
    node_ones = output(
        x,
        rbf,
        torch.tensor([0, 0, 1, 1, 2]),
        num_nodes=3,
        edge_mask=edge_mask_ones,
    )
    assert torch.allclose(node_plain, node_ones, atol=1.0e-6)


def test_gemnet_triplet_interaction_applies_edge_and_triplet_masks():
    torch.manual_seed(11)
    n_edges = 4
    n_triplets = 5
    m = torch.randn(n_edges, 6)
    rbf3 = torch.randn(n_edges, 3)
    id3_ba = torch.tensor([0, 1, 2, 0, 3])
    id3_ca = torch.tensor([1, 2, 3, 3, 0])
    id3_ragged_idx = torch.tensor([0, 0, 0, 1, 0])
    id_swap = torch.tensor([1, 0, 3, 2])
    cbf3 = (
        torch.randn(n_edges, 2, 2),
        torch.randn(n_edges, 2, 2),
    )
    edge_mask_ones = torch.ones(n_edges)
    edge_mask_partial = torch.tensor([1.0, 0.0, 0.5, 0.25])

    interaction = TripletInteraction(
        emb_size_edge=6,
        emb_size_trip=4,
        emb_size_bilinear=5,
        emb_size_rbf=3,
        emb_size_cbf=2,
        activation="silu",
    )
    out_plain = interaction(m, rbf3, cbf3, id3_ragged_idx, id_swap, id3_ba, id3_ca)
    out_ones = interaction(
        m,
        rbf3,
        cbf3,
        id3_ragged_idx,
        id_swap,
        id3_ba,
        id3_ca,
        edge_mask=edge_mask_ones,
    )
    out_masked = interaction(
        m,
        rbf3,
        cbf3,
        id3_ragged_idx,
        id_swap,
        id3_ba,
        id3_ca,
        edge_mask=edge_mask_partial,
    )

    assert torch.allclose(out_plain, out_ones, atol=1.0e-6)
    assert not torch.allclose(out_plain, out_masked)


def test_save_sample_outputs_writes_expected_files(tmp_path):
    data = _toy_data()
    mask = torch.tensor([0.9, 0.2, 0.7])
    save_sample_outputs(
        output_dir=tmp_path,
        data=data,
        edge_mask=mask,
        sample_id="s1",
        top_k=2,
        summary={"sample_id": "s1", "model_prediction": [1.0]},
    )

    assert (tmp_path / "explanation_edges.csv").exists()
    assert (tmp_path / "bond_type_importance.csv").exists()
    assert (tmp_path / "explanation_summary.json").exists()
    assert (tmp_path / "masks.pt").exists()
    with (tmp_path / "explanation_edges.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["sample_id"] == "s1"


def test_explain_cli_dry_run(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text(
        """
trainer: property
dataset:
  src: data.lmdb
model:
  name: property_model
  backbone:
    name: cgcnn
  pooling:
    name: mean
  latent: {}
task:
  tasks:
    H:
      type: regression
optim:
  batch_size: 1
  num_workers: 0
  lr_initial: 0.001
  max_epochs: 1
""",
        encoding="utf-8",
    )

    code = explain_cli_main(
        [
            "--config",
            str(config),
            "--checkpoint",
            "model.pt",
            "--lmdb",
            "data.lmdb",
            "--out-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )

    assert code == 0


def test_ensemble_explain_cli_dry_run(tmp_path):
    base_config = tmp_path / "base.yml"
    base_config.write_text(
        """
trainer: property
dataset:
  src: data.lmdb
model:
  name: property_model
  backbone:
    name: cgcnn
  pooling:
    name: mean
  latent: {}
task:
  tasks:
    H:
      type: regression
optim:
  batch_size: 1
  num_workers: 0
  lr_initial: 0.001
  max_epochs: 1
""",
        encoding="utf-8",
    )
    manifest = tmp_path / "ensemble_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "base_config": str(base_config),
                "members": [{"index": 0, "seed": 1, "checkpoint": "m0.pt"}],
            }
        ),
        encoding="utf-8",
    )

    code = ensemble_explain_cli_main(
        [
            "--manifest",
            str(manifest),
            "--lmdb",
            "data.lmdb",
            "--out-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )

    assert code == 0
