"""Generic materials property model with decoupled backbone/pooling/head blocks."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import (
    AttentionalAggregation,
    MessagePassing,
    global_add_pool,
    global_mean_pool,
    radius_graph,
)
from torch_geometric.nn.models.schnet import GaussianSmearing

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import get_pbc_distances, radius_graph_pbc
from ocpmodels.datasets.embeddings import KHOT_EMBEDDINGS, QMOF_KHOT_EMBEDDINGS
from ocpmodels.models.base import BaseModel
from ocpmodels.models.gemnet.gemnet import GemNetT


def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 2,
    activation: str = "silu",
    dropout: float = 0.0,
) -> nn.Sequential:
    activation_cls = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "softplus": nn.Softplus,
        "silu": nn.SiLU,
    }.get(activation, nn.SiLU)
    layers: List[nn.Module] = []
    if num_layers <= 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    current_dim = in_dim
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers)


class PoolingWrapper(nn.Module):
    def __init__(self, name: str, hidden_dim: int):
        super().__init__()
        self.name = (name or "mean").lower()
        if self.name == "attention":
            self.attn = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
            )
        elif self.name not in {"mean", "add", "sum"}:
            raise ValueError(f"Unsupported pooling '{name}'")

    def forward(self, node_emb: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.name == "mean":
            return global_mean_pool(node_emb, batch)
        if self.name in {"add", "sum"}:
            return global_add_pool(node_emb, batch)
        return self.attn(node_emb, batch)


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 1):
        super().__init__()
        self.network = _build_mlp(in_dim, hidden_dim, out_dim, num_layers=2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.network = _build_mlp(in_dim, hidden_dim, out_dim, num_layers=2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class CGCNNConv(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__(aggr="add")
        self.lin1 = nn.Linear(2 * node_dim + edge_dim, 2 * node_dim)
        self.bn1 = nn.BatchNorm1d(2 * node_dim)
        self.ln1 = nn.LayerNorm(node_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        self.bn1.reset_parameters()
        self.ln1.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            size=(x.size(0), x.size(0)),
        )
        return nn.Softplus()(self.ln1(out) + x)

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, x_j, edge_attr], dim=1)
        z = self.lin1(z)
        z = self.bn1(z)
        z1, z2 = z.chunk(2, dim=1)
        return torch.sigmoid(z1) * nn.Softplus()(z2)


class CGCNNBackbone(nn.Module):
    def __init__(
        self,
        bond_feat_dim: int,
        hidden_dim: int = 128,
        num_graph_conv_layers: int = 6,
        embeddings: str = "khot",
        use_pbc: bool = True,
        cutoff: float = 6.0,
        num_gaussians: int = 50,
        otf_graph: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = 50

        if embeddings == "khot":
            embeddings_src = KHOT_EMBEDDINGS
        elif embeddings == "qmof":
            embeddings_src = QMOF_KHOT_EMBEDDINGS
        else:
            raise ValueError("CGCNN embeddings must be 'khot' or 'qmof'")

        self.embedding = torch.zeros(100, len(embeddings_src[1]))
        for i in range(100):
            self.embedding[i] = torch.tensor(embeddings_src[i + 1])
        self.embedding_fc = nn.Linear(len(embeddings_src[1]), hidden_dim)
        self.convs = nn.ModuleList(
            [CGCNNConv(hidden_dim, bond_feat_dim) for _ in range(num_graph_conv_layers)]
        )
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

    @property
    def blocks(self):
        return self.convs

    def forward(self, data):
        if self.embedding.device != data.atomic_numbers.device:
            self.embedding = self.embedding.to(data.atomic_numbers.device)
        data.x = self.embedding[data.atomic_numbers.long() - 1]
        pos = data.pos
        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, self.max_neighbors
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors
        if self.use_pbc:
            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
            )
            data.edge_index = out["edge_index"]
            distances = out["distances"]
        else:
            data.edge_index = radius_graph(pos, r=self.cutoff, batch=data.batch)
            row, col = data.edge_index
            distances = (pos[row] - pos[col]).norm(dim=-1)
        data.edge_attr = self.distance_expansion(distances)
        node_emb = self.embedding_fc(data.x)
        for conv in self.convs:
            node_emb = conv(node_emb, data.edge_index, data.edge_attr)
        return {"node_emb": node_emb}


class SchNetBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        use_pbc: bool = True,
        otf_graph: bool = False,
    ):
        super().__init__()
        from torch_geometric.nn import SchNet

        self.hidden_dim = hidden_dim
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = 50
        schnet = SchNet(
            hidden_channels=hidden_dim,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout="add",
        )
        self.embedding = schnet.embedding
        self.interactions = schnet.interactions
        self.distance_expansion = schnet.distance_expansion
        self.lin1 = schnet.lin1
        self.act = schnet.act
        self.lin2 = schnet.lin2

    @property
    def blocks(self):
        return self.interactions

    def forward(self, data):
        z = data.atomic_numbers.long()
        pos = data.pos
        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, self.max_neighbors
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors
        if self.use_pbc:
            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
            )
            edge_index = out["edge_index"]
            edge_weight = out["distances"]
            edge_attr = self.distance_expansion(edge_weight)
        else:
            edge_index = radius_graph(pos, r=self.cutoff, batch=data.batch)
            row, col = edge_index
            edge_weight = (pos[row] - pos[col]).norm(dim=-1)
            edge_attr = self.distance_expansion(edge_weight)
        node_emb = self.embedding(z)
        for interaction in self.interactions:
            node_emb = node_emb + interaction(node_emb, edge_index, edge_weight, edge_attr)
        node_emb = self.lin1(node_emb)
        node_emb = self.act(node_emb)
        node_emb = self.lin2(node_emb)
        return {"node_emb": node_emb}


class GemNetBackbone(nn.Module):
    def __init__(self, bond_feat_dim: int, **backbone_config):
        del bond_feat_dim
        config = dict(backbone_config)
        config.pop("name", None)
        config.setdefault("regress_forces", False)
        config.setdefault("direct_forces", False)
        config["build_output_blocks"] = False
        self.hidden_dim = config["emb_size_atom"]
        super().__init__()
        self.model = GemNetT(
            num_atoms=None,
            bond_feat_dim=0,
            num_targets=1,
            **config,
        )

    @property
    def blocks(self):
        return self.model.int_blocks

    def forward(self, data):
        return self.model.forward_features(data)


def build_backbone(backbone_config: Dict, bond_feat_dim: int) -> nn.Module:
    name = (backbone_config.get("name") or "cgcnn").lower()
    if name == "cgcnn":
        return CGCNNBackbone(
            bond_feat_dim=bond_feat_dim,
            hidden_dim=backbone_config.get("hidden_dim", 128),
            num_graph_conv_layers=backbone_config.get("num_graph_conv_layers", 6),
            embeddings=backbone_config.get("embeddings", "khot"),
            use_pbc=backbone_config.get("use_pbc", True),
            cutoff=backbone_config.get("cutoff", 6.0),
            num_gaussians=backbone_config.get("num_gaussians", 50),
            otf_graph=backbone_config.get("otf_graph", False),
        )
    if name == "schnet":
        return SchNetBackbone(
            hidden_dim=backbone_config.get("hidden_dim", 128),
            num_filters=backbone_config.get("num_filters", 128),
            num_interactions=backbone_config.get("num_interactions", 6),
            num_gaussians=backbone_config.get("num_gaussians", 50),
            cutoff=backbone_config.get("cutoff", 10.0),
            use_pbc=backbone_config.get("use_pbc", True),
            otf_graph=backbone_config.get("otf_graph", False),
        )
    if name in {"gemnet", "gemnet_t"}:
        return GemNetBackbone(bond_feat_dim=bond_feat_dim, **backbone_config)
    raise ValueError(f"Unsupported property backbone '{name}'")


@registry.register_model("property_model")
class PropertyModel(BaseModel):
    def __init__(
        self,
        num_atoms: Optional[int],
        bond_feat_dim: int,
        num_targets: int,
        backbone: Optional[Dict] = None,
        pooling: Optional[Dict] = None,
        latent: Optional[Dict] = None,
        tasks: Optional[Dict] = None,
        return_latent_by_default: bool = False,
    ):
        super().__init__(num_atoms, bond_feat_dim, num_targets)
        backbone = backbone or {"name": "cgcnn"}
        latent = latent or {}
        tasks = tasks or OrderedDict({"target": {"type": "regression", "out_dim": 1}})
        if not isinstance(tasks, OrderedDict):
            tasks = OrderedDict(tasks.items())

        self.task_specs = tasks
        self.return_latent_by_default = return_latent_by_default
        self.backbone = build_backbone(backbone, bond_feat_dim)
        hidden_dim = getattr(self.backbone, "hidden_dim")
        pooling_name = pooling["name"] if isinstance(pooling, dict) else pooling
        self.pooling = PoolingWrapper(pooling_name or "mean", hidden_dim)
        latent_hidden = latent.get("hidden_dim", hidden_dim)
        latent_dim = latent.get("out_dim", hidden_dim)
        self.latent_projector = _build_mlp(
            hidden_dim,
            latent_hidden,
            latent_dim,
            num_layers=latent.get("num_layers", 2),
            activation=latent.get("activation", "silu"),
            dropout=latent.get("dropout", 0.0),
        )

        self.heads = nn.ModuleDict()
        for task_name, spec in self.task_specs.items():
            task_type = spec.get("type", "regression")
            if task_type == "classification":
                out_dim = int(spec.get("num_classes", spec.get("out_dim", 2)))
                self.heads[task_name] = ClassificationHead(
                    latent_dim,
                    spec.get("hidden_dim", latent_hidden),
                    out_dim,
                )
            else:
                out_dim = int(spec.get("out_dim", 1))
                self.heads[task_name] = RegressionHead(
                    latent_dim,
                    spec.get("hidden_dim", latent_hidden),
                    out_dim,
                )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_first_k_blocks(self, k: int):
        for block in list(self.backbone.blocks)[:k]:
            for param in block.parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def get_param_groups(self):
        return {
            "backbone": list(self.backbone.parameters()),
            "latent": list(self.latent_projector.parameters()),
            "heads": list(self.heads.parameters()),
        }

    def forward(self, data, return_latent: bool = False):
        features = self.backbone(data)
        node_emb = features["node_emb"]
        graph_emb = self.pooling(node_emb, data.batch)
        z = self.latent_projector(graph_emb)

        preds = OrderedDict()
        for task_name, head in self.heads.items():
            pred = head(z)
            if pred.shape[-1] == 1:
                pred = pred.view(-1)
            preds[task_name] = pred

        pred_output = next(iter(preds.values())) if len(preds) == 1 else preds
        output = {"pred": pred_output, "task_preds": preds}
        if return_latent or self.return_latent_by_default:
            output["z"] = z
            output["node_emb"] = node_emb
            output["graph_emb"] = graph_emb
        return output
