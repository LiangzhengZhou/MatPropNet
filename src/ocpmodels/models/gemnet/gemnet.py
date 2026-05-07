"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional

import numpy as np
import torch
from ocpmodels.models.gemnet.layers.me_module import MEModule
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from torch_sparse import SparseTensor

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)

from .layers.atom_update_block import OutputBlock
from .layers.base_layers import Dense
from .layers.efficient import EfficientInteractionDownProjection
from .layers.embedding_block import AtomEmbedding, EdgeEmbedding, AtomGroupEmbedding
from .layers.interaction_block import InteractionBlockTripletsOnly
from .layers.radial_basis import RadialBasis
from .layers.scaling import AutomaticFit
from .layers.spherical_basis import CircularBasisLayer
from .utils import (
    inner_product_normalized,
    mask_neighbors,
    ragged_range,
    repeat_blocks,
)


@registry.register_model("gemnet_t")
class GemNetT(torch.nn.Module):
    """
    GemNet-T, triplets-only variant of GemNet

    Parameters
    ----------
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets: int
            Number of prediction targets.

        num_spherical: int
            Controls maximum frequency.
        num_radial: int
            Controls maximum frequency.
        num_blocks: int
            Number of building blocks to be stacked.

        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.

        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.

        regress_forces: bool
            Whether to predict forces. Default: True
        direct_forces: bool
            If True predict forces based on aggregation of interatomic directions.
            If False predict forces based on negative gradient of energy potential.

        cutoff: float
            Embedding cutoff for interactomic directions in Angstrom.
        rbf: dict
            Name and hyperparameters of the radial basis function.
        envelope: dict
            Name and hyperparameters of the envelope function.
        cbf: dict
            Name and hyperparameters of the cosine basis function.
        extensive: bool
            Whether the output should be extensive (proportional to the number of atoms)
        output_init: str
            Initialization method for the final dense layer.
        activation: str
            Name of the activation function.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        num_atoms: Optional[int],
        bond_feat_dim: int,
        num_targets: int,
        num_spherical: int,
        num_radial: int,
        num_blocks: int,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_trip: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_bil_trip: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        regress_forces: bool = True,
        direct_forces: bool = False,
        cutoff: float = 6.0,
        max_neighbors: int = 50,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        extensive: bool = True,
        otf_graph: bool = False,
        use_pbc: bool = True,
        output_init: str = "HeOrthogonal",
        activation: str = "swish",
        scale_file: Optional[str] = None,
        build_output_blocks: bool = True,
    ):
        super().__init__()
        self.num_targets = num_targets
        assert num_blocks > 0
        self.num_blocks = num_blocks
        self.extensive = extensive
        self.emb_size_atom = emb_size_atom
        self.build_output_blocks = build_output_blocks

        self.cutoff = cutoff
        if self.cutoff <= 0:
            raise ValueError("GemNetT requires cutoff to be a positive number.")

        self.max_neighbors = max_neighbors
        if self.max_neighbors < 1:
            raise ValueError(
                "GemNetT requires max_neighbors to be a positive integer."
            )

        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc

        #elf.group_pairs = torch.tensor([[1, 1], [2, 8], [3, 1], [4, 2], [5, 3], [6, 4], [7, 5], [8, 6], [9, 7], [10, 8], [11, 1], [12, 2], [13, 3], [14, 4], [15, 5], [16, 6], [17, 7], [18, 8], [19, 1], [20, 2], [21, 9], [22, 9], [23, 9], [24, 9], [25, 9], [26, 9], [27, 9], [28, 9], [29, 9], [30, 9], [31, 3], [32, 4], [33, 5], [34, 6], [35, 7], [36, 8], [37, 1], [38, 2], [39, 9], [40, 9], [41, 9], [42, 9], [43, 9], [44, 9], [45, 9], [46, 9], [47, 9], [48, 9], [49, 3], [50, 4], [51, 5], [52, 6], [53, 7], [54, 8], [55, 1], [56, 2], [57, 9], [58, 9], [59, 9], [60, 9], [61, 9], [62, 9], [63, 9], [64, 9], [65, 9], [66, 9], [67, 9], [68, 9], [69, 9], [70, 9], [71, 9], [72, 9], [73, 9], [74, 9], [75, 9], [76, 9], [77, 9], [78, 9], [79, 9], [80, 9], [81, 3], [82, 4], [83, 5], [84, 6], [85, 7], [86, 8], [87, 1], [88, 2], [89, 9], [90, 9], [91, 9], [92, 9], [93, 9], [94, 9], [95, 9], [96, 9], [97, 9], [98, 9], [99, 9], [100, 9], [101, 9], [102, 9], [103, 9], [104, 9], [105, 9], [106, 9], [107, 9], [108, 9], [109, 9], [110, 9], [111, 9], [112, 9], [113, 3], [114, 4], [115, 5], [116, 6], [117, 7], [118, 8]], device='cuda')

        emb_size_attention = 128 #$$$
        num_modules = 3
        AutomaticFit.reset()  # make sure that queue is empty (avoid potential error)

        # GemNet variants
        self.direct_forces = direct_forces

        ### ---------------------------------- Basis Functions ---------------------------------- ###
        self.radial_basis_attn = RadialBasis(
            num_radial=num_radial*2,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        ) #$$$

        radial_basis_cbf3 = RadialBasis(
            num_radial=num_radial*2,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )
        self.cbf_basis3 = CircularBasisLayer(
            num_spherical,
            radial_basis=radial_basis_cbf3,
            cbf=cbf,
            efficient=True,
            emb_size_attention=emb_size_attention,
            num_radial=num_radial,
            num_modules=num_modules
        )
        ### ------------------------------------------------------------------------------------- ###
        self.me_block = MEModule(num_modules = num_modules, emb_size_attention = emb_size_attention, num_radial = num_radial) #$$$
       
        ### ------------------------------- Share Down Projections ------------------------------ ###
        # Share down projection across all interaction blocks
        self.mlp_rbf3 = Dense(
            num_radial, ###$$$$
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf3 = EfficientInteractionDownProjection(
            num_spherical, num_radial, emb_size_cbf ###$$$$
        )

        # Share the dense Layer of the atom embedding block accross the interaction blocks
        self.mlp_rbf_h = Dense(
            num_radial, ###$$$$
            emb_size_rbf, ###$$$$
            activation=None,
            bias=False,
        )
        self.mlp_rbf_out = Dense(
            num_radial, ###$$$$
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        ### ------------------------------------------------------------------------------------- ###
        # Embedding block
        self.atom_emb = AtomEmbedding(emb_size_atom)
        self.atom_emb_attention = AtomEmbedding(emb_size_attention) #$$$
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation #$$$
        )

        int_blocks = []

        # Interaction Blocks
        interaction_block = InteractionBlockTripletsOnly  # GemNet-(d)T
        for i in range(num_blocks):
            int_blocks.append(
                interaction_block(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip=emb_size_trip,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_bil_trip=emb_size_bil_trip,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    activation=activation,
                    scale_file=scale_file,
                    name=f"IntBlock_{i+1}",
                )
            )

        if self.build_output_blocks:
            out_blocks = []
            for i in range(num_blocks + 1):
                out_blocks.append(
                    OutputBlock(
                        emb_size_atom=emb_size_atom,
                        emb_size_edge=emb_size_edge,
                        emb_size_rbf=emb_size_rbf,
                        nHidden=num_atom,
                        num_targets=num_targets,
                        activation=activation,
                        output_init=output_init,
                        direct_forces=direct_forces,
                        scale_file=scale_file,
                        name=f"OutBlock_{i}",
                    )
                )
            self.out_blocks = torch.nn.ModuleList(out_blocks)
        else:
            self.out_blocks = None
        self.int_blocks = torch.nn.ModuleList(int_blocks)

        self.shared_parameters = [
            (self.mlp_rbf3.linear.weight, self.num_blocks),
            (self.mlp_cbf3.weight, self.num_blocks),
            (self.mlp_rbf_h.linear.weight, self.num_blocks),
        ]
        if self.build_output_blocks:
            self.shared_parameters.append(
                (self.mlp_rbf_out.linear.weight, self.num_blocks + 1)
            )

    def get_triplets(self, edge_index, num_atoms):
        """
        Get all b->a for each edge c->a.
        It is possible that b=c, as long as the edges are distinct.

        Returns
        -------
        id3_ba: torch.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        id3_ca: torch.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        id3_ragged_idx: torch.Tensor, shape (num_triplets,)
            Indices enumerating the copies of id3_ca for creating a padded matrix
        """
        idx_s, idx_t = edge_index  # c->a (source=c, target=a)

        value = torch.arange(
            idx_s.size(0), device=idx_s.device, dtype=idx_s.dtype
        )
        # Possibly contains multiple copies of the same edge (for periodic interactions)
        adj = SparseTensor(
            row=idx_t,
            col=idx_s,
            value=value,
            sparse_sizes=(num_atoms, num_atoms),
        )
        adj_edges = adj[idx_t]

        # Edge indices (b->a, c->a) for triplets.
        id3_ba = adj_edges.storage.value()
        id3_ca = adj_edges.storage.row()

        # Remove self-loop triplets
        # Compare edge indices, not atom indices to correctly handle periodic interactions
        mask = id3_ba != id3_ca
        id3_ba = id3_ba[mask]
        id3_ca = id3_ca[mask]

        # Get indices to reshape the neighbor indices b->a into a dense matrix.
        # id3_ca has to be sorted for this to work.
        num_triplets = torch.bincount(id3_ca, minlength=idx_s.size(0))
        id3_ragged_idx = ragged_range(num_triplets)

        return id3_ba, id3_ca, id3_ragged_idx

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(
        self,
        edge_index,
        cell_offsets,
        neighbors,
        edge_dist,
        edge_vector,
        edge_mask=None,
    ):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(
            batch_edge, minlength=neighbors.size(0)
        )

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_dist_new = self.select_symmetric_edges(
            edge_dist, mask, edge_reorder_idx, False
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )
        edge_mask_new = None
        if edge_mask is not None:
            edge_mask_new = self.select_symmetric_edges(
                edge_mask, mask, edge_reorder_idx, False
            )

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_dist_new,
            edge_vector_new,
            edge_mask_new,
        )

    def select_edges(
        self,
        data,
        edge_index,
        cell_offsets,
        neighbors,
        edge_dist,
        edge_vector,
        cutoff=None,
    ):
        if cutoff is not None:
            edge_mask = edge_dist <= cutoff

            edge_index = edge_index[:, edge_mask]
            cell_offsets = cell_offsets[edge_mask]
            neighbors = mask_neighbors(neighbors, edge_mask)
            edge_dist = edge_dist[edge_mask]
            edge_vector = edge_vector[edge_mask]

        empty_image = neighbors == 0
        if torch.any(empty_image):
            raise ValueError(
                f"An image has no neighbors: id={data.id[empty_image]}, "
                f"sid={data.sid[empty_image]}, fid={data.fid[empty_image]}"
            )
        return edge_index, cell_offsets, neighbors, edge_dist, edge_vector

    def _initial_edge_mask_for_data(self, data, edge_mask, edge_index):
        if edge_mask is None:
            return None
        edge_mask = edge_mask.to(device=edge_index.device).view(-1)
        if edge_mask.numel() != edge_index.shape[1]:
            raise ValueError(
                f"edge_mask has {edge_mask.numel()} entries, "
                f"expected {edge_index.shape[1]}."
            )
        return edge_mask

    def generate_interaction_graph(self, data, edge_mask=None):
        num_atoms = data.atomic_numbers.size(0)

        if self.use_pbc:
            if self.otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data, self.cutoff, self.max_neighbors
                )
                edge_mask = None
            else:
                edge_index = data.edge_index
                cell_offsets = data.cell_offsets
                neighbors = data.neighbors
                edge_mask = self._initial_edge_mask_for_data(
                    data, edge_mask, edge_index
                )

            # Switch the indices, so the second one becomes the target index,
            # over which we can efficiently aggregate.
            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            D_st = out["distances"]
            # These vectors actually point in the opposite direction.
            # But we want to use col as idx_t for efficient aggregation.
            V_st = -out["distance_vec"] / D_st[:, None]
            # offsets_ca = -out["offsets"]  # a - c + offset
        else:
            self.otf_graph = True
            edge_index = radius_graph(
                data.pos,
                r=self.cutoff,
                batch=data.batch,
                max_num_neighbors=self.max_neighbors,
            )
            j, i = edge_index
            distance_vec = data.pos[j] - data.pos[i]

            D_st = distance_vec.norm(dim=-1)
            V_st = -distance_vec / D_st[:, None]
            cell_offsets = torch.zeros(
                edge_index.shape[1], 3, device=data.pos.device
            )
            neighbors = compute_neighbors(data, edge_index)
            edge_mask = None

        # Mask interaction edges if required
        if self.otf_graph or np.isclose(self.cutoff, 6):
            select_cutoff = None
        else:
            select_cutoff = self.cutoff
        if edge_mask is not None and select_cutoff is not None:
            edge_mask = edge_mask[D_st <= select_cutoff]
        (edge_index, cell_offsets, neighbors, D_st, V_st,) = self.select_edges(
            data=data,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            neighbors=neighbors,
            edge_dist=D_st,
            edge_vector=V_st,
            cutoff=select_cutoff,
        )

        (
            edge_index,
            cell_offsets,
            neighbors,
            D_st,
            V_st,
            edge_mask,
        ) = self.reorder_symmetric_edges(
            edge_index, cell_offsets, neighbors, D_st, V_st, edge_mask=edge_mask
        )

        # Indices for swapping c->a and a->c (for symmetric MP)
        block_sizes = neighbors // 2
        id_swap = repeat_blocks(
            block_sizes,
            repeats=2,
            continuous_indexing=False,
            start_idx=block_sizes[0],
            block_inc=block_sizes[:-1] + block_sizes[1:],
            repeat_inc=-block_sizes,
        )

        id3_ba, id3_ca, id3_ragged_idx = self.get_triplets(
            edge_index, num_atoms=num_atoms
        )

        return (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
            edge_mask,
        )

    @property
    def blocks(self):
        return self.int_blocks

    @conditional_grad(torch.enable_grad())
    def _compute_interaction_states(self, data, edge_mask=None):
        pos = data.pos
        batch = data.batch
        atomic_numbers = data.atomic_numbers.long()

        if self.regress_forces and not self.direct_forces:
            pos.requires_grad_(True)

        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        ) = self.generate_interaction_graph(data, edge_mask=edge_mask)
        idx_s, idx_t = edge_index
        if edge_mask is not None:
            edge_mask = edge_mask.to(device=D_st.device).view(-1)
            if edge_mask.numel() != D_st.shape[0]:
                raise ValueError(
                    f"edge_mask has {edge_mask.numel()} entries, "
                    f"expected {D_st.shape[0]} after GemNet graph generation."
                )

        cos_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        h_atomic_data = self.atom_emb_attention(atomic_numbers)
        rad_cbf3, cbf3 = self.cbf_basis3(
            D_st, cos_cab, id3_ca, h_atomic_data, idx_s, idx_t
        )
        rbf_attn = self.radial_basis_attn(D_st)
        if edge_mask is not None:
            rbf_attn = rbf_attn * edge_mask.view(-1, 1)
        me_block = self.me_block(rbf_attn, h_atomic_data, idx_s, idx_t)

        h = self.atom_emb(atomic_numbers)
        m = self.edge_emb(h, me_block, idx_s, idx_t)
        if edge_mask is not None:
            m = m * edge_mask.view(-1, 1)

        rbf3 = self.mlp_rbf3(me_block)
        if edge_mask is not None:
            rbf3 = rbf3 * edge_mask.view(-1, 1)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)
        rbf_h = self.mlp_rbf_h(me_block)
        if edge_mask is not None:
            rbf_h = rbf_h * edge_mask.view(-1, 1)
        rbf_out = self.mlp_rbf_out(me_block)
        if edge_mask is not None:
            rbf_out = rbf_out * edge_mask.view(-1, 1)

        states = [(h, m)]
        for i in range(self.num_blocks):
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf3=rbf3,
                cbf3=cbf3,
                id3_ragged_idx=id3_ragged_idx,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
                edge_mask=edge_mask,
            )
            states.append((h, m))

        return {
            "states": states,
            "rbf_out": rbf_out,
            "idx_t": idx_t,
            "batch": batch,
            "pos": pos,
            "edge_vector": V_st,
            "edge_mask": edge_mask,
        }

    @conditional_grad(torch.enable_grad())
    def forward_features(self, data, edge_mask=None, node_mask=None, explain_mode=False):
        del node_mask, explain_mode
        features = self._compute_interaction_states(data, edge_mask=edge_mask)
        node_emb, edge_emb = features["states"][-1]
        return {
            "node_emb": node_emb,
            "edge_emb": edge_emb,
            "rbf_out": features["rbf_out"],
            "idx_t": features["idx_t"],
            "batch": features["batch"],
            "pos": features["pos"],
            "edge_vector": features["edge_vector"],
        }

    def _predict_from_interaction_states(self, data, features):
        rbf_out = features["rbf_out"]
        idx_t = features["idx_t"]
        batch = features["batch"]
        pos = features["pos"]
        edge_vector = features["edge_vector"]

        atom_predictions = []
        edge_predictions = []
        for i, (h, m) in enumerate(features["states"]):
            energy_block, force_block = self.out_blocks[i](h, m, rbf_out, idx_t)
            atom_predictions.append(energy_block)
            edge_predictions.append(force_block)

        energy_per_atom = torch.stack(atom_predictions, dim=0).sum(dim=0)
        edge_forces = torch.stack(edge_predictions, dim=0).sum(dim=0)

        num_molecules = int(batch.max().item()) + 1
        reduce_mode = "add" if self.extensive else "mean"
        energy = scatter(
            energy_per_atom,
            batch,
            dim=0,
            dim_size=num_molecules,
            reduce=reduce_mode,
        )

        if not self.regress_forces:
            return energy

        if self.direct_forces:
            edge_force_vectors = edge_forces[:, :, None] * edge_vector[:, None, :]
            forces = scatter(
                edge_force_vectors,
                idx_t,
                dim=0,
                dim_size=data.atomic_numbers.size(0),
                reduce="add",
            ).squeeze(1)
            return energy, forces

        if self.num_targets > 1:
            forces = []
            for i in range(self.num_targets):
                forces.append(
                    -torch.autograd.grad(
                        energy[:, i].sum(), pos, create_graph=True
                    )[0]
                )
            return energy, torch.stack(forces, dim=1)

        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=True
        )[0]
        return energy, forces

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        if not self.build_output_blocks or self.out_blocks is None:
            raise RuntimeError(
                "GemNetT was initialized without output blocks. "
                "Use forward_features() when using it as a backbone."
            )

        features = self._compute_interaction_states(data)
        return self._predict_from_interaction_states(data, features)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
