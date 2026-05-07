"""Materials-oriented feature extraction for edge explanations."""

from __future__ import annotations

import math
from typing import Any

import torch

try:
    from ase.data import chemical_symbols
except Exception:  # pragma: no cover
    chemical_symbols = None


def element_symbol(atomic_number: int | float | None) -> str:
    if atomic_number is None:
        return ""
    z = int(atomic_number)
    if chemical_symbols is not None and 0 <= z < len(chemical_symbols):
        return chemical_symbols[z]
    return f"Z{z}"


def edge_distances(data) -> list[float | None]:
    if hasattr(data, "distances"):
        return [float(x) for x in data.distances.detach().cpu().view(-1).tolist()]
    if not hasattr(data, "pos") or not hasattr(data, "edge_index"):
        return [None] * int(data.edge_index.shape[1])
    src, dst = data.edge_index
    pos = data.pos
    offsets = None
    if hasattr(data, "cell_offsets") and hasattr(data, "cell"):
        cell = data.cell[0] if data.cell.ndim == 3 else data.cell
        offsets = data.cell_offsets.to(pos.device).float().matmul(cell.float())
    vectors = pos[dst] - pos[src]
    if offsets is not None:
        vectors = vectors + offsets
    return [float(x) for x in vectors.norm(dim=-1).detach().cpu().tolist()]


def cell_volume(data) -> float | None:
    if not hasattr(data, "cell"):
        return None
    cell = data.cell[0] if data.cell.ndim == 3 else data.cell
    volume = torch.det(cell.float()).abs().item()
    if not math.isfinite(volume) or volume <= 0:
        return None
    return float(volume)


def edge_metadata(data) -> list[dict[str, Any]]:
    src, dst = data.edge_index.detach().cpu()
    atomic_numbers = getattr(data, "atomic_numbers", None)
    if atomic_numbers is not None:
        atomic_numbers = atomic_numbers.detach().cpu().view(-1).tolist()
    distances = edge_distances(data)
    cell_offsets = getattr(data, "cell_offsets", None)
    if cell_offsets is not None:
        cell_offsets = cell_offsets.detach().cpu().tolist()

    rows = []
    for edge_id, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
        src_z = int(atomic_numbers[s]) if atomic_numbers is not None else None
        dst_z = int(atomic_numbers[d]) if atomic_numbers is not None else None
        src_element = element_symbol(src_z)
        dst_element = element_symbol(dst_z)
        bond_pair = sorted([src_element, dst_element])
        rows.append(
            {
                "edge_id": edge_id,
                "src": int(s),
                "dst": int(d),
                "src_Z": src_z,
                "dst_Z": dst_z,
                "src_element": src_element,
                "dst_element": dst_element,
                "bond_type": "-".join(bond_pair),
                "distance": distances[edge_id],
                "cell_offset": (
                    cell_offsets[edge_id] if cell_offsets is not None else None
                ),
            }
        )
    return rows
