"""Convert property CSV files into LMDB datasets."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import pickle
from pathlib import Path

import ase.io
import lmdb
import numpy as np
import torch
from tqdm import tqdm

from ocpmodels.preprocessing import AtomsToGraphs


NULL_TOKENS = {"", "na", "nan", "none", "null", "missing"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert generic materials property CSV files to LMDB."
    )
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--out-path", help="Output directory for one LMDB split")
    parser.add_argument("--out-root", help="Output root for split/K-fold datasets")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--cif-column", default="cif")
    parser.add_argument("--target-columns", default=None)
    parser.add_argument("--task-types", default=None)
    parser.add_argument("--num-classes", default=None)
    parser.add_argument("--cif-mode", choices=["auto", "inline", "path"], default="auto")
    parser.add_argument("--cif-root", default=None)
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--max-neigh", type=int, default=50)
    parser.add_argument("--get-edges", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--map-size-gb", type=int, default=4)
    parser.add_argument("--split", nargs=3, type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--split-column", default=None)
    parser.add_argument("--kfolds", type=int, default=0)
    parser.add_argument("--fold-val-ratio", type=float, default=0.1)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def load_rows(csv_path: str):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def parse_csv_list(value, cast=str):
    if not value:
        return []
    return [cast(item.strip()) for item in value.split(",")]


def infer_target_columns(fieldnames, id_column, cif_column, split_column):
    exclude = {id_column, cif_column}
    if split_column:
        exclude.add(split_column)
    return [field for field in fieldnames if field not in exclude]


def build_task_schema(args, fieldnames):
    target_columns = parse_csv_list(args.target_columns) or infer_target_columns(
        fieldnames, args.id_column, args.cif_column, args.split_column
    )
    task_types = parse_csv_list(args.task_types) or ["regression"] * len(target_columns)
    if len(task_types) != len(target_columns):
        raise SystemExit("--task-types must align with --target-columns.")
    valid_task_types = {"regression", "classification"}
    invalid_task_types = [
        task_type for task_type in task_types if task_type not in valid_task_types
    ]
    if invalid_task_types:
        raise SystemExit(
            f"Unsupported task type(s): {', '.join(invalid_task_types)}. "
            "Expected regression or classification."
        )
    num_classes = parse_csv_list(args.num_classes, int) if args.num_classes else []
    if num_classes and len(num_classes) != len(target_columns):
        raise SystemExit("--num-classes must align with --target-columns.")
    schema = []
    for idx, column in enumerate(target_columns):
        spec = {"name": column, "type": task_types[idx]}
        if task_types[idx] == "classification":
            spec["num_classes"] = num_classes[idx] if num_classes else 2
        schema.append(spec)
    return schema


def resolve_atoms(cif_value, cif_mode, cif_root):
    cif_value = str(cif_value).strip()
    if cif_mode == "inline" or (
        cif_mode == "auto" and ("\n" in cif_value or cif_value.startswith("data_"))
    ):
        return ase.io.read(io.StringIO(cif_value), format="cif")
    cif_path = Path(cif_value)
    if cif_root is not None:
        cif_path = Path(cif_root) / cif_path
    return ase.io.read(str(cif_path))


def parse_target(raw_value, task_type):
    if raw_value is None:
        return 0.0, 0
    value = str(raw_value).strip()
    if value.lower() in NULL_TOKENS:
        return 0.0, 0
    if task_type == "classification":
        return int(float(value)), 1
    return float(value), 1


def split_by_column(rows, split_column):
    buckets = {"train": [], "val": [], "test": []}
    for row in rows:
        split_name = str(row[split_column]).strip().lower()
        if split_name not in buckets:
            raise ValueError(f"Unsupported split label '{split_name}'")
        buckets[split_name].append(row)
    return buckets


def random_split(rows, fractions, seed):
    if len(fractions) != 3:
        raise ValueError("random_split expects exactly three fractions for train/val/test.")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    frac = np.asarray(fractions, dtype=np.float64)
    if np.any(frac < 0):
        raise ValueError("Split fractions must be non-negative.")
    if frac.sum() <= 0:
        raise ValueError("Split fractions must sum to a positive value.")
    frac = frac / frac.sum()
    n_total = len(rows)
    n_train = int(frac[0] * n_total)
    n_val = int(frac[1] * n_total)
    return {
        "train": [rows[i] for i in indices[:n_train]],
        "val": [rows[i] for i in indices[n_train : n_train + n_val]],
        "test": [rows[i] for i in indices[n_train + n_val :]],
    }


def kfold_indices(num_samples, num_folds, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    fold_sizes = np.full(num_folds, num_samples // num_folds, dtype=int)
    fold_sizes[: num_samples % num_folds] += 1
    folds = []
    current = 0
    for fold_size in fold_sizes:
        folds.append(indices[current : current + fold_size])
        current += fold_size
    return folds


def compute_stats(task_schema, targets_list, masks_list):
    means, stds = [], []
    for task_idx, task in enumerate(task_schema):
        if task["type"] == "classification":
            means.append(0.0)
            stds.append(1.0)
            continue
        values = [
            targets[task_idx]
            for targets, masks in zip(targets_list, masks_list)
            if masks[task_idx] == 1
        ]
        if values:
            values = np.asarray(values, dtype=np.float32)
            means.append(float(values.mean()))
            std = float(values.std())
            stds.append(std if std > 1e-12 else 1.0)
        else:
            means.append(0.0)
            stds.append(1.0)
    return {"target_mean": means, "target_std": stds}


def build_data_object(a2g, row, row_idx, task_schema, args):
    atoms = resolve_atoms(row[args.cif_column], args.cif_mode, args.cif_root)
    data_object = a2g.convert(atoms)

    targets, mask = [], []
    for task in task_schema:
        value, valid = parse_target(row.get(task["name"]), task["type"])
        targets.append(value)
        mask.append(valid)

    data_object.y = torch.tensor(targets, dtype=torch.float)
    data_object.target_mask = torch.tensor(mask, dtype=torch.bool)
    data_object.sid = row_idx
    data_object.sample_id = str(row.get(args.id_column, row_idx))
    data_object.target_names = [task["name"] for task in task_schema]
    edge_index = getattr(data_object, "edge_index", None)
    neighbors = int(edge_index.shape[1]) if edge_index is not None else 0
    return data_object, targets, mask, int(data_object.natoms), neighbors


def write_metadata(out_dir, natoms_list, neighbors_list):
    np.savez(
        os.path.join(out_dir, "metadata.npz"),
        natoms=np.asarray(natoms_list, dtype=np.int64),
        neighbors=np.asarray(neighbors_list, dtype=np.int64),
    )


def write_schema(out_dir, task_schema, stats):
    with open(os.path.join(out_dir, "target_schema.json"), "w", encoding="utf-8") as handle:
        json.dump({"tasks": task_schema, "stats": stats}, handle, indent=2)


def _cleanup_lmdb_path(db_path):
    for maybe_partial in (db_path, f"{db_path}-lock"):
        if os.path.isdir(maybe_partial):
            for entry in os.listdir(maybe_partial):
                os.remove(os.path.join(maybe_partial, entry))
            os.rmdir(maybe_partial)
        elif os.path.exists(maybe_partial):
            os.remove(maybe_partial)


def _open_lmdb_for_writing(db_path, map_size):
    candidate_map_sizes = [map_size]
    reduced_sizes = [512 * 1024**2, 128 * 1024**2]
    for reduced_size in reduced_sizes:
        if reduced_size < map_size:
            candidate_map_sizes.append(reduced_size)

    attempts = []
    for candidate_size in candidate_map_sizes:
        attempts.append(
            {
                "map_size": candidate_size,
                "subdir": False,
                "meminit": False,
                "map_async": True,
            }
        )
        attempts.append(
            {
                "map_size": candidate_size,
                "subdir": False,
                "meminit": False,
                "map_async": False,
            }
        )
    for candidate_size in candidate_map_sizes:
        attempts.append(
            {
                "map_size": candidate_size,
                "subdir": True,
                "meminit": True,
                "map_async": False,
            }
        )

    last_error = None
    for open_kwargs in attempts:
        _cleanup_lmdb_path(db_path)
        if open_kwargs["subdir"]:
            os.makedirs(db_path, exist_ok=True)
        try:
            return lmdb.open(db_path, **open_kwargs)
        except lmdb.Error as exc:
            last_error = exc
            continue
    raise last_error


def write_lmdb_split(rows, out_dir, task_schema, args):
    os.makedirs(out_dir, exist_ok=True)
    a2g = AtomsToGraphs(
        max_neigh=args.max_neigh,
        radius=args.radius,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=args.get_edges,
        r_fixed=True,
    )
    db_path = os.path.join(out_dir, "data.lmdb")
    db = _open_lmdb_for_writing(db_path, args.map_size_gb * 1024**3)

    targets_list, masks_list = [], []
    natoms_list, neighbors_list = [], []
    written = 0
    failed = 0
    failed_samples = []
    with db.begin(write=True) as txn:
        for row_idx, row in enumerate(tqdm(rows, desc=f"Writing {out_dir}")):
            try:
                data_object, targets, mask, natoms, neighbors = build_data_object(
                    a2g, row, row_idx, task_schema, args
                )
            except Exception as exc:
                if not args.skip_failed:
                    db.close()
                    raise
                failed += 1
                failed_samples.append(
                    {
                        "row_idx": row_idx,
                        "sample_id": str(row.get(args.id_column, row_idx)),
                        "error": str(exc),
                    }
                )
                continue
            txn.put(f"{written}".encode("ascii"), pickle.dumps(data_object, protocol=-1))
            targets_list.append(targets)
            masks_list.append(mask)
            natoms_list.append(natoms)
            neighbors_list.append(neighbors)
            written += 1
        txn.put("length".encode("ascii"), pickle.dumps(written, protocol=-1))

    db.sync()
    db.close()
    stats = compute_stats(task_schema, targets_list, masks_list)
    write_metadata(out_dir, natoms_list, neighbors_list)
    write_schema(out_dir, task_schema, stats)
    if failed_samples:
        with open(os.path.join(out_dir, "failed_samples.json"), "w", encoding="utf-8") as handle:
            json.dump(failed_samples, handle, indent=2)
    return {"num_samples": written, "num_failed": failed, **stats}


def write_manifest(manifest_path, split_rows, id_column):
    payload = {
        split_name: [str(row.get(id_column, idx)) for idx, row in enumerate(rows)]
        for split_name, rows in split_rows.items()
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_preprocess(args: argparse.Namespace):
    if args.out_path is None and args.out_root is None:
        raise SystemExit("Either --out-path or --out-root must be provided.")

    rows, fieldnames = load_rows(args.csv)
    task_schema = build_task_schema(args, fieldnames)
    if args.kfolds and args.kfolds > 1:
        if not args.out_root:
            raise SystemExit("--out-root is required with --kfolds.")
        folds = kfold_indices(len(rows), args.kfolds, args.split_seed)
        all_indices = np.arange(len(rows))
        summaries = {}
        for fold_idx, test_idx in enumerate(folds):
            train_val_idx = np.setdiff1d(all_indices, test_idx, assume_unique=False)
            rng = np.random.default_rng(args.split_seed + fold_idx)
            rng.shuffle(train_val_idx)
            n_val = max(1, int(math.ceil(len(train_val_idx) * args.fold_val_ratio)))
            split_rows = {
                "train": [rows[i] for i in train_val_idx[n_val:]],
                "val": [rows[i] for i in train_val_idx[:n_val]],
                "test": [rows[i] for i in test_idx],
            }
            fold_dir = os.path.join(args.out_root, "folds", f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)
            write_manifest(os.path.join(fold_dir, "split_manifest.json"), split_rows, args.id_column)
            fold_summary = {}
            for split_name, split_data in split_rows.items():
                if not split_data:
                    fold_summary[split_name] = {"num_samples": 0, "num_failed": 0}
                    continue
                out_dir = os.path.join(fold_dir, split_name)
                fold_summary[split_name] = write_lmdb_split(split_data, out_dir, task_schema, args)
            summaries[f"fold_{fold_idx}"] = fold_summary
        with open(os.path.join(args.out_root, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2)
        return summaries

    if args.out_path is not None:
        return write_lmdb_split(rows, args.out_path, task_schema, args)

    if args.split_column:
        split_rows = split_by_column(rows, args.split_column)
    else:
        split_rows = random_split(rows, args.split or [0.8, 0.1, 0.1], args.split_seed)

    os.makedirs(args.out_root, exist_ok=True)
    write_manifest(os.path.join(args.out_root, "split_manifest.json"), split_rows, args.id_column)
    summary = {}
    for split_name, split_data in split_rows.items():
        if not split_data:
            summary[split_name] = {"num_samples": 0, "num_failed": 0}
            continue
        out_dir = os.path.join(args.out_root, split_name)
        summary[split_name] = write_lmdb_split(split_data, out_dir, task_schema, args)
    with open(os.path.join(args.out_root, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
