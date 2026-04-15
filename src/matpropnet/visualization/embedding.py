from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Embedding visualization requires matplotlib. "
            "Install matpropnet[visualization] or matplotlib manually."
        ) from exc
    return plt


def _load_sklearn():
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Embedding visualization requires scikit-learn. "
            "Install matpropnet[visualization] or scikit-learn manually."
        ) from exc
    return PCA, TSNE


def _load_umap():
    try:
        import umap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "UMAP visualization requires umap-learn. "
            "Install matpropnet[visualization] or umap-learn manually."
        ) from exc
    return umap


def _parse_vector(entry: str | list[float]) -> list[float]:
    if isinstance(entry, list):
        return [float(value) for value in entry]
    return [float(value) for value in json.loads(entry)]


def _parse_node_embedding(
    entry: str | list[list[float]], reduction: str
) -> list[float]:
    if reduction not in {"mean", "max"}:
        raise ValueError(
            "node_reduction must be one of {'mean', 'max'} when "
            "representation='node_emb'."
        )
    if isinstance(entry, list):
        node_vectors = entry
    else:
        node_vectors = json.loads(entry)
    node_array = np.asarray(node_vectors, dtype=np.float32)
    if node_array.ndim != 2 or node_array.shape[0] == 0:
        raise ValueError("node_emb must contain at least one node vector.")
    if reduction == "mean":
        reduced = node_array.mean(axis=0)
    else:
        reduced = node_array.max(axis=0)
    return reduced.astype(np.float32).tolist()


def _extract_embedding_matrix(
    predictions: dict[str, list[Any]],
    representation: str,
    *,
    node_reduction: str,
) -> np.ndarray:
    if representation not in predictions:
        raise KeyError(
            f"Predictions do not contain '{representation}'. "
            "Make sure the requested representation was exported."
        )
    entries = predictions[representation]
    if not entries:
        raise ValueError(f"No entries found for representation '{representation}'.")

    vectors: list[list[float]] = []
    for entry in entries:
        if representation == "node_emb":
            vectors.append(_parse_node_embedding(entry, node_reduction))
        else:
            vectors.append(_parse_vector(entry))
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(
            f"Expected 2D matrix for representation '{representation}', got {matrix.ndim}D."
        )
    return matrix


def _fit_reducer(
    matrix: np.ndarray,
    reducer_name: str,
    reducer_params: dict[str, Any],
) -> np.ndarray:
    if reducer_name == "pca":
        PCA, _ = _load_sklearn()
        reducer = PCA(
            n_components=int(reducer_params.get("n_components", 2)),
            random_state=reducer_params.get("random_state"),
        )
        return reducer.fit_transform(matrix)
    if reducer_name == "tsne":
        _, TSNE = _load_sklearn()
        reducer = TSNE(
            n_components=int(reducer_params.get("n_components", 2)),
            perplexity=float(reducer_params.get("perplexity", 30.0)),
            learning_rate=float(reducer_params.get("learning_rate", 200.0)),
            max_iter=int(reducer_params.get("n_iter", 1000)),
            random_state=reducer_params.get("random_state"),
            init=reducer_params.get("init", "pca"),
        )
        return reducer.fit_transform(matrix)
    if reducer_name == "umap":
        umap = _load_umap()
        reducer = umap.UMAP(
            n_components=int(reducer_params.get("n_components", 2)),
            n_neighbors=int(reducer_params.get("n_neighbors", 15)),
            min_dist=float(reducer_params.get("min_dist", 0.1)),
            metric=reducer_params.get("metric", "euclidean"),
            random_state=reducer_params.get("random_state"),
        )
        return reducer.fit_transform(matrix)
    raise ValueError(f"Unsupported reducer '{reducer_name}'.")


def _embedding_rows(
    predictions: dict[str, list[Any]],
    reduced: np.ndarray,
    *,
    task_names: list[str],
) -> list[dict[str, Any]]:
    ids = predictions.get("id", [])
    if len(ids) != reduced.shape[0]:
        raise ValueError(
            "Prediction ids and reduced embeddings have mismatched lengths."
        )

    rows: list[dict[str, Any]] = []
    for idx, sample_id in enumerate(ids):
        row: dict[str, Any] = {
            "id": sample_id,
            "x": float(reduced[idx, 0]),
            "y": float(reduced[idx, 1]),
        }
        for task_name in task_names:
            target_key = f"target_{task_name}"
            pred_key = f"pred_{task_name}"
            if target_key in predictions:
                row[target_key] = predictions[target_key][idx]
            if pred_key in predictions:
                row[pred_key] = predictions[pred_key][idx]
        rows.append(row)
    return rows


def _write_embedding_table(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["id", "x", "y"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_embedding_table(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _plot_rows(
    rows: list[dict[str, Any]],
    *,
    task_names: list[str],
    output_dir: Path,
    representation: str,
    reducer_name: str,
    save_format: str,
    plot_params: dict[str, Any],
) -> list[str]:
    plt = _load_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    figsize = plot_params.get("figsize", [8.0, 6.0])
    if isinstance(figsize, str):
        width, height = figsize.split(",")
        figsize = [float(width), float(height)]
    dpi = int(plot_params.get("dpi", 200))
    point_size = float(plot_params.get("point_size", 18))
    alpha = float(plot_params.get("alpha", 0.85))
    cmap = plot_params.get("cmap", "viridis")

    generated_files: list[str] = []
    x = np.asarray([float(row["x"]) for row in rows], dtype=np.float32)
    y = np.asarray([float(row["y"]) for row in rows], dtype=np.float32)

    for task_name in task_names:
        color_key = f"target_{task_name}"
        if color_key not in rows[0]:
            raise KeyError(
                f"Embedding table does not contain '{color_key}' for task coloring."
            )
        colors = np.asarray([float(row[color_key]) for row in rows], dtype=np.float32)
        fig, ax = plt.subplots(figsize=(float(figsize[0]), float(figsize[1])), dpi=dpi)
        scatter = ax.scatter(
            x,
            y,
            c=colors,
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            edgecolors="none",
        )
        ax.set_title(f"{representation} {reducer_name.upper()} colored by {task_name}")
        ax.set_xlabel("dim_1")
        ax.set_ylabel("dim_2")
        fig.colorbar(scatter, ax=ax, label=task_name)
        output_path = output_dir / (
            f"{representation}_{reducer_name}_color_{task_name}.{save_format}"
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        generated_files.append(str(output_path))
    return generated_files


def _save_plot_spec(spec: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, sort_keys=False)


def _load_plot_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def generate_embedding_visualizations(
    predictions: dict[str, list[Any]],
    *,
    task_names: list[str],
    representation: str,
    reducer_name: str,
    reducer_params: dict[str, Any],
    plot_params: dict[str, Any],
    output_dir: str | Path,
    save_format: str = "png",
    node_reduction: str = "mean",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    matrix = _extract_embedding_matrix(
        predictions, representation, node_reduction=node_reduction
    )
    reduced = _fit_reducer(matrix, reducer_name, reducer_params)
    if reduced.shape[1] != 2:
        raise ValueError("Only 2D embedding visualization is supported.")

    rows = _embedding_rows(predictions, reduced, task_names=task_names)
    table_path = output_root / "embedding_table.csv"
    _write_embedding_table(rows, table_path)

    figures_dir = output_root / "figures"
    figure_paths = _plot_rows(
        rows,
        task_names=task_names,
        output_dir=figures_dir,
        representation=representation,
        reducer_name=reducer_name,
        save_format=save_format,
        plot_params=plot_params,
    )

    spec = {
        "representation": representation,
        "node_reduction": node_reduction if representation == "node_emb" else None,
        "reducer": reducer_name,
        "reducer_params": reducer_params,
        "plot_params": plot_params,
        "save_format": save_format,
        "tasks": task_names,
        "embedding_table": str(table_path),
        "figures_dir": str(figures_dir),
        "metadata": metadata or {},
    }
    spec_path = output_root / "plot_spec.yaml"
    _save_plot_spec(spec, spec_path)

    return {
        "embedding_table": str(table_path),
        "plot_spec": str(spec_path),
        "figures": figure_paths,
    }


def replot_embedding_visualizations(
    *,
    plot_spec_path: str | Path,
    embedding_table_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    plot_spec = _load_plot_spec(Path(plot_spec_path))
    table_path = (
        Path(embedding_table_path)
        if embedding_table_path is not None
        else Path(plot_spec["embedding_table"])
    )
    rows = _read_embedding_table(table_path)
    output_root = Path(output_dir) if output_dir is not None else Path(
        plot_spec_path
    ).resolve().parent
    figures_dir = output_root / "figures"
    figure_paths = _plot_rows(
        rows,
        task_names=list(plot_spec.get("tasks", [])),
        output_dir=figures_dir,
        representation=plot_spec["representation"],
        reducer_name=plot_spec["reducer"],
        save_format=plot_spec.get("save_format", "png"),
        plot_params=plot_spec.get("plot_params", {}),
    )
    return {
        "embedding_table": str(table_path),
        "plot_spec": str(Path(plot_spec_path)),
        "figures": figure_paths,
    }
