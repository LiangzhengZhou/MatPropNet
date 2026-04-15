from __future__ import annotations

import argparse
import json

from matpropnet.tasks import run_embedding_visualization
from matpropnet.utils.runtime import setup_runtime_logging


def _parse_tasks(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate embedding visualizations from a MatPropNet checkpoint."
    )
    parser.add_argument("--config", help="Path to config YAML.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--lmdb", default=None, help="Optional LMDB path override.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--representation",
        choices=["z", "graph_emb", "node_emb"],
        default="z",
    )
    parser.add_argument(
        "--reducer", choices=["pca", "umap", "tsne"], default="pca"
    )
    parser.add_argument("--tasks", default=None, help="Comma-separated task subset.")
    parser.add_argument(
        "--node-reduction", choices=["mean", "max"], default="mean"
    )
    parser.add_argument("--save-format", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument("--figsize", default="8,6")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", default="euclidean")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-learning-rate", type=float, default=200.0)
    parser.add_argument("--tsne-n-iter", type=int, default=1000)
    parser.add_argument("--plot-spec", default=None)
    parser.add_argument("--embedding-table", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args, overrides = parser.parse_known_args(argv)
    setup_runtime_logging(level=args.log_level, log_file=args.log_file, force=True)

    if args.plot_spec is None and not args.config:
        parser.error("--config is required unless --plot-spec is provided.")

    reducer_params = {
        "n_components": 2,
        "random_state": args.random_state,
        "n_neighbors": args.umap_n_neighbors,
        "min_dist": args.umap_min_dist,
        "metric": args.umap_metric,
        "perplexity": args.tsne_perplexity,
        "learning_rate": args.tsne_learning_rate,
        "n_iter": args.tsne_n_iter,
    }
    plot_params = {
        "figsize": args.figsize,
        "dpi": args.dpi,
        "point_size": args.point_size,
        "alpha": args.alpha,
        "cmap": args.cmap,
    }

    result = run_embedding_visualization(
        args.config,
        checkpoint=args.checkpoint,
        lmdb=args.lmdb,
        output_dir=args.out_dir,
        representation=args.representation,
        reducer=args.reducer,
        tasks=_parse_tasks(args.tasks),
        node_reduction=args.node_reduction,
        reducer_params=reducer_params,
        plot_params=plot_params,
        save_format=args.save_format,
        plot_spec=args.plot_spec,
        embedding_table=args.embedding_table,
        overrides=overrides,
        cpu=args.cpu if args.cpu else None,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
