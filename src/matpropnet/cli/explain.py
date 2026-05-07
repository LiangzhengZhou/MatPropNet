from __future__ import annotations

import argparse

from matpropnet.explain import explain_checkpoint


def build_parser():
    parser = argparse.ArgumentParser(
        description="Explain a trained MatPropNet checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lmdb", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--algorithm",
        default="matpropnet_edge_mask",
        choices=["matpropnet_edge_mask", "auto"],
    )
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--edge-size-weight", type=float, default=0.005)
    parser.add_argument("--edge-entropy-weight", type=float, default=0.001)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    explain_checkpoint(
        args.config,
        checkpoint=args.checkpoint,
        lmdb=args.lmdb,
        output_dir=args.out_dir,
        algorithm=args.algorithm,
        target_index=args.target_index,
        num_samples=args.num_samples,
        top_k=args.top_k,
        epochs=args.epochs,
        lr=args.lr,
        edge_size_weight=args.edge_size_weight,
        edge_entropy_weight=args.edge_entropy_weight,
        cpu=args.cpu,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
