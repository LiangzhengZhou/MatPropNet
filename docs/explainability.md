# Explainability

MatPropNet provides post-hoc graph explanation for trained property models. The
workflow loads an existing checkpoint or Deep Ensemble manifest, learns an edge
importance mask for each requested sample, and writes materials-oriented tables.

The explainer is not part of normal training. Train the property model first,
then run explanation on a checkpoint:

```bash
matpropnet-explain \
  --config /path/to/config.yml \
  --checkpoint /path/to/best_checkpoint.pt \
  --lmdb /path/to/test/data.lmdb \
  --out-dir /path/to/explain_out \
  --algorithm matpropnet_edge_mask \
  --target-index 0 \
  --num-samples 20 \
  --top-k 20
```

## Deep Ensemble Explanation

Deep Ensemble explanation runs the explainer independently for each member, then
aggregates the masks:

```bash
matpropnet-ensemble-explain \
  --manifest /path/to/ensemble_manifest.json \
  --lmdb /path/to/test/data.lmdb \
  --out-dir /path/to/ensemble_explain \
  --algorithm matpropnet_edge_mask \
  --target-index 0 \
  --num-samples 20 \
  --top-k 20
```

For edge `e`, ensemble output includes:

```text
mean_mask_e = mean_m mask_e^(m)
std_mask_e = std_m mask_e^(m)
relative_uncertainty_e = std_mask_e / (mean_mask_e + eps)
confidence_e = mean_mask_e / (std_mask_e + eps)
```

## Outputs

Each sample gets its own directory:

```text
explain_out/
  config_resolved.yml
  explain_manifest.json
  sample_id/
    explanation_edges.csv
    bond_type_importance.csv
    explanation_summary.json
    masks.pt
```

`explanation_edges.csv` contains one row per graph edge with source/destination
node ids, elements, bond type, distance when available or computable, periodic
cell offset, edge mask, rank, and top-k membership.

`bond_type_importance.csv` aggregates edge masks by element pair, including mean,
max, sum, top-k count, and mean distance.

For ensembles, edge rows also contain member masks, ensemble mean/std, relative
uncertainty, and confidence.

## Interpretation

`edge_mask` is a model attribution score: it measures how strongly the trained
MatPropNet prediction depends on an edge under the explainer objective. It is not
a bond energy, COHP, ICOHP, force constant, or direct physical observable.

For graph-level regression, the explainer also records top-k sufficiency and
necessity-style metrics:

```text
y_full = f(G)
y_keep = f(G_topk)
y_remove = f(G_without_topk)
sufficiency_error = |y_keep - y_full|
necessity_drop = |y_remove - y_full|
```

## Known Limitations

- Edge masks are model attributions, not ground-truth bonding mechanisms.
- Periodic edges are reported with cell offsets; simple connectivity summaries
  should be interpreted as original-cell graph approximations.
- GemNet and DimeNet++ propagate masks through their internal radial basis,
  spherical/circular basis, triplet interactions, and edge-to-node updates.
  Triplet contributions use the product of the participating edge masks.
- Complex geometric backbone masks are MatPropNet-native attributions. Their
  semantics differ from PyG's built-in MessagePassing mask injection and should
  still be validated with deletion baselines for publication use.
- Regression fidelity is continuous and not identical to classification
  accuracy-based GraphFramEx fidelity.
