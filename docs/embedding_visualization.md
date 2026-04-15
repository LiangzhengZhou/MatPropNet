# Embedding Visualization

MatPropNet can turn a trained property checkpoint plus an LMDB dataset into
2D embedding plots. The visualization path reuses the existing prediction
pipeline so it can extract the same latent outputs already supported by
`matpropnet-predict`.

## Supported representations

- `z`: task-facing latent representation after the latent projector
- `graph_emb`: pooled graph-level embedding before the latent projector
- `node_emb`: atom-level embeddings aggregated per sample with `mean` or `max`

For most material-level analyses, start with `z` or `graph_emb`.

## Supported reducers

- `pca`
- `umap`
- `tsne`

Install optional visualization dependencies first if needed:

```bash
pip install -e .[visualization]
```

## Basic usage

```bash
matpropnet-embed-vis \
  --config /absolute/path/to/config.yml \
  --checkpoint /absolute/path/to/best_checkpoint.pt \
  --lmdb /absolute/path/to/test/data.lmdb \
  --representation z \
  --reducer pca \
  --out-dir /absolute/path/to/embed_vis
```

This command:

1. loads the checkpoint and dataset
2. extracts the requested representation
3. reduces it to 2D
4. writes one figure per task, colored by that task's target value
5. saves the plotting spec for later reuse

## Multi-task behavior

If your checkpoint predicts multiple tasks, MatPropNet writes one plot per task.

Example outputs:

- `z_pca_color_log_b.png`
- `z_pca_color_log_g.png`

This keeps each task on its own color scale and avoids mixed-color ambiguity.

## Replotting without rerunning the model

Each visualization run writes:

- `embedding_table.csv`
- `plot_spec.yaml`

To regenerate the figures later:

```bash
matpropnet-embed-vis \
  --plot-spec /absolute/path/to/embed_vis/plot_spec.yaml \
  --embedding-table /absolute/path/to/embed_vis/embedding_table.csv \
  --out-dir /absolute/path/to/embed_vis_replot
```

This is useful when you want to change only the output location or regenerate
the same figures on another machine.

## Common parameters

### Representation and reducer

```bash
--representation z|graph_emb|node_emb
--reducer pca|umap|tsne
--node-reduction mean|max
```

### Plot appearance

```bash
--figsize 8,6
--dpi 200
--point-size 18
--alpha 0.85
--cmap viridis
--save-format png|pdf|svg
```

### Reducer-specific controls

```bash
--random-state 0
--umap-n-neighbors 15
--umap-min-dist 0.1
--umap-metric euclidean
--tsne-perplexity 30
--tsne-learning-rate 200
--tsne-n-iter 1000
```

## Output structure

```text
embed_vis/
  embedding_table.csv
  plot_spec.yaml
  figures/
    z_pca_color_task1.png
    z_pca_color_task2.png
```

The CSV stores the 2D coordinates together with sample ids and task labels so
you can inspect or post-process the embedding outside the CLI.
