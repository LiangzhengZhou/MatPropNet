# Property Quickstart

This quickstart belongs to MatPropNet, a downstream fork and refactor of
`bt403/ocp-gemnet-gnn` for general materials property prediction.

This quickstart shows how to use the new generic property pipeline in this repository for CSV
inputs shaped like:

```csv
id,target1,target2,cif
mp-10018,33.426734597608046,0.0216390625,"# generated using pymatgen
data_Ac
..."
```

The `cif` column can contain inline CIF text exactly like the example above. It can also contain
file paths if you switch the preprocessing mode.


## 1. Convert CSV to LMDB

Create train/val/test LMDBs from a single CSV:

```bash
matpropnet-preprocess \
  --csv data/property/all.csv \
  --out-root data/property \
  --target-columns target1,target2 \
  --task-types regression,regression \
  --cif-column cif \
  --id-column id \
  --cif-mode inline \
  --get-edges \
  --split 0.8 0.1 0.1
```

This writes:

- `data/property/train/data.lmdb`
- `data/property/val/data.lmdb`
- `data/property/test/data.lmdb`
- `data/property/split_manifest.json`
- `data/property/*/target_schema.json`
- `data/property/summary.json`

Stored LMDB samples contain:

```python
{
    "sample_id": str,
    "pos": Tensor,
    "atomic_numbers": Tensor,
    "cell": Tensor,
    "y": Tensor,
    "target_mask": Tensor,
}
```


## 2. K-fold generation

Generate five folds:

```bash
python scripts/preprocess_property.py \
  --csv data/property/all.csv \
  --out-root data/property_kfold \
  --target-columns target1,target2 \
  --task-types regression,regression \
  --cif-mode inline \
  --get-edges \
  --kfolds 5 \
  --fold-val-ratio 0.1
```

This writes fold directories under `data/property_kfold/folds/fold_*/`.


## 3. Train a generic multi-target model

Use the provided example config:

```bash
matpropnet-train --config configs/property/cgcnn_multitask.yml
```

That config uses:

- trainer: `property`
- model: `property_model`
- backbone: `cgcnn`
- pooling -> latent -> independent task heads

To train with GemNet as the decoupled backbone:

```bash
matpropnet-train --config configs/property/gemnet_multitask.yml
```

This path uses the new `GemNetT.forward_features()` backbone interface instead of the old
`out_blocks`-driven task output path.

Additional backbone examples:

```bash
matpropnet-train --config configs/property/cgcnn_single_task.yml
matpropnet-train --config configs/property/schnet_single_task.yml
matpropnet-train --config configs/property/gemnet_single_task.yml
matpropnet-train --config configs/property/dimenet_single_task.yml
matpropnet-train --config configs/property/dimenetplusplus_single_task.yml
matpropnet-train --config configs/property/forcenet_single_task.yml
matpropnet-train --config configs/property/spinconv_single_task.yml
```

Multi-task backbone examples:

```bash
matpropnet-train --config configs/property/dimenet_multitask.yml
matpropnet-train --config configs/property/dimenetplusplus_multitask.yml
matpropnet-train --config configs/property/forcenet_multitask.yml
matpropnet-train --config configs/property/spinconv_multitask.yml
```


## 4. Predict with a checkpoint

```bash
matpropnet-predict \
  --config configs/property/cgcnn_multitask.yml \
  --checkpoint checkpoints/<run-id>/best_checkpoint.pt
```


## 5. Visualize embeddings from a checkpoint

Generate 2D embedding plots from `z` or `graph_emb` for an existing LMDB split:

```bash
matpropnet-embed-vis \
  --config configs/property/gemnet_multitask.yml \
  --checkpoint checkpoints/<run-id>/best_checkpoint.pt \
  --lmdb data/property/test/data.lmdb \
  --representation z \
  --reducer pca \
  --out-dir analysis/embed_vis
```

This writes:

- `analysis/embed_vis/embedding_table.csv`
- `analysis/embed_vis/plot_spec.yaml`
- `analysis/embed_vis/figures/*.png`

For multi-task runs, the command generates one figure per task, each colored by
that task's target value.

Replot later without rerunning the model:

```bash
matpropnet-embed-vis \
  --plot-spec analysis/embed_vis/plot_spec.yaml \
  --embedding-table analysis/embed_vis/embedding_table.csv \
  --out-dir analysis/embed_vis_replot
```


## 6. Finetune controls

Add this to the config if you want to freeze the backbone:

```yaml
task:
  finetune:
    freeze_backbone: true
    freeze_first_k_blocks: 2
```

You can also use different learning rates for backbone, latent projector, and heads via:

```yaml
optim:
  param_groups:
    - name: backbone
      lr: 1.0e-4
    - name: latent
      lr: 5.0e-4
    - name: heads
      lr: 1.0e-3
```
