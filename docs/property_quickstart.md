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
python scripts/preprocess_property.py \
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
python main.py --mode train --config-yml configs/property/cgcnn_multitask.yml
```

That config uses:

- trainer: `property`
- model: `property_model`
- backbone: `cgcnn`
- pooling -> latent -> independent task heads

To train with GemNet as the decoupled backbone:

```bash
python main.py --mode train --config-yml configs/property/gemnet_multitask.yml
```

This path uses the new `GemNetT.forward_features()` backbone interface instead of the old
`out_blocks`-driven task output path.


## 4. Predict with a checkpoint

```bash
python main.py --mode predict \
  --config-yml configs/property/cgcnn_multitask.yml \
  --checkpoint checkpoints/<run-id>/best_checkpoint.pt
```


## 5. Finetune controls

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
