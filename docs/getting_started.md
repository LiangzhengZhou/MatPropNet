# Getting Started With MatPropNet

This guide is for someone who has never seen the repository before and wants to
go from zero to a successful run as quickly as possible.

## 1. What MatPropNet Is

MatPropNet is a materials property prediction framework for crystal structures.

You can use it to predict one or more targets from structure data such as:

- hardness
- band gap
- formation energy
- elastic properties
- binary or multi-class material labels

It supports:

- single-target learning
- multi-target learning
- regression and classification
- GemNet, CGCNN, and SchNet backbones
- latent embedding export
- CSV to LMDB preprocessing
- training, evaluation, and prediction through CLI commands

## 2. The Core Workflow

Most users will follow this order:

1. Prepare a CSV file with structure and targets.
2. Convert the CSV into LMDB files.
3. Point a config file at those LMDB files.
4. Train a model.
5. Evaluate or predict with a checkpoint.

The overall data flow is:

`CSV + CIF -> LMDB -> backbone -> graph embedding -> latent z -> task heads -> predictions`

## 3. Required CSV Format

The simplest format is:

```csv
id,target1,cif
mp-1,3.14,"... inline CIF text ..."
```

For multiple targets:

```csv
id,target1,target2,target3,cif
mp-1,3.14,1.2,0,"... inline CIF text ..."
```

Notes:

- `id` is your sample identifier.
- `cif` can contain inline multi-line CIF text.
- Missing labels are allowed and will be masked during training.

## 4. Install The Package

From the repository root:

```bash
conda env create -f env.cpu.yml
conda activate matpropnet
pip install -e .
```

After that, the CLI commands should be available:

```bash
matpropnet-preprocess
matpropnet-train
matpropnet-eval
matpropnet-predict
```

## 5. Preprocess Data

Example single-target regression:

```bash
matpropnet-preprocess \
  --csv /path/to/all.csv \
  --out-root /path/to/property_data \
  --target-columns target1 \
  --task-types regression \
  --id-column id \
  --cif-column cif \
  --cif-mode inline \
  --split 0.8 0.1 0.1
```

This creates:

- `train/data.lmdb`
- `val/data.lmdb`
- `test/data.lmdb`
- `summary.json`
- `split_manifest.json`
- `target_schema.json`

## 6. Create A Config

Start from one of the examples in [configs/property](../configs/property).

The most important parts are:

- `dataset`
- `model`
- `task`
- `optim`

Paths are resolved relative to the config file itself, not the current working
directory.

That means you can run:

```bash
cd /tmp
matpropnet-train --config /absolute/path/to/config.yml
```

and it will still find the data correctly.

## 7. Train

```bash
matpropnet-train --config /path/to/config.yml
```

Useful options:

- `--cpu`
- `--log-level DEBUG`
- `--log-file /path/to/train.log`
- `--dry-run`

`--dry-run` loads and resolves the config without starting training.

## 8. Evaluate

```bash
matpropnet-eval \
  --config /path/to/config.yml \
  --checkpoint /path/to/best_checkpoint.pt
```

Regression tasks report:

- `MAE`
- `MSE`
- `RMSE`
- `R²`

Classification tasks report:

- accuracy

## 9. Predict

If you already have a test LMDB in the config:

```bash
matpropnet-predict \
  --config /path/to/config.yml \
  --checkpoint /path/to/best_checkpoint.pt
```

If you want to predict directly from a new CSV:

```bash
matpropnet-predict \
  --config /path/to/config.yml \
  --checkpoint /path/to/best_checkpoint.pt \
  --input /path/to/new_samples.csv \
  --output /path/to/preds.csv
```

The prediction CSV can include:

- predicted targets
- target values if present
- latent `z`
- `graph_emb`
- `node_emb`

These exports are controlled by the config under:

```yaml
task:
  predict:
    export_latent: true
    export_graph_emb: true
    export_node_emb: true
```

## 10. If You Only Need One Model Choice

Set the backbone in the config:

```yaml
model:
  backbone:
    name: cgcnn
```

Available built-in choices:

- `cgcnn`
- `schnet`
- `gemnet_t`

## 11. Where To Look Next

- [Architecture Overview](./architecture_overview.md)
- [Config Guide](./config_guide.md)
- [Property Quickstart](./property_quickstart.md)
- [General Refactor Guide](./general_materials_framework_refactor.md)
