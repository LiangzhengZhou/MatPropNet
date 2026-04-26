# MatPropNet

MatPropNet is a general materials property prediction framework built around a
decoupled `backbone -> pooling -> latent -> head` design.

This repository must be credited as a downstream fork and substantial refactor of:

- [bt403/ocp-gemnet-gnn](https://github.com/bt403/ocp-gemnet-gnn)
- the broader [Open Catalyst Project / ocp](https://github.com/Open-Catalyst-Project/ocp)

The original codebase focused on OCP-style energy and force workflows. MatPropNet
refocuses the project on reusable crystal and materials property learning.

## What MatPropNet Supports

- single-target property prediction
- dual-target and small multi-task learning
- regression and classification heads
- latent representation extraction
- backbone freezing and staged finetuning
- CSV -> LMDB preprocessing for CIF-based crystal datasets
- GemNet, SpinConv, DimeNet++, ForceNet, DimeNet, SchNet, and CGCNN backbones under one training interface
- unified multi-task loss weighting with static, GradNorm, and uncertainty modes
- checkpoint-driven embedding visualization for `z`, `graph_emb`, and pooled `node_emb`

## Project Layout

```text
configs/property/          property training configs
docs/                      refactor guide and quickstart
ocpmodels/models/          backbones and generic property model
ocpmodels/datasets/        LMDB datasets
ocpmodels/preprocessing/   structure-to-graph conversion
ocpmodels/trainers/        generic property trainer
scripts/preprocess_property.py
```

## Installation

Create an environment and install MatPropNet as a package:

```bash
conda env create -f env.cpu.yml
conda activate matpropnet
pip install -e .
```

If you use GPU, adapt the environment file to your CUDA and PyTorch stack.

## Data Format

MatPropNet expects CSV data like:

```csv
id,target1,target2,cif
mp-10018,33.426734597608046,0.0216390625,"# generated using pymatgen
..."
```

The `cif` column can contain inline multi-line CIF text directly in the CSV.

## CLI Usage

After installation, you can run MatPropNet from any directory:

```bash
matpropnet-preprocess --csv /path/to/all.csv --out-root /path/to/output
matpropnet-train --config /path/to/config.yml
matpropnet-eval --config /path/to/config.yml --checkpoint /path/to/model.pt
matpropnet-predict --config /path/to/config.yml --checkpoint /path/to/model.pt --input /path/to/test.csv --output /path/to/preds.csv
matpropnet-embed-vis --config /path/to/config.yml --checkpoint /path/to/model.pt --lmdb /path/to/test/data.lmdb --representation z --reducer pca --out-dir /path/to/embed_vis
matpropnet-ensemble-train --config /path/to/ensemble.yml
matpropnet-ensemble-predict --manifest /path/to/ensemble_manifest.json --lmdb /path/to/new/data.lmdb --out-dir /path/to/ensemble_pred
matpropnet-ensemble-aggregate --predictions member_0.csv member_1.csv member_2.csv --out ensemble.csv
```

The legacy wrappers `python main.py --mode ...` and `python scripts/preprocess_property.py ...`
are still kept for compatibility.

## CSV -> LMDB

Convert inline-CIF CSV data into LMDB with:

```bash
python scripts/preprocess_property.py \
  --csv data/property/all.csv \
  --out-root data/property \
  --target-columns target1,target2 \
  --task-types regression,regression \
  --id-column id \
  --cif-column cif \
  --cif-mode inline \
  --get-edges \
  --split 0.8 0.1 0.1
```

The generated LMDB records contain:

```python
{
    "id": str,
    "pos": Tensor,
    "atomic_numbers": Tensor,
    "cell": Tensor,
    "targets": Tensor,
    "target_mask": Tensor,
    "num_atoms": int,
}
```

## Train a Property Model

Example GemNet multi-target training:

```bash
python main.py --mode train --config-yml configs/property/gemnet_multitask.yml
```

Or with the package CLI:

```bash
matpropnet-train --config /absolute/path/to/config.yml
```

The property stack supports:

- latent output via `return_latent=True`
- missing-label masking
- weighted multi-task losses
- configurable regression losses with sample weighting and asymmetric penalties
- train / val / test splits
- K-fold preprocessing support

## Backbone and Latent API

The generic forward path is:

```text
input -> backbone -> node_emb -> pooling -> graph_emb -> latent z -> head -> pred
```

Property models can return:

```python
{
    "pred": ...,
    "z": ...,
    "node_emb": ...,
    "graph_emb": ...,
}
```

GemNet can now be used in two modes:

- legacy prediction mode with output blocks
- backbone mode via `forward_features()` for the generic property framework

## Documentation

- [Getting Started](docs/getting_started.md)
- [Architecture Overview](docs/architecture_overview.md)
- [Config Guide](docs/config_guide.md)
- [Refactor Guide](docs/general_materials_framework_refactor.md)
- [Property Quickstart](docs/property_quickstart.md)
- [Loss Configuration](docs/loss_configuration.md)
- [Deep Ensemble Uncertainty](docs/deep_ensemble_uncertainty.md)
- [Embedding Visualization](docs/embedding_visualization.md)

## License and Attribution

- Keep the original [LICENSE.md](LICENSE.md)
- Preserve attribution to `bt403/ocp-gemnet-gnn` and OCP in derivative work
- Document downstream modifications when redistributing or publishing results
