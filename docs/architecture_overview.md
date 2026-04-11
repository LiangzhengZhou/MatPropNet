# Architecture Overview

This document explains how the repository is organized and how data moves
through the system.

## 1. Top-Level Structure

The main code lives under `src/`.

```text
src/
  matpropnet/
    cli/
    config/
    preprocessing/
    tasks/
    utils/
  ocpmodels/
    common/
    datasets/
    models/
    modules/
    preprocessing/
    tasks/
    trainers/
```

## 2. Why There Are Two Packages

`matpropnet/` contains the new package-facing interface:

- CLI entrypoints
- config loading
- reusable task runners
- runtime utilities

`ocpmodels/` contains the underlying training, dataset, and model code that was
refactored from the original GemNet/OCP-style stack.

In practice:

- end users start from `matpropnet`
- model and trainer internals still live in `ocpmodels`

## 3. Main Execution Layers

### CLI Layer

Files:

- [train.py](../src/matpropnet/cli/train.py)
- [eval.py](../src/matpropnet/cli/eval.py)
- [predict.py](../src/matpropnet/cli/predict.py)
- [preprocess.py](../src/matpropnet/cli/preprocess.py)

Responsibilities:

- parse command-line arguments
- set up runtime logging
- call Python task functions

### Task Layer

File:

- [core.py](../src/matpropnet/tasks/core.py)

Responsibilities:

- load configs
- finalize runtime options
- build trainer and task objects
- run preprocess, train, eval, or predict

### Config Layer

File:

- [loader.py](../src/matpropnet/config/loader.py)

Responsibilities:

- load YAML configs
- resolve `includes`
- apply CLI overrides like `--optim.batch_size=16`
- resolve relative paths against the config file directory

### Model/Trainer Layer

Important files:

- [property_model.py](../src/ocpmodels/models/property_model.py)
- [property_trainer.py](../src/ocpmodels/trainers/property_trainer.py)
- [gemnet.py](../src/ocpmodels/models/gemnet/gemnet.py)

Responsibilities:

- define backbones
- create latent embeddings
- manage task heads
- compute losses and metrics
- run optimization and checkpointing

## 4. Data Flow

The property model follows:

`structure -> backbone -> node_emb -> pooling -> graph_emb -> latent z -> heads -> predictions`

Detailed meaning:

- `backbone`: message passing over the atomic graph
- `node_emb`: atom-level learned features
- `pooling`: graph-level aggregation
- `graph_emb`: one vector per crystal
- `latent z`: shared compact representation
- `heads`: one head per target

## 5. Multi-Task Design

Multi-task learning uses:

- one shared backbone
- one shared latent representation
- one independent head per target

Each target can define:

- task type
- loss type
- loss weight
- class count if classification

The total loss is:

`L_total = sum(weight_i * loss_i)`

with missing labels masked out per target.

## 6. Prediction Exports

The model can export more than just final predictions.

Available outputs:

- `pred`
- `task_preds`
- `z`
- `graph_emb`
- `node_emb`

This is useful for:

- representation analysis
- transfer learning
- clustering
- downstream property modeling

## 7. Backbones

Currently wired into the generic property interface:

- `cgcnn`
- `schnet`
- `gemnet_t`

GemNet now supports both:

- legacy output-block mode
- decoupled backbone mode through `forward_features()`

## 8. Compatibility Strategy

The repository still keeps legacy wrappers:

- [main.py](../main.py)
- [preprocess_property.py](../scripts/preprocess_property.py)

They forward into the new package entrypoints so old command habits do not break
immediately.
