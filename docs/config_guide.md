# Config Guide

This guide explains how to read and edit MatPropNet config files.

## 1. Core Sections

A typical config contains:

```yaml
trainer: property
dataset:
  - src: ../../data/property/train/data.lmdb
  - src: ../../data/property/val/data.lmdb
  - src: ../../data/property/test/data.lmdb
model:
  name: property_model
  backbone:
    name: cgcnn
task:
  dataset: property_lmdb
  tasks:
    target1:
      type: regression
optim:
  batch_size: 16
```

## 2. `includes`

Configs can inherit from other configs:

```yaml
includes:
  - base.yml
```

Relative include paths are resolved relative to the config file that contains
them.

## 3. Dataset Paths

Paths like:

```yaml
dataset:
  - src: ../../data/property/train/data.lmdb
```

are resolved relative to the config file directory, not the shell working
directory.

This is one of the main engineering changes that makes the package usable from
any directory.

## 4. Choosing A Backbone

Examples:

```yaml
model:
  backbone:
    name: cgcnn
```

```yaml
model:
  backbone:
    name: schnet
```

```yaml
model:
  backbone:
    name: gemnet_t
```

Additional supported backbone names:

- `spinconv`
- `dimenetplusplus` or `dimenet_plus_plus`
- `forcenet`

## 5. Defining Tasks

Single target:

```yaml
task:
  tasks:
    hardness:
      type: regression
      loss: mae
      weight: 1.0
```

For non-negative physical properties such as hardness, a regression task can
constrain the exported prediction with a softplus output activation:

```yaml
task:
  tasks:
    H:
      type: regression
      loss: mae
      weight: 1.0
      output_activation: softplus
```

When target normalization is enabled, MatPropNet applies softplus in the
physical target space and maps the value back to normalized space for the loss.
This keeps old normalization, checkpoint, metric, and prediction-export behavior
compatible while preventing negative physical predictions. The default remains
linear if `output_activation` is omitted.

Mixed multi-task:

```yaml
task:
  tasks:
    hardness:
      type: regression
      loss: mae
      weight: 1.0
    metallicity:
      type: classification
      loss: cross_entropy
      num_classes: 2
      weight: 0.5
```

## 6. Latent Export

To include latent representations in prediction output:

```yaml
task:
  predict:
    export_latent: true
    export_graph_emb: true
    export_node_emb: true
```

## 7. Finetuning Options

```yaml
task:
  finetune:
    freeze_backbone: true
    freeze_first_k_blocks: 2
```

These are applied in the property trainer before optimization starts.

## 8. Optimizer Parameter Groups

You can give different learning rates to different parts of the model:

```yaml
optim:
  param_groups:
    - name: backbone
      lr: 1.0e-4
    - name: latent
      lr: 3.0e-4
    - name: heads
      lr: 1.0e-3
```

Available built-in group names:

- `backbone`
- `latent`
- `heads`

## 9. Logging Options

CLI logging:

```bash
matpropnet-train --config config.yml --log-level DEBUG --log-file train.log
```

This controls console and file logging for the application runtime.

Training metrics still flow through the configured trainer logger such as
TensorBoard.

## 10. CLI Overrides

You can override nested values from the command line:

```bash
matpropnet-train \
  --config config.yml \
  --optim.batch_size=8 \
  --task.tasks.target1.weight=2.0
```

This is useful for quick experiments without copying config files.

## 11. Loss Weighting

MatPropNet now supports a unified multi-task loss weighting section:

```yaml
loss_weighting:
  mode: static
```

Supported modes:

- `static`
- `gradnorm`
- `uncertainty`

If you omit `loss_weighting`, MatPropNet automatically falls back to:

```yaml
loss_weighting:
  mode: static
```

See [`loss_weighting.md`](./loss_weighting.md) for mode-specific guidance, paper mapping, and logging fields.
