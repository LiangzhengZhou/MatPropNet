# General Materials Property Framework Refactor Guide

This document translates the current OCP-style codebase into a concrete refactor plan for a
general-purpose materials property learning framework. It is written for this repository, not as
an abstract architecture note, so each section maps the new design back to the files that exist
today.

This repository is a downstream fork and substantial refactor of
`bt403/ocp-gemnet-gnn`, which itself builds on the Open Catalyst Project ecosystem.

The target is a framework that supports:

- single-target regression
- dual-target learning
- small-scale multi-task learning
- regression + classification in one model
- latent representation extraction
- flexible finetuning and freezing
- a reusable CSV -> LMDB -> training pipeline


## 1. Why the current codebase needs refactoring

The current project inherits the Open Catalyst Project training style. That design works well for
energy/force tasks, but it creates friction for general material property prediction:

- model outputs are coupled to OCP tasks such as energy and forces
- `out_blocks` in GemNet act as both readout and task head
- trainer logic assumes task-specific prediction keys like `energy` and `forces`
- dataset utilities are optimized for OCP structures instead of general property tables
- loss and metrics are organized around a small set of built-in OCP tasks

In practice, this makes it awkward to add targets such as hardness, band gap, formation energy,
bulk modulus, elastic constants, phase class, or mixed property bundles.


## 2. Refactor target architecture

The target data flow should be:

`input -> backbone -> node_emb -> pooling -> graph_emb -> latent z -> head -> pred`

Responsibilities of each module:

- `backbone`: message passing and structure representation learning
- `pooling`: converts node-level features into graph-level features
- `graph_emb`: pooled crystal representation
- `latent z`: compact shared representation exposed to downstream tasks
- `head`: maps `z` to one or more task outputs

The key principle is that the backbone should not know whether the downstream task is hardness
regression, band-gap classification, or joint multi-property prediction.


## 3. How the current repository maps to the old design

The most important coupling points in this repo are:

- `main.py`
  - builds config
  - instantiates trainer through the registry
  - instantiates task through the registry
- `ocpmodels/trainers/base_trainer.py`
  - combines dataset loading, model creation, optimizer, scheduler, logging, and task-specific
    evaluation conventions
- `ocpmodels/trainers/energy_trainer.py`
  - assumes outputs like `energy`
  - owns parts of loss behavior and prediction export
- `ocpmodels/trainers/forces_trainer.py`
  - assumes `energy` + `forces`
  - owns relaxations and force-specific prediction flow
- `ocpmodels/tasks/task.py`
  - task execution still depends on OCP-specific trainer types
- `ocpmodels/datasets/lmdb_dataset.py`
  - loads LMDBs well, but the stored sample schema is not yet standardized for general property
    learning
- `ocpmodels/models/gemnet/gemnet.py`
  - `out_blocks` directly produce final outputs
  - output semantics are still tied to energy/force style training


## 4. New project structure

The target top-level structure should look like:

```text
models/
data/
tasks/
trainer/
utils/
```

Inside this repository, a practical migration path is:

```text
ocpmodels/
  models/
    backbones/
    pooling/
    heads/
    property_model.py
  data/
    datasets/
    preprocess/
    splitters.py
  tasks/
    property_task.py
  trainers/
    property_trainer.py
  utils/
    losses.py
    metrics.py
    finetune.py
```

You do not need to delete the old OCP modules immediately. A safer approach is:

1. add the new generic path in parallel
2. migrate configs and experiments
3. remove obsolete task-specific pieces after parity is confirmed


## 5. Backbone/readout/head decoupling plan

### 5.1 Backbone

Backbone modules should stop producing task-specific predictions. They should output only learned
representations, for example:

```python
{
    "node_emb": node_emb,
    "edge_emb": edge_emb,
}
```

For GemNet, this means:

- keep interaction blocks and geometric encodings
- remove the assumption that `out_blocks` are the final prediction path
- expose the final node representation after message passing

Suggested direction:

- move GemNet geometric and interaction logic into a reusable backbone class
- rename task-facing output code into a separate readout/head path

### 5.2 Pooling

Introduce explicit graph pooling modules such as:

- `MeanPooling`
- `SumPooling`
- `AttentionPooling`
- `Set2SetPooling` or other advanced graph pooling if needed later

A simple interface is enough:

```python
graph_emb = pooling(node_emb, batch_index)
```

For extensive properties:

- use sum-style pooling or atom-count-aware scaling

For intensive properties:

- use mean-style pooling or a learned normalization-aware pooling

### 5.3 Latent projection

Add a dedicated latent projection block:

```python
z = latent_projector(graph_emb)
```

This is useful for:

- downstream finetuning
- feature export
- clustering and visualization
- transfer learning across property families

### 5.4 Heads

Heads should be independent from the backbone and instantiated from config.

Examples:

- `RegressionHead`
- `ClassificationHead`
- `MultiTaskHead`
- `HeteroscedasticRegressionHead`

For a multi-task setup, a clean pattern is:

```python
self.heads = nn.ModuleDict({
    "hardness": RegressionHead(...),
    "band_gap": RegressionHead(...),
    "metallicity": ClassificationHead(...),
})
```


## 6. Standard forward interface

All property models should converge on a stable forward API:

```python
def forward(self, data, return_latent=False):
    ...
```

Recommended return format:

```python
{
    "pred": pred,
    "z": z,
    "node_emb": node_emb,
    "graph_emb": graph_emb,
}
```

For multi-task cases:

```python
{
    "pred": {
        "hardness": hardness_pred,
        "band_gap": band_gap_pred,
        "metallicity": metallicity_logits,
    },
    "z": z,
    "node_emb": node_emb,
    "graph_emb": graph_emb,
}
```

Design notes:

- always return a dictionary
- keep representation keys stable
- let `pred` be either a tensor or a task dictionary
- avoid returning task-specific names like `energy` unless compatibility mode is required


## 7. What to do with `out_blocks`

This is the most important model-side change.

Today in GemNet:

- `out_blocks` are deeply embedded in model execution
- they behave like task heads
- they preserve the OCP-style prediction path

Refactor goal:

- either remove `out_blocks` entirely
- or shrink them into a backbone-side node update/readout helper that is no longer a final task
  predictor

Recommended strategy:

1. create a new generic GemNet backbone class
2. keep interaction blocks
3. stop using `OutputBlock` as the final prediction layer
4. emit final node embeddings
5. attach pooling and head outside the backbone

If immediate deletion is too risky, do this in two phases:

- Phase A: keep `out_blocks` only as an internal feature refinement block
- Phase B: replace them with a thinner feature projection layer once the new path is stable


## 8. Finetuning and freezing support

The new model base class should support:

```python
freeze_backbone()
freeze_first_k_blocks(k)
unfreeze_all()
```

Recommended implementation pattern:

```python
def freeze_backbone(self):
    for p in self.backbone.parameters():
        p.requires_grad = False

def freeze_first_k_blocks(self, k):
    for block in self.backbone.blocks[:k]:
        for p in block.parameters():
            p.requires_grad = False

def unfreeze_all(self):
    for p in self.parameters():
        p.requires_grad = True
```

Also add optimizer parameter groups:

- lower LR for pretrained backbone
- higher LR for new heads
- optional separate LR for latent projector

Example config idea:

```yaml
optim:
  optimizer: AdamW
  lr: 1e-3
  param_groups:
    - name: backbone
      lr: 1e-4
    - name: latent
      lr: 5e-4
    - name: heads
      lr: 1e-3
```


## 9. Multi-task learning design

The framework should support a shared backbone and separate heads.

Recommended sample definition:

```yaml
task:
  tasks:
    hardness:
      type: regression
      weight: 1.0
    band_gap:
      type: regression
      weight: 0.5
    metallicity:
      type: classification
      num_classes: 2
      weight: 0.3
```

Implementation principles:

- shared backbone and latent `z`
- one head per task
- one loss function per task
- one metric set per task
- weighted sum of valid losses
- mask invalid or missing labels

Recommended batch format:

```python
batch.targets
batch.target_mask
```

Example:

```python
targets = {
    "hardness": tensor(...),
    "band_gap": tensor(...),
    "metallicity": tensor(...),
}

target_mask = {
    "hardness": tensor(..., dtype=torch.bool),
    "band_gap": tensor(..., dtype=torch.bool),
    "metallicity": tensor(..., dtype=torch.bool),
}
```

This allows partial labels, which is critical for real materials datasets.


## 10. Loss and metric system refactor

The old loss system is too narrow for a general framework. Replace task-specific logic with a
task registry or config-driven builder.

Recommended losses:

- regression: `L1`, `MSE`, `Huber`
- classification: `CrossEntropy`, `BCEWithLogits`
- uncertainty-aware regression: optional heteroscedastic loss

Recommended metric families:

- regression: `MAE`, `RMSE`, `R2`
- classification: `Accuracy`, `F1`, `AUROC`

Suggested total loss pattern:

```python
total_loss = 0.0
for task_name, spec in task_specs.items():
    if valid_mask.any():
        task_loss = loss_fn(pred[task_name], target[task_name])
        total_loss += spec.weight * task_loss
```

Important:

- compute each task loss only on valid labels
- log each task loss independently
- track a configurable `primary_metric` for checkpoint selection


## 11. Trainer refactor

### 11.1 Current limitation

`BaseTrainer`, `EnergyTrainer`, and `ForcesTrainer` currently mix these responsibilities:

- task semantics
- loss definition
- prediction export format
- model execution
- scheduler and optimizer orchestration

### 11.2 New direction

Add a generic `PropertyTrainer` with the following responsibilities:

- load datasets
- build the generic property model
- build task-aware loss functions
- handle train/val/test loops
- support single-task and multi-task metrics
- support early stopping
- export predictions in a task-agnostic format

Keep OCP-specific trainers during migration, but stop extending them for new property work.

### 11.3 Trainer output conventions

The trainer should no longer assume keys like `energy` or `forces`.

Instead, standardize on:

```python
model_out["pred"]
batch.targets
batch.target_mask
```

Prediction export examples:

- single target:
  - `id`, `target`, `prediction`
- multi-task:
  - `id`, `hardness_target`, `hardness_pred`, `band_gap_target`, `band_gap_pred`, ...


## 12. Dataset and LMDB schema redesign

The target CSV input is:

```text
id,target1,target2,cif
```

This should be generalized so the CSV can contain:

- one or more target columns
- one CIF path or inline structure source
- optional split column
- optional metadata columns

Recommended LMDB sample schema:

```python
{
    "id": str,
    "pos": Tensor,
    "atomic_numbers": Tensor,
    "cell": Tensor,
    "targets": Tensor,
    "target_mask": Tensor,
    "target_names": list,
    "num_atoms": int
}
```

If task types vary, add task metadata in config rather than each sample whenever possible.

Suggested preprocessing directory layout:

```text
data/
  raw/
    properties.csv
    cifs/
  processed/
    train/
      data.lmdb
      metadata.npz
    val/
      data.lmdb
      metadata.npz
    test/
      data.lmdb
      metadata.npz
    split.json
    target_schema.json
```


## 13. CSV -> LMDB preprocessing tutorial

### 13.1 Input CSV examples

Single-task regression:

```csv
id,hardness,cif
mat_001,12.4,data/cifs/mat_001.cif
mat_002,9.8,data/cifs/mat_002.cif
```

Dual-task regression:

```csv
id,hardness,band_gap,cif
mat_001,12.4,1.85,data/cifs/mat_001.cif
mat_002,9.8,0.00,data/cifs/mat_002.cif
```

Mixed regression + classification:

```csv
id,hardness,metallicity,cif
mat_001,12.4,0,data/cifs/mat_001.cif
mat_002,9.8,1,data/cifs/mat_002.cif
```

### 13.2 Preprocessing requirements

The new preprocessing script should:

1. read CSV rows
2. load CIF structures
3. convert structures to graph objects
4. collect one or more targets
5. generate a label mask for missing values
6. serialize into LMDB
7. save target schema and split metadata

### 13.3 Missing-label support

If a target is missing:

- fill `targets[i]` with a placeholder value such as `0`
- set `target_mask[i] = 0`
- exclude this entry from the corresponding task loss

This enables training on partially labeled materials datasets.


## 14. Train/valid/test and K-fold splitting

The framework should support both standard random splitting and K-fold evaluation.

### 14.1 Standard split

Requirements:

- deterministic split with seed
- saved split indices or IDs
- optional stratification for classification tasks

Suggested saved artifact:

```json
{
  "seed": 42,
  "train_ids": ["mat_001", "mat_002"],
  "val_ids": ["mat_101"],
  "test_ids": ["mat_201"]
}
```

### 14.2 K-fold split

Requirements:

- configurable `k`
- each fold rotates the held-out test partition
- train partition is further split into train/valid
- each fold saved for reproducibility

Suggested directory layout:

```text
processed/
  folds/
    fold_0/
      train.lmdb
      val.lmdb
      test.lmdb
    fold_1/
      ...
```


## 15. Suggested model configuration style

A target config for the new framework can look like:

```yaml
model:
  name: property_model
  backbone:
    name: gemnet_backbone
    num_blocks: 4
    cutoff: 6.0
    max_neighbors: 50
  pooling:
    name: mean
  latent:
    hidden_dim: 256
    out_dim: 128
  heads:
    hardness:
      type: regression
      out_dim: 1
    band_gap:
      type: regression
      out_dim: 1
    metallicity:
      type: classification
      out_dim: 2

task:
  primary_metric: hardness_mae
  tasks:
    hardness:
      type: regression
      loss: l1
      metric: mae
      weight: 1.0
    band_gap:
      type: regression
      loss: l1
      metric: mae
      weight: 0.5
    metallicity:
      type: classification
      loss: cross_entropy
      metric: accuracy
      weight: 0.3
```

This config style is more extensible than encoding task behavior inside the trainer class.


## 16. Migration plan by phase

This is the safest implementation sequence for this repository.

### Phase 1: introduce generic model API

Goals:

- add a generic property model wrapper
- expose `node_emb`, `graph_emb`, `z`
- keep current training path working

Deliverables:

- new `property_model.py`
- initial pooling module
- latent projector
- generic regression head

### Phase 2: replace task-coupled output logic

Goals:

- remove the dependency on `out_blocks` for final prediction
- convert GemNet into a reusable backbone

Deliverables:

- `GemNetBackbone`
- separate pooling and head path

### Phase 3: add finetune and freezing utilities

Goals:

- backbone freezing
- partial block freezing
- optimizer param groups

Deliverables:

- freeze APIs on model
- param-group builder utilities

### Phase 4: add multi-task training

Goals:

- multiple heads
- multiple losses
- missing-label masking

Deliverables:

- `MultiTaskHead`
- generic loss builder
- generic metric builder
- task-aware export logic

### Phase 5: rebuild data system

Goals:

- CSV -> LMDB with target schema
- standard splits and K-fold

Deliverables:

- preprocessing script for general materials data
- split utility
- saved schema metadata

### Phase 6: deprecate old task-specific paths

Goals:

- reduce maintenance burden
- keep compatibility only where needed

Candidates for cleanup:

- force-only logic not needed for property prediction
- energy/force-only loss conventions
- OCP-specific export helpers in general training flow


## 17. README and license guidance

Since this repository is derived from OCP code:

- keep the original `LICENSE.md`
- preserve author attribution and source notice
- document clearly which parts were inherited and which parts were refactored

Recommended README additions:

- project goal: general materials property prediction
- supported tasks and target types
- preprocessing guide
- finetune guide
- latent extraction guide
- migration note from original OCP-style design


## 18. Practical implementation checklist

Use this as the execution checklist while refactoring.

- [ ] create a generic backbone interface
- [ ] create explicit pooling modules
- [ ] add latent projector and latent-return API
- [ ] create generic single-task regression head
- [ ] create generic classification head
- [ ] create multi-task head container
- [ ] add `freeze_backbone()`
- [ ] add `freeze_first_k_blocks(k)`
- [ ] add `unfreeze_all()`
- [ ] refactor trainer to stop assuming `energy` and `forces`
- [ ] refactor loss system to be task-driven
- [ ] add target masks for missing labels
- [ ] implement CSV -> LMDB preprocessing
- [ ] implement deterministic split saving
- [ ] implement K-fold support
- [ ] update README and tutorials


## 19. Recommended first implementation targets in this repo

If you want the highest-value path with the lowest immediate risk, start here:

1. `ocpmodels/models/`
   - extract GemNet backbone
   - add pooling and latent modules
2. `ocpmodels/trainers/`
   - add `property_trainer.py`
3. `ocpmodels/tasks/`
   - add a generic property task
4. `scripts/`
   - add a CSV -> LMDB preprocessing script for general materials properties
5. `configs/`
   - add one single-task and one multi-task example

That sequence preserves the maximum amount of existing code while opening the path to a real
general-purpose framework.


## 20. Final outcome

After the refactor, this repository should become:

- a general material property learning framework
- capable of single-task, dual-task, and small multi-task learning
- compatible with regression and classification together
- able to expose latent embeddings for downstream analysis
- suitable for pretraining and finetuning workflows

That is the architectural shift from:

`OCP task implementation`

to:

`general, reusable materials representation learning framework`
