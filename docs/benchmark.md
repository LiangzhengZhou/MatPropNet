# Model Benchmark CLI

`matpropnet-benchmark` runs a list of normal single-model training configs one
after another and writes a shared comparison table. It is intended for the
first-stage question: which backbone works best on the same preprocessed LMDB
split, before spending GPU time on deep ensembles.

## When To Use It

Use this command when you already have train / val / test LMDB files and want to
compare several model configs under the same task, loss, split, and seed. A
typical run is GemNet, SpinConv, DimeNet++, ForceNet, DimeNet, SchNet, and
CGCNN on the same single-task `H` dataset with MAE loss.

This workflow does not change `matpropnet-train`. Each model is still trained
through the existing trainer, checkpoint, early-stopping, prediction export, and
TensorBoard paths.

## Command

```bash
matpropnet-benchmark --config /path/to/benchmark.yml
```

To validate the plan without training:

```bash
matpropnet-benchmark --config /path/to/benchmark.yml --dry-run
```

## Benchmark Config

```yaml
benchmark:
  name: H_backbone_mae_benchmark_seed42
  output_dir: /path/to/Running/H_backbone_mae_benchmark_seed42
  seed: 42
  print_every: 10
  stop_on_error: false

  train:
    checkpoint_name: best_checkpoint.pt
    log_file_name: train.log
    log_level: INFO
    amp: false
    cpu: false

  evaluate:
    splits:
      - val
      - test

  models:
    - name: gemnet_t
      config: /path/to/config_H_gemnet_mae.yml
    - name: schnet
      config: /path/to/config_H_schnet_mae.yml
```

Each file listed under `models[].config` is a regular `matpropnet-train` config.
For a fair comparison, keep the dataset, task, loss, batch-size policy, and
early-stopping metric aligned across those files, and only change the backbone
and model-size-specific optimizer settings.

## Outputs

The benchmark output directory contains:

```text
benchmark_config.yml
benchmark_manifest.json
runs/
  gemnet_t/
    config.resolved.yml
    train.log
    checkpoints/...
    results/...
  schnet/
    ...
summary/
  benchmark_summary.csv
  benchmark_summary.json
```

`benchmark_summary.csv` has one row per model. Metric columns are flattened as
`val_H_mae`, `test_H_mae`, `test_H_rmse`, and so on, based on whatever metrics
the trainer returns. If a model fails and `stop_on_error: false`, the workflow
records `status=failed` and the error message, then continues to the next model.

## Practical Notes

- Use the same preprocessed LMDB split for every config.
- Use the same `task.primary_metric` so early stopping is comparable.
- Start with MAE or Huber for the backbone screen; save Gaussian NLL and deep
  ensemble uncertainty for a second-stage run on the best few backbones.
- Check both `val_*` and `test_*` columns. The test split is useful for final
  comparison, but model selection should primarily look at validation behavior.
- If a large backbone repeatedly OOMs, reduce its batch size in that model's
  config instead of changing the global benchmark workflow.
