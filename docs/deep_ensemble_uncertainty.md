# Deep Ensemble Uncertainty

MatPropNet supports probabilistic regression heads for Deep Ensemble workflows.
For a regression task with Gaussian output, each ensemble member predicts:

```text
mu_m(x)
sigma_m^2(x)
```

The member distribution is:

```text
p(y | x, theta_m) = Normal(mu_m(x), sigma_m^2(x))
```

The ensemble predictive distribution is a mixture:

```text
p(y | x) = 1 / M * sum_m Normal(mu_m(x), sigma_m^2(x))
```

MatPropNet aggregates this into:

```text
mean = mean_m(mu_m)
var_epistemic = mean_m(mu_m^2) - mean^2
var_aleatoric = mean_m(sigma_m^2)
var_total = var_epistemic + var_aleatoric
```

For active learning, `std_epistemic` is usually the most useful uncertainty
source because it measures model disagreement. `std_aleatoric` measures
irreducible noise or label ambiguity, while `std_total` combines both.

## Training a probabilistic member

Use `gaussian_nll` as the task loss:

```yaml
task:
  tasks:
    H:
      type: regression
      loss:
        name: gaussian_nll
        min_log_var: -10.0
        max_log_var: 5.0
```

When this loss is used, MatPropNet automatically switches that regression head
to a Gaussian output head and exports:

- `pred_H`
- `pred_H_log_var`
- `pred_H_sigma`

`pred_H` and `pred_H_sigma` are exported in the original target units when a
target normalizer is configured. `pred_H_log_var` is the clamped log variance in
the normalized training space.

## Automated ensemble training

Use `matpropnet-ensemble-train` with an ensemble config:

```yaml
ensemble:
  name: gemnet_H_deep_ensemble
  base_config: /path/to/config_gemnet_H_gaussian.yml
  output_dir: /path/to/runs/gemnet_H_deep_ensemble

  num_members: 5
  seeds: [11, 23, 37, 51, 71]

  train:
    mode: sequential
    checkpoint_name: best_checkpoint.pt
    overwrite: false

  evaluate:
    splits:
      - val
      - test

  aggregate:
    tasks:
      - H
    include_members: true
```

Then run:

```bash
matpropnet-ensemble-train --config /path/to/ensemble.yml
```

This trains each member on the same preprocessed train/val/test LMDB splits but
with different training seeds. The output directory contains:

```text
ensemble_manifest.json
base_config.resolved.yml
members/member_000/config.resolved.yml
members/member_000/checkpoints/.../best_checkpoint.pt
members/member_000/ensemble_predictions/test.csv
aggregate/test_ensemble.csv
aggregate/ensemble_metrics.json
```

`ensemble_manifest.json` is the index for the ensemble. It records each member's
seed, run directory, resolved config, checkpoint, and prediction files. Keep it
with the run because `matpropnet-ensemble-predict` uses it later.

## Predicting a new LMDB

After training, run all members on a new LMDB and aggregate uncertainty:

```bash
matpropnet-ensemble-predict \
  --manifest /path/to/runs/gemnet_H_deep_ensemble/ensemble_manifest.json \
  --lmdb /path/to/new_pool/data.lmdb \
  --out-dir /path/to/new_pool_ensemble_predictions
```

The output directory contains per-member CSV files plus:

```text
ensemble_predictions.csv
prediction_manifest.json
ensemble_metrics.json  # only when targets exist
```

## Aggregating existing member CSVs

After training several members with different seeds, run prediction for each
checkpoint and aggregate the resulting CSV files:

```bash
matpropnet-ensemble-aggregate \
  --predictions member_0/test.csv member_1/test.csv member_2/test.csv \
  --out ensemble_predictions.csv \
  --tasks H
```

The output contains:

- `pred_H_mean`
- `pred_H_std_epistemic`
- `pred_H_std_aleatoric`
- `pred_H_std_total`
- per-member predictions when `--no-members` is not set

If member CSV files do not contain `pred_H_sigma`, aggregation still works but
`std_aleatoric` is set to zero. In that case the ensemble is a deterministic
ensemble and `std_epistemic` is only a model-disagreement proxy.
