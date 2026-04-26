# Configurable Regression Losses

MatPropNet supports configurable single-task and per-task regression losses
through a composable `base loss + sample weight + asymmetry` stack.

## Single-task config

You can declare a top-level loss for single-task regression runs:

```yaml
loss:
  name: huber
  delta: 2.0
  weight:
    type: target_power
    alpha: 2.0
    gamma: 1.0
    ref: p95
    max_weight: 5.0
  asymmetry:
    enabled: true
    mode: threshold_underestimation
    threshold: 20.0
    factor: 2.0
```

If `weight` is omitted, MatPropNet defaults to `type: none`. If `asymmetry` is
omitted, it defaults to `enabled: false`.

## Multi-task config

For multi-task regression, configure loss per task under `task.tasks`:

```yaml
task:
  tasks:
    H:
      type: regression
      loss:
        name: huber
        delta: 2.0
        weight:
          type: target_power
          alpha: 2.0
          gamma: 1.0
          ref: p95
          max_weight: 5.0
    log_H:
      type: regression
      loss:
        name: mae
        weight:
          type: none
```

These per-task losses are computed before the existing multi-task
`loss_weighting` stage (`static`, `gradnorm`, `uncertainty`) is applied.

## Supported base regression losses

- `mae`
- `mse`
- `huber`
- `smooth_l1`
- `log_cosh`
- `gaussian_nll`

`gaussian_nll` enables a probabilistic regression head for the task. The model
predicts both `mu` and `log_var`; `mu` is kept in `task_preds`, while `log_var`
is returned in `task_log_vars` for loss computation and prediction export.

## Supported sample-weight strategies

- `none`
- `target_power`
- `threshold`
- `bin_balanced`

`target_power` and `bin_balanced` consume statistics computed from the training
split at trainer startup.

## Supported asymmetry modes

- `underestimation`
- `threshold_underestimation`

These modes multiply the sample weight when the prediction falls below the
target, optionally only above a configured target threshold.

## Runtime logging

At startup, MatPropNet logs the active loss configuration for each regression
task, including training-derived reference values and bin-balanced weights when
used.

During training and validation, the trainer logs:

- `loss/base_loss/<task>`
- `loss/weighted_loss/<task>`
- `loss/mean_weight/<task>`
- `loss/max_weight/<task>`
- `loss/min_weight/<task>`
- `loss/log_var_mean/<task>` when using `gaussian_nll`
- `loss/sigma_mean/<task>` when using `gaussian_nll`
