# Multi-Task Loss Weighting

MatPropNet supports three loss weighting modes under one training stack:

- `static`
- `gradnorm`
- `uncertainty`

Configure them with:

```yaml
loss_weighting:
  mode: static
```

## When To Use Each Mode

- `static`: best default when task scales are already aligned or you want fully deterministic behavior.
- `gradnorm`: best when one task learns much slower than the others and you want the shared backbone gradients to stay balanced.
- `uncertainty`: best when task noise levels differ and you want the model to learn a softer weighting automatically.

## Paper Mapping

### GradNorm

Implementation source:

- *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*

Mapped concepts:

- `sum_i w_i * L_i` -> `GradNormLossWeighting.compute_weighted_loss`
- `L_i(t) / L_i(0)` -> `initial_losses` buffer plus current batch loss ratio
- relative training rate `r_i(t)` -> `loss_weighting/relative_rate/<task>`
- target gradient norm `G*_i` -> `loss_weighting/target_grad_norm/<task>`
- GradNorm auxiliary objective -> `loss_weighting/gradnorm_loss`

MatPropNet uses an engineering-oriented implementation that measures gradient norms on the configured shared parameter scope. The default is `backbone_last_block`, which keeps the implementation close to the shared representation layer while limiting overhead.

### Uncertainty Weighting

Implementation source:

- homoscedastic uncertainty weighting formulation for multi-task learning

Mapped concepts:

- regression loss term `0.5 * exp(-s_i) * L_i + 0.5 * s_i` -> `UncertaintyLossWeighting.compute_weighted_loss`
- learned `log variance` parameter `s_i` -> `log_vars[task_name]`
- implied uncertainty `sigma_i` -> `loss_weighting/sigma/<task>`
- effective weight `exp(-s_i)` (or `0.5 * exp(-s_i)` for regression) -> `loss_weighting/weight/<task>`

## Logging Fields

All modes log:

- `loss_weighting/raw_loss/<task>`
- `loss_weighting/weighted_loss/<task>`
- `loss_weighting/weight/<task>`
- `loss_weighting/total_loss`

GradNorm also logs:

- `loss_weighting/grad_norm/<task>`
- `loss_weighting/target_grad_norm/<task>`
- `loss_weighting/relative_rate/<task>`
- `loss_weighting/mean_grad_norm`
- `loss_weighting/gradnorm_loss`

Uncertainty also logs:

- `loss_weighting/log_var/<task>`
- `loss_weighting/sigma/<task>`

## Stability Notes

- `static` is the safest fallback mode.
- `gradnorm` skips its auxiliary update when fewer than two tasks are active in a batch.
- `uncertainty` clamps each `log_var` to avoid runaway values.
- all adaptive modes use epsilon guards to reduce divide-by-zero risk.
- if a non-finite GradNorm update is detected, MatPropNet logs a warning and skips that auxiliary update instead of crashing the whole run.

## Common Troubleshooting

- If GradNorm weights oscillate strongly, lower `loss_weighting.lr` or increase `warmup_steps`.
- If uncertainty `log_var` values drift to the clamp boundary, reduce the uncertainty learning rate or standardize targets.
- If one task still dominates, inspect the raw per-task loss curves before changing the weighting mode.
