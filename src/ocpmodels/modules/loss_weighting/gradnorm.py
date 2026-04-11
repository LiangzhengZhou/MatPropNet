from __future__ import annotations

from typing import Any

import torch

from .base import LossWeightingStrategy


class GradNormLossWeighting(LossWeightingStrategy):
    """Engineering-oriented GradNorm implementation with separate weight updates."""

    def __init__(
        self,
        task_names: list[str],
        task_specs: dict[str, dict[str, Any]],
        alpha: float = 1.5,
        epsilon: float = 1.0e-8,
        clamp_min: float = 1.0e-3,
        clamp_max: float = 1.0e3,
        lr: float = 1.0e-3,
        update_every: int = 1,
        warmup_steps: int = 0,
    ) -> None:
        super().__init__(task_names, task_specs)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.lr = float(lr)
        self.update_every = max(int(update_every), 1)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.requires_post_backward = True

        self.logits = torch.nn.Parameter(
            torch.zeros(len(task_names), dtype=torch.float32)
        )
        self.register_buffer(
            "initial_losses", torch.full((len(task_names),), float("nan"))
        )
        self._optimizer: torch.optim.Optimizer | None = None
        self._cached_gradnorm_loss: torch.Tensor | None = None
        self._cached_active_names: list[str] = []
        self._cached_stats: dict[str, Any] = {}
        self._last_warning: str | None = None

    def bind_optimizer(self) -> None:
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam([self.logits], lr=self.lr)

    def _normalized_weights(self, active_idx: list[int]) -> torch.Tensor:
        active_logits = self.logits[active_idx]
        weights = torch.softmax(active_logits, dim=0) * max(len(active_idx), 1)
        return torch.clamp(weights, self.clamp_min, self.clamp_max)

    def compute_weighted_loss(
        self,
        task_losses: dict[str, torch.Tensor],
        step_context: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not task_losses:
            raise ValueError("GradNorm loss weighting received no active task losses.")

        self._cached_gradnorm_loss = None
        self._cached_active_names = []
        self._cached_stats = {}
        self._last_warning = None

        device = next(iter(task_losses.values())).device
        total_loss = torch.zeros((), device=device)
        active_names = list(task_losses.keys())
        active_idx = [self.task_names.index(name) for name in active_names]
        weights = self._normalized_weights(active_idx)
        detached_weights = weights.detach()

        stats: dict[str, Any] = {"loss_weighting/mode": "gradnorm"}
        raw_losses = []
        valid_idx = []
        for index, task_name in enumerate(active_names):
            raw_loss = task_losses[task_name]
            task_idx = active_idx[index]
            if torch.isnan(self.initial_losses[task_idx]):
                self.initial_losses[task_idx] = raw_loss.detach().float()

            weighted_loss = detached_weights[index] * raw_loss
            total_loss = total_loss + weighted_loss
            raw_losses.append(raw_loss.float())
            valid_idx.append(task_idx)

            stats[f"loss_weighting/raw_loss/{task_name}"] = float(
                raw_loss.detach().cpu()
            )
            stats[f"loss_weighting/weight/{task_name}"] = float(
                detached_weights[index].detach().cpu()
            )
            stats[f"loss_weighting/weighted_loss/{task_name}"] = float(
                weighted_loss.detach().cpu()
            )

        stats["loss_weighting/total_loss"] = float(total_loss.detach().cpu())

        should_update = (
            step_context.get("is_training", False)
            and step_context.get("global_step", 0) % self.update_every == 0
            and step_context.get("global_step", 0) > self.warmup_steps
            and len(active_names) > 1
            and step_context.get("shared_params")
        )
        if not should_update:
            return total_loss, stats

        grad_norms = []
        shared_params = step_context["shared_params"]
        grad_weights = weights

        for index, raw_loss in enumerate(raw_losses):
            grads = torch.autograd.grad(
                grad_weights[index] * raw_loss,
                shared_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            norms = [grad.norm(2) for grad in grads if grad is not None]
            if not norms:
                grad_norm = raw_loss.new_tensor(0.0)
            else:
                grad_norm = torch.norm(torch.stack(norms), p=1)
            grad_norms.append(grad_norm)

        grad_norms_tensor = torch.stack(grad_norms)
        mean_grad_norm = grad_norms_tensor.mean()
        initial_losses = self.initial_losses[valid_idx].to(device=device)
        relative_losses = torch.stack(raw_losses) / (initial_losses + self.epsilon)
        relative_rates = relative_losses / (
            relative_losses.mean().detach() + self.epsilon
        )
        target_grad = mean_grad_norm.detach() * (relative_rates ** self.alpha)
        gradnorm_loss = torch.abs(grad_norms_tensor - target_grad).sum()

        if not torch.isfinite(gradnorm_loss):
            self._last_warning = "gradnorm_loss_non_finite"
            stats["loss_weighting/warning"] = self._last_warning
            return total_loss, stats

        self._cached_gradnorm_loss = gradnorm_loss
        self._cached_active_names = active_names
        self._cached_stats = {
            "loss_weighting/gradnorm_loss": float(gradnorm_loss.detach().cpu()),
            "loss_weighting/mean_grad_norm": float(mean_grad_norm.detach().cpu()),
        }
        for index, task_name in enumerate(active_names):
            self._cached_stats[f"loss_weighting/grad_norm/{task_name}"] = float(
                grad_norms_tensor[index].detach().cpu()
            )
            self._cached_stats[f"loss_weighting/relative_rate/{task_name}"] = float(
                relative_rates[index].detach().cpu()
            )
            self._cached_stats[
                f"loss_weighting/target_grad_norm/{task_name}"
            ] = float(target_grad[index].detach().cpu())

        stats.update(self._cached_stats)
        return total_loss, stats

    def on_after_backward(self, step_context: dict[str, Any]) -> dict[str, Any]:
        if self._cached_gradnorm_loss is None:
            return (
                {"loss_weighting/warning": self._last_warning}
                if self._last_warning
                else {}
            )

        self.bind_optimizer()
        assert self._optimizer is not None
        self._optimizer.zero_grad(set_to_none=True)
        grad = torch.autograd.grad(
            self._cached_gradnorm_loss,
            self.logits,
            retain_graph=False,
            allow_unused=False,
        )[0]
        if grad is None or not torch.isfinite(grad).all():
            self._last_warning = "gradnorm_weight_grad_non_finite"
            return {"loss_weighting/warning": self._last_warning}

        self.logits.grad = grad
        self._optimizer.step()
        with torch.no_grad():
            self.logits.data = self.logits.data - self.logits.data.mean()
        return dict(self._cached_stats)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if self._optimizer is not None:
            state["_gradnorm_optimizer"] = self._optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        optimizer_state = state_dict.pop("_gradnorm_optimizer", None)
        result = super().load_state_dict(state_dict, strict=strict)
        if optimizer_state is not None:
            self.bind_optimizer()
            assert self._optimizer is not None
            self._optimizer.load_state_dict(optimizer_state)
        return result
