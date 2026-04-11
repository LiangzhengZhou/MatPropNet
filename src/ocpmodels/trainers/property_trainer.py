"""Generic trainer for single-task and multi-task materials property prediction."""

from __future__ import annotations

import copy
import csv
import json
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.data_parallel import (
    BalancedBatchSampler,
    OCPDataParallel,
    ParallelCollater,
)
from ocpmodels.common.registry import registry
from ocpmodels.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)
from ocpmodels.modules.loss_weighting import build_loss_weighting_strategy
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scheduler import LRScheduler
from ocpmodels.trainers.base_trainer import BaseTrainer


@registry.register_trainer("property")
class PropertyTrainer(BaseTrainer):
    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        normalizer=None,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        is_vis=False,
        is_hpo=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm={},
        loss=None,
    ):
        super().__init__(
            task=task,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            identifier=identifier,
            normalizer=normalizer,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            is_vis=is_vis,
            is_hpo=is_hpo,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            name="property",
            slurm=slurm,
            loss=loss,
        )

    def load_task(self):
        tasks_cfg = self.config["task"].get("tasks")
        if tasks_cfg:
            self.task_specs = OrderedDict(tasks_cfg.items())
        else:
            labels = self.config["task"].get("labels", ["target"])
            task_type = self.config["task"].get("type", "regression")
            self.task_specs = OrderedDict(
                (label, {"type": task_type, "weight": 1.0})
                for label in labels
            )
        self.task_names = list(self.task_specs.keys())
        self.task_name_to_idx = {
            name: idx for idx, name in enumerate(self.task_names)
        }
        self.num_targets = len(self.task_names)
        default_primary = (
            f"{self.task_names[0]}_mae"
            if self.task_specs[self.task_names[0]].get("type", "regression")
            == "regression"
            else f"{self.task_names[0]}_accuracy"
        )
        self.primary_metric = self.config["task"].get(
            "primary_metric", default_primary
        )
        self.primary_metric_mode = self.config["task"].get(
            "primary_metric_mode", "min"
        )
        self.loss_weighting = build_loss_weighting_strategy(
            self.config, self.task_specs
        ).to(self.device)

    def get_sampler(self, dataset, batch_size, shuffle):
        if self.config["optim"].get("disable_load_balancing", False):
            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
            return BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        return BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=distutils.get_world_size(),
            rank=distutils.get_rank(),
            device=self.device,
            mode=self.config["optim"].get("load_balancing", "atoms"),
            shuffle=shuffle,
            force_balancing=self.config["optim"].get(
                "force_balancing", False
            ),
        )

    def get_dataloader(self, dataset, sampler):
        return DataLoader(
            dataset,
            collate_fn=self.parallel_collater,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            batch_sampler=sampler,
        )

    def load_datasets(self):
        self.parallel_collater = ParallelCollater(
            0 if self.cpu else 1,
            self.config["model_attributes"]
            .get("backbone", {})
            .get("otf_graph", False),
        )
        self.train_loader = self.val_loader = self.test_loader = None
        dataset_name = self.config["task"].get("dataset", "property_lmdb")

        if self.config.get("dataset", None):
            self.train_dataset = registry.get_dataset_class(dataset_name)(
                self.config["dataset"]
            )
            self.train_sampler = self.get_sampler(
                self.train_dataset,
                self.config["optim"]["batch_size"],
                shuffle=True,
            )
            self.train_loader = self.get_dataloader(
                self.train_dataset, self.train_sampler
            )

        if self.config.get("val_dataset", None):
            self.val_dataset = registry.get_dataset_class(dataset_name)(
                self.config["val_dataset"]
            )
            self.val_sampler = self.get_sampler(
                self.val_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"]["batch_size"]
                ),
                shuffle=False,
            )
            self.val_loader = self.get_dataloader(
                self.val_dataset, self.val_sampler
            )

        if self.config.get("test_dataset", None):
            self.test_dataset = registry.get_dataset_class(dataset_name)(
                self.config["test_dataset"]
            )
            self.test_sampler = self.get_sampler(
                self.test_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"]["batch_size"]
                ),
                shuffle=False,
            )
            self.test_loader = self.get_dataloader(
                self.test_dataset, self.test_sampler
            )

        self.normalizers = {}
        mean = self.config["task"].get("target_mean")
        std = self.config["task"].get("target_std")
        if mean is None and self.config.get("dataset", None):
            mean = self.config["dataset"].get("target_mean")
            std = self.config["dataset"].get("target_std")
        if mean is not None and std is not None:
            self.normalizers["target"] = Normalizer(
                mean=mean, std=std, device=self.device
            )

    def load_model(self):
        if distutils.is_master():
            logging.info(f"Loading model: {self.config['model']}")

        bond_feat_dim = self.config["model_attributes"].get(
            "bond_feat_dim",
            self.config["model_attributes"]
            .get("backbone", {})
            .get("num_gaussians", 50),
        )
        model_attributes = copy.deepcopy(self.config["model_attributes"])
        model_attributes["tasks"] = copy.deepcopy(self.task_specs)
        self.model = registry.get_model_class(self.config["model"])(
            None,
            bond_feat_dim,
            self.num_targets,
            **model_attributes,
        ).to(self.device)

        finetune_cfg = self.config["task"].get("finetune", {})
        if finetune_cfg.get("freeze_backbone", False):
            self.model.freeze_backbone()
        if finetune_cfg.get("freeze_first_k_blocks", 0) > 0:
            self.model.freeze_first_k_blocks(
                int(finetune_cfg["freeze_first_k_blocks"])
            )

        if distutils.is_master():
            logging.info(
                "Loaded %s with %s parameters.",
                self.model.__class__.__name__,
                self.model.num_params,
            )

        if self.logger is not None:
            self.logger.watch(self.model)

        self.model = OCPDataParallel(
            self.model,
            output_device=self.device,
            num_gpus=1 if not self.cpu else 0,
        )
        if distutils.initialized():
            self.model = torch.nn.parallel.distributed.DistributedDataParallel(
                self.model, device_ids=[self.device]
            )

    def load_loss(self):
        self.loss_fns = {}
        for task_name, spec in self.task_specs.items():
            loss_name = spec.get("loss")
            if loss_name is None:
                loss_name = (
                    "cross_entropy"
                    if spec.get("type") == "classification"
                    else "mae"
                )
            loss_name = loss_name.lower()
            if loss_name in {"mae", "l1"}:
                self.loss_fns[task_name] = nn.L1Loss()
            elif loss_name == "mse":
                self.loss_fns[task_name] = nn.MSELoss()
            elif loss_name in {"smooth_l1", "huber"}:
                self.loss_fns[task_name] = nn.SmoothL1Loss()
            elif loss_name in {"cross_entropy", "ce"}:
                self.loss_fns[task_name] = nn.CrossEntropyLoss()
            elif loss_name in {"bce", "bce_with_logits"}:
                self.loss_fns[task_name] = nn.BCEWithLogitsLoss()
            else:
                raise NotImplementedError(
                    f"Unsupported loss '{loss_name}' for task '{task_name}'"
                )

    def load_optimizer(self):
        optimizer_cls = getattr(
            optim, self.config["optim"].get("optimizer", "AdamW")
        )
        optimizer_params = self.config["optim"].get("optimizer_params", {})
        weight_decay = self.config["optim"].get("weight_decay", 0.0)
        param_groups_cfg = self.config["optim"].get("param_groups", [])

        if param_groups_cfg:
            available = self.model.module.get_param_groups()
            used = set()
            param_groups = []
            for group_cfg in param_groups_cfg:
                params = [
                    p
                    for p in available.get(group_cfg["name"], [])
                    if p.requires_grad
                ]
                if not params:
                    continue
                used.add(group_cfg["name"])
                param_groups.append(
                    {
                        "params": params,
                        "lr": group_cfg.get(
                            "lr", self.config["optim"]["lr_initial"]
                        ),
                        "weight_decay": group_cfg.get(
                            "weight_decay", weight_decay
                        ),
                    }
                )
            remaining = []
            for name, params in available.items():
                if name in used:
                    continue
                remaining.extend([p for p in params if p.requires_grad])
            if remaining:
                param_groups.append(
                    {
                        "params": remaining,
                        "lr": self.config["optim"]["lr_initial"],
                        "weight_decay": weight_decay,
                    }
                )
            param_groups.extend(self.loss_weighting.extra_optimizer_param_groups())
            self.optimizer = optimizer_cls(param_groups, **optimizer_params)
            return

        params = [p for p in self.model.parameters() if p.requires_grad]
        param_groups = [
            {
                "params": params,
                "lr": self.config["optim"]["lr_initial"],
                "weight_decay": weight_decay,
            }
        ]
        param_groups.extend(self.loss_weighting.extra_optimizer_param_groups())
        self.optimizer = optimizer_cls(param_groups, **optimizer_params)

    def load_extras(self):
        self.scheduler = LRScheduler(self.optimizer, self.config["optim"])
        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm")
        self.ema_decay = self.config["optim"].get("ema_decay")
        self.ema = (
            ExponentialMovingAverage(self.model.parameters(), self.ema_decay)
            if self.ema_decay
            else None
        )

    def _split_batch(self, batch_list):
        return batch_list[0]

    def _normalize_targets(self, targets):
        if "target" not in self.normalizers:
            return targets
        normalized = targets.clone()
        for task_name, idx in self.task_name_to_idx.items():
            if self.task_specs[task_name].get("type", "regression") != "regression":
                continue
            mean = self.normalizers["target"].mean[idx]
            std = self.normalizers["target"].std[idx]
            normalized[:, idx] = (normalized[:, idx] - mean) / std
        return normalized

    def _denormalize_prediction(self, task_name, prediction):
        if "target" not in self.normalizers:
            return prediction
        idx = self.task_name_to_idx[task_name]
        if self.task_specs[task_name].get("type", "regression") != "regression":
            return prediction
        mean = self.normalizers["target"].mean[idx].to(prediction.device)
        std = self.normalizers["target"].std[idx].to(prediction.device)
        return prediction * std + mean

    def _forward(self, batch_list):
        return self.model(batch_list)

    def _forward_with_optional_latent(self, batch_list, return_latent=False):
        if not return_latent:
            return self._forward(batch_list)
        model = self._unwrap_model()
        batch = self._split_batch(batch_list).to(self.device)
        return model(batch, return_latent=True)

    def _unwrap_model(self):
        model = self.model
        while hasattr(model, "module"):
            model = model.module
        return model

    def _get_loss_weighting_shared_params(self):
        scope = (
            self.config.get("loss_weighting", {}).get(
                "shared_parameter_scope", "backbone_last_block"
            )
            if self.loss_weighting is not None
            else "backbone_last_block"
        )
        model = self._unwrap_model()
        backbone = getattr(model, "backbone", None)
        latent = getattr(model, "latent", None)
        latent_projector = getattr(model, "latent_projector", None)

        if scope == "backbone":
            params = [
                param
                for param in getattr(backbone, "parameters", lambda: [])()
                if param.requires_grad
            ]
            if params:
                return params
        if scope == "latent_projector":
            params = [
                param
                for param in getattr(latent_projector, "parameters", lambda: [])()
                if param.requires_grad
            ]
            if params:
                return params
        if scope == "backbone_last_block" and backbone is not None:
            blocks = getattr(backbone, "blocks", None)
            if blocks is not None and len(blocks) > 0:
                params = [param for param in blocks[-1].parameters() if param.requires_grad]
                if params:
                    return params
        for module in (latent_projector, latent, backbone):
            if module is None:
                continue
            params = [param for param in module.parameters() if param.requires_grad]
            if params:
                return params
        return [param for param in model.parameters() if param.requires_grad]

    def _build_loss_weighting_context(self, is_training: bool):
        return {
            "global_step": self.step,
            "epoch": self.epoch,
            "is_training": is_training,
            "amp_enabled": self.scaler is not None,
            "shared_params": self._get_loss_weighting_shared_params()
            if is_training
            else [],
        }

    def _compute_task_losses(self, out, batch_list):
        batch = self._split_batch(batch_list).to(self.device)
        preds = out["task_preds"]
        targets = batch.y.to(self.device).view(-1, self.num_targets)
        targets_normed = self._normalize_targets(targets)
        target_mask = getattr(batch, "target_mask", torch.ones_like(targets)).to(
            self.device
        )
        target_mask = target_mask.view(-1, self.num_targets).bool()

        task_losses = {}
        for task_name, spec in self.task_specs.items():
            idx = self.task_name_to_idx[task_name]
            valid_mask = target_mask[:, idx]
            if valid_mask.sum() == 0:
                continue
            pred = preds[task_name][valid_mask]
            target = targets_normed[:, idx][valid_mask]
            if spec.get("type", "regression") == "classification":
                if pred.ndim == 1 or pred.shape[-1] == 1:
                    target = target.float()
                else:
                    target = target.long()
            task_losses[task_name] = self.loss_fns[task_name](pred, target)
        return task_losses

    def _compute_loss(self, out, batch_list, is_training=True):
        task_losses = self._compute_task_losses(out, batch_list)
        total_loss, loss_weighting_stats = self.loss_weighting.compute_weighted_loss(
            task_losses,
            self._build_loss_weighting_context(is_training=is_training),
        )
        out["per_task_loss"] = {
            task_name: loss_value.detach()
            for task_name, loss_value in task_losses.items()
        }
        out["loss_weighting_stats"] = loss_weighting_stats
        return total_loss

    def _update_metric(self, metrics, name, total, numel):
        if isinstance(total, dict):
            if name not in metrics:
                metrics[name] = {"metric": 0.0}
            for key, value in total.items():
                if key == "metric":
                    continue
                metrics[name][key] = metrics[name].get(key, 0.0) + float(value)
            metrics[name]["metric"] = float(total.get("metric", 0.0))
            return metrics
        if name not in metrics:
            metrics[name] = {"total": 0.0, "numel": 0.0, "metric": 0.0}
        metrics[name]["total"] += float(total)
        metrics[name]["numel"] += float(numel)
        metrics[name]["metric"] = (
            metrics[name]["total"] / max(metrics[name]["numel"], 1.0)
        )
        return metrics

    def _compute_metrics(self, out, batch_list, evaluator=None, metrics=None):
        del evaluator
        metrics = {} if metrics is None else metrics
        batch = self._split_batch(batch_list).to(self.device)
        preds = out["task_preds"]
        targets = batch.y.to(self.device).view(-1, self.num_targets)
        target_mask = getattr(batch, "target_mask", torch.ones_like(targets)).to(
            self.device
        )
        target_mask = target_mask.view(-1, self.num_targets).bool()

        for task_name, spec in self.task_specs.items():
            idx = self.task_name_to_idx[task_name]
            valid_mask = target_mask[:, idx]
            if valid_mask.sum() == 0:
                continue
            pred = preds[task_name][valid_mask]
            target = targets[:, idx][valid_mask]
            if spec.get("type", "regression") == "classification":
                if pred.ndim > 1 and pred.shape[-1] > 1:
                    predicted_class = pred.argmax(dim=-1)
                else:
                    predicted_class = (torch.sigmoid(pred.view(-1)) >= 0.5).long()
                acc = (predicted_class == target.long()).float().sum().item()
                metrics = self._update_metric(
                    metrics, f"{task_name}_accuracy", acc, valid_mask.sum().item()
                )
            else:
                denorm_pred = self._denormalize_prediction(task_name, pred)
                mae_total = torch.abs(denorm_pred - target).sum().item()
                mse_total = torch.square(denorm_pred - target).sum().item()
                ss_res_total = torch.square(denorm_pred - target).sum().item()
                sum_y = target.sum().item()
                sum_y2 = torch.square(target).sum().item()
                count = valid_mask.sum().item()
                metrics = self._update_metric(
                    metrics, f"{task_name}_mae", mae_total, count
                )
                metrics = self._update_metric(
                    metrics, f"{task_name}_mse", mse_total, count
                )
                metrics[f"{task_name}_rmse"] = {
                    "total": metrics[f"{task_name}_mse"]["total"],
                    "numel": metrics[f"{task_name}_mse"]["numel"],
                    "metric": np.sqrt(
                        metrics[f"{task_name}_mse"]["total"]
                        / max(metrics[f"{task_name}_mse"]["numel"], 1.0)
                    ),
                }
                metrics = self._update_metric(
                    metrics,
                    f"{task_name}_r2",
                    {
                        "ss_res": ss_res_total,
                        "sum_y": sum_y,
                        "sum_y2": sum_y2,
                        "count": count,
                        "metric": 0.0,
                    },
                    count,
                )

        for task_name, loss_value in out.get("per_task_loss", {}).items():
            metrics = self._update_metric(
                metrics, f"{task_name}_loss", float(loss_value.item()), 1
            )
        for key, value in out.get("loss_weighting_stats", {}).items():
            if isinstance(value, (int, float)):
                metrics = self._update_metric(metrics, key, float(value), 1)
        return metrics

    def _aggregate_metrics(self, metrics):
        aggregated = {}
        for key, value in metrics.items():
            if key.endswith("_r2"):
                ss_res = distutils.all_reduce(
                    value["ss_res"], average=False, device=self.device
                )
                sum_y = distutils.all_reduce(
                    value["sum_y"], average=False, device=self.device
                )
                sum_y2 = distutils.all_reduce(
                    value["sum_y2"], average=False, device=self.device
                )
                count = distutils.all_reduce(
                    value["count"], average=False, device=self.device
                )
                ss_tot = max(sum_y2 - (sum_y * sum_y) / max(count, 1.0), 0.0)
                if count <= 1 or ss_tot <= 1e-12:
                    metric = 0.0
                else:
                    metric = 1.0 - (ss_res / ss_tot)
                aggregated[key] = {
                    "ss_res": ss_res,
                    "sum_y": sum_y,
                    "sum_y2": sum_y2,
                    "count": count,
                    "metric": metric,
                }
                continue
            total = distutils.all_reduce(
                value["total"], average=False, device=self.device
            )
            numel = distutils.all_reduce(
                value["numel"], average=False, device=self.device
            )
            aggregated[key] = {
                "total": total,
                "numel": numel,
                "metric": total / max(numel, 1.0),
            }
        for key in list(aggregated.keys()):
            if key.endswith("_rmse"):
                mse_key = key.replace("_rmse", "_mse")
                aggregated[key]["metric"] = np.sqrt(aggregated[mse_key]["metric"])
        return aggregated

    def train(self, disable_eval_tqdm=False):
        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        patience = self.config["optim"].get("early_stopping_patience")
        best_primary = np.inf if self.primary_metric_mode == "min" else -np.inf
        epochs_without_improvement = 0
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            if hasattr(self.train_sampler, "set_epoch"):
                self.train_sampler.set_epoch(epoch_int)
            train_loader_iter = iter(self.train_loader)

            for i in range(len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()
                batch = next(train_loader_iter)
                self.optimizer.zero_grad(set_to_none=True)
                retain_graph = self.loss_weighting.requires_post_backward

                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch, is_training=True)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward(retain_graph=retain_graph)
                    loss_weighting_log = self.loss_weighting.on_after_backward(
                        self._build_loss_weighting_context(is_training=True)
                    )
                    if self.clip_grad_norm:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward(retain_graph=retain_graph)
                    loss_weighting_log = self.loss_weighting.on_after_backward(
                        self._build_loss_weighting_context(is_training=True)
                    )
                    if self.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad_norm
                        )
                    self.optimizer.step()
                loss_weighting_log.update(
                    self.loss_weighting.on_after_optimizer_step(
                        self._build_loss_weighting_context(is_training=True)
                    )
                )
                if self.ema:
                    self.ema.update()

                metrics = self._compute_metrics(out, batch, metrics={})
                metrics = self._update_metric(
                    metrics, "loss", loss.detach().item(), 1
                )
                log_dict = {k: v["metric"] for k, v in metrics.items()}
                for key, value in out.get("loss_weighting_stats", {}).items():
                    if isinstance(value, (int, float)):
                        log_dict[key] = float(value)
                for key, value in loss_weighting_log.items():
                    if isinstance(value, (int, float)):
                        log_dict[key] = float(value)
                log_dict["epoch"] = self.epoch
                log_dict["lr"] = self.scheduler.get_lr()
                if self.logger is not None:
                    self.logger.log(log_dict, step=self.step, split="train")
                if distutils.is_master() and self.step % self.config["cmd"]["print_every"] == 0:
                    logging.info(
                        ", ".join(
                            [f"{key}: {value:.4f}" for key, value in log_dict.items()]
                        )
                    )

                if self.scheduler.scheduler_type != "ReduceLROnPlateau":
                    self.scheduler.step()

                if self.val_loader is not None and self.step % eval_every == 0:
                    val_metrics = self.validate(
                        split="val", disable_tqdm=disable_eval_tqdm
                    )
                    primary_value = val_metrics[self.primary_metric]["metric"]
                    improved = (
                        primary_value < best_primary
                        if self.primary_metric_mode == "min"
                        else primary_value > best_primary
                    )
                    if improved:
                        best_primary = primary_value
                        epochs_without_improvement = 0
                        self.save(
                            metrics=val_metrics,
                            checkpoint_file="best_checkpoint.pt",
                            training_state=False,
                        )
                        if self.test_loader is not None:
                            self.export_predictions(split="test")
                    else:
                        epochs_without_improvement += 1

                    self.save(metrics=val_metrics)
                    if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                        self.scheduler.step(primary_value)

                    if patience is not None and epochs_without_improvement >= patience:
                        logging.info("Early stopping triggered.")
                        return

    @torch.no_grad()
    def validate(self, split="val", disable_tqdm=False):
        if distutils.is_master():
            logging.info("Evaluating on %s.", split)
        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        loader = self.val_loader if split == "val" else self.test_loader
        metrics = {}
        rank = distutils.get_rank()
        for _, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc=f"device {rank}",
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)
                loss = self._compute_loss(out, batch, is_training=False)
            metrics = self._compute_metrics(out, batch, metrics=metrics)
            metrics = self._update_metric(
                metrics, "loss", loss.detach().item(), 1
            )

        metrics = self._aggregate_metrics(metrics)
        if distutils.is_master():
            logging.info(
                ", ".join(
                    [
                        f"{key}: {value['metric']:.4f}"
                        for key, value in metrics.items()
                    ]
                )
            )
        if self.logger is not None:
            self.logger.log(
                {k: v["metric"] for k, v in metrics.items()},
                step=self.step,
                split=split,
            )

        if self.ema:
            self.ema.restore()
        return metrics

    def _extract_sample_ids(self, batch):
        if hasattr(batch, "sample_id"):
            sample_id = batch.sample_id
            if isinstance(sample_id, str):
                return [sample_id]
            return [str(item) for item in sample_id]
        if hasattr(batch, "sid"):
            sid = batch.sid
            if torch.is_tensor(sid):
                return [str(item) for item in sid.tolist()]
            return [str(item) for item in sid]
        return [str(i) for i in range(batch.y.shape[0])]

    @torch.no_grad()
    def predict(
        self,
        data_loader,
        per_image=True,
        results_file=None,
        disable_tqdm=False,
    ):
        del per_image
        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on property dataset.")

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        predict_cfg = self.config.get("task", {}).get("predict", {})
        include_z = predict_cfg.get("export_latent", False)
        include_graph_emb = predict_cfg.get("export_graph_emb", False)
        include_node_emb = predict_cfg.get("export_node_emb", False)
        return_latent = include_z or include_graph_emb or include_node_emb

        predictions = {"id": []}
        for task_name in self.task_names:
            predictions[f"pred_{task_name}"] = []
            predictions[f"target_{task_name}"] = []
        if include_z:
            predictions["z"] = []
        if include_graph_emb:
            predictions["graph_emb"] = []
        if include_node_emb:
            predictions["node_emb"] = []

        rank = distutils.get_rank()
        for _, batch_list in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc=f"device {rank}",
            disable=disable_tqdm,
        ):
            batch = batch_list[0]
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward_with_optional_latent(
                    batch_list, return_latent=return_latent
                )
            preds = out["task_preds"]
            targets = batch.y.view(-1, self.num_targets)
            ids = self._extract_sample_ids(batch)
            predictions["id"].extend(ids)

            for task_name in self.task_names:
                idx = self.task_name_to_idx[task_name]
                pred = preds[task_name].detach().cpu()
                if self.task_specs[task_name].get("type", "regression") == "regression":
                    pred = self._denormalize_prediction(task_name, pred).cpu()
                elif pred.ndim > 1 and pred.shape[-1] > 1:
                    pred = pred.argmax(dim=-1)
                predictions[f"pred_{task_name}"].extend(pred.tolist())
                predictions[f"target_{task_name}"].extend(
                    targets[:, idx].detach().cpu().tolist()
                )
            if include_z:
                predictions["z"].extend(
                    json.dumps(item)
                    for item in out["z"].detach().cpu().tolist()
                )
            if include_graph_emb:
                predictions["graph_emb"].extend(
                    json.dumps(item)
                    for item in out["graph_emb"].detach().cpu().tolist()
                )
            if include_node_emb:
                node_emb = out["node_emb"].detach().cpu()
                batch_index = batch.batch.detach().cpu()
                for graph_idx in range(len(ids)):
                    graph_nodes = node_emb[batch_index == graph_idx].tolist()
                    predictions["node_emb"].append(json.dumps(graph_nodes))

        if self.ema:
            self.ema.restore()

        if results_file is not None:
            keys = [key for key in predictions if key != "id"]
            self.save_results(predictions, results_file, keys=keys)
        return predictions

    def export_predictions(self, split=None):
        splits = (
            [split]
            if split is not None
            else self.config["task"].get(
                "export_splits", ["train", "val", "test"]
            )
        )
        export_dir = self.config["task"].get(
            "export_predictions_dir",
            os.path.join(self.config["cmd"]["results_dir"], "property_predictions"),
        )
        loader_map = {
            "train": self.train_loader,
            "val": self.val_loader,
            "test": self.test_loader,
        }
        for split_name in splits:
            loader = loader_map.get(split_name)
            if loader is None:
                continue
            rows = self.predict(
                loader,
                results_file=None,
                disable_tqdm=self.config.get("hide_eval_progressbar", False),
            )
            out_path = os.path.join(export_dir, f"{split_name}.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fieldnames = ["sample_id"]
            for task_name in self.task_names:
                fieldnames.extend([f"{task_name}_target", f"{task_name}_pred"])
            with open(out_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for idx, sample_id in enumerate(rows["id"]):
                    row = {"sample_id": sample_id}
                    for task_name in self.task_names:
                        row[f"{task_name}_target"] = rows[f"target_{task_name}"][idx]
                        row[f"{task_name}_pred"] = rows[f"pred_{task_name}"][idx]
                    writer.writerow(row)
            logging.info("Wrote %s predictions to %s", split_name, out_path)
