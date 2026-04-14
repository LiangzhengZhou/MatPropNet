from __future__ import annotations

from ocpmodels.trainers.base_trainer import BaseTrainer


class DummyTrainer(BaseTrainer):
    def load_logger(self):
        self.logger = None

    def load_datasets(self):
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def load_task(self):
        pass

    def load_model(self):
        self.model = None

    def load_loss(self):
        pass

    def load_optimizer(self):
        self.optimizer = None

    def load_extras(self):
        self.scheduler = None
        self.clip_grad_norm = None
        self.ema = None

    def train(self):
        return None

    def _forward(self, batch_list):
        return {}

    def _compute_loss(self, out, batch_list):
        return 0


def test_base_trainer_preserves_top_level_runtime_config():
    extra_config = {
        "mode": "train",
        "checkpoint": "/tmp/checkpoint.pt",
        "loss_weighting": {"mode": "gradnorm", "alpha": 1.5},
        "hide_eval_progressbar": True,
    }

    trainer = DummyTrainer(
        task={"dataset": "property_lmdb"},
        model={"name": "property_model", "backbone": {"name": "cgcnn"}},
        dataset=None,
        optimizer={"lr_initial": 1.0e-3},
        identifier="",
        logger="tensorboard",
        cpu=True,
        name="property",
        extra_config=extra_config,
    )

    assert trainer.config["loss_weighting"]["mode"] == "gradnorm"
    assert trainer.config["loss_weighting"]["alpha"] == 1.5
    assert trainer.config["hide_eval_progressbar"] is True
    assert trainer.config["checkpoint"] == "/tmp/checkpoint.pt"
