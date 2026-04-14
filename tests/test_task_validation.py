import pytest

from ocpmodels.tasks.task import PredictTask, ValidateTask


class DummyTrainer:
    def __init__(self, *, val_loader=None, test_loader=None):
        self.val_loader = val_loader
        self.test_loader = test_loader

    def predict(self, *args, **kwargs):
        return None

    def validate(self, *args, **kwargs):
        return None


def test_predict_task_requires_test_loader_and_checkpoint():
    task = PredictTask({"checkpoint": None, "hide_eval_progressbar": False})
    task.trainer = DummyTrainer(test_loader=None)

    with pytest.raises(ValueError):
        task.run()


def test_validate_task_requires_val_loader_and_checkpoint():
    task = ValidateTask({"checkpoint": None, "hide_eval_progressbar": False})
    task.trainer = DummyTrainer(val_loader=None)

    with pytest.raises(ValueError):
        task.run()
