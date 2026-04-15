import pickle

from ocpmodels.trainers.base_trainer import _load_torch_checkpoint_compat


def test_load_torch_checkpoint_compat_falls_back_on_weights_only_error(monkeypatch):
    calls = []

    def fake_torch_load(path, map_location=None, **kwargs):
        calls.append({"path": path, "map_location": map_location, **kwargs})
        if len(calls) == 1:
            raise pickle.UnpicklingError(
                "Weights only load failed. Unsupported global: "
                "GLOBAL numpy.core.multiarray.scalar"
            )
        return {"state_dict": {"model.weight": 1}}

    monkeypatch.setattr("ocpmodels.trainers.base_trainer.torch.load", fake_torch_load)

    result = _load_torch_checkpoint_compat("dummy.pt", map_location="cpu")

    assert result == {"state_dict": {"model.weight": 1}}
    assert calls[0] == {"path": "dummy.pt", "map_location": "cpu"}
    assert calls[1] == {
        "path": "dummy.pt",
        "map_location": "cpu",
        "weights_only": False,
    }
