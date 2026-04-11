from pathlib import Path

from matpropnet.config import load_config


def test_load_config_resolves_relative_paths(tmp_path):
    config_dir = tmp_path / "configs"
    data_dir = tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()

    (config_dir / "base.yml").write_text(
        "trainer: property\n"
        "model:\n"
        "  name: property_model\n"
        "task:\n"
        "  dataset: property_lmdb\n"
        "optim:\n"
        "  batch_size: 1\n"
        "dataset:\n"
        "  src: ../data/train.lmdb\n",
        encoding="utf-8",
    )
    (config_dir / "child.yml").write_text(
        "includes:\n"
        "  - base.yml\n"
        "val_dataset:\n"
        "  src: ../data/val.lmdb\n"
        "run_dir: ../runs\n",
        encoding="utf-8",
    )

    cfg = load_config(config_dir / "child.yml")
    assert cfg["dataset"]["src"] == str((data_dir / "train.lmdb").resolve())
    assert cfg["val_dataset"]["src"] == str((data_dir / "val.lmdb").resolve())
    assert cfg["run_dir"] == str((tmp_path / "runs").resolve())


def test_load_config_applies_overrides(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "trainer: property\n"
        "model:\n"
        "  name: property_model\n"
        "task:\n"
        "  dataset: property_lmdb\n"
        "optim:\n"
        "  batch_size: 4\n",
        encoding="utf-8",
    )
    cfg = load_config(config_path, overrides=["--optim.batch_size=16"])
    assert cfg["optim"]["batch_size"] == 16
