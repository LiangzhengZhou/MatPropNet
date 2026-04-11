"""Config loading and path resolution helpers."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import yaml


def _merge_dicts(base: dict, update: dict):
    merged = copy.deepcopy(base)
    duplicates = []
    for key, value in update.items():
        if key not in merged:
            merged[key] = value
            continue
        if isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key], nested_duplicates = _merge_dicts(merged[key], value)
            duplicates.extend([f"{key}.{item}" for item in nested_duplicates])
        else:
            merged[key] = value
            duplicates.append(key)
    return merged, duplicates


def _parse_value(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _set_nested(target: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    current = target
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def apply_overrides(config: dict, overrides: list[str] | None):
    if not overrides:
        return config
    override_dict = {}
    for raw_arg in overrides:
        arg = raw_arg.lstrip("-")
        if "=" not in arg:
            raise ValueError(
                f"Override '{raw_arg}' must use key=value syntax."
            )
        key, raw_value = arg.split("=", 1)
        _set_nested(override_dict, key, _parse_value(raw_value))
    merged, _ = _merge_dicts(config, override_dict)
    return merged


def _load_config_recursive(path: Path, chain: list[Path] | None = None):
    chain = list(chain or [])
    path = path.resolve()
    if path in chain:
        raise ValueError(
            f"Cyclic config include detected: {[str(item) for item in chain + [path]]}"
        )
    chain.append(path)

    with path.open("r", encoding="utf-8") as handle:
        direct_config = yaml.safe_load(handle) or {}

    includes = direct_config.pop("includes", [])
    if not isinstance(includes, list):
        raise TypeError("'includes' must be a list of config paths.")

    config = {}
    duplicates_warning = []
    duplicates_error = []

    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            relative_to_config = path.parent / include_path
            include_path = (
                relative_to_config
                if relative_to_config.exists()
                else include_path.resolve()
            )
        include_cfg, include_warn, include_err = _load_config_recursive(
            include_path, chain
        )
        duplicates_warning.extend(include_warn)
        duplicates_error.extend(include_err)
        config, merge_dup_error = _merge_dicts(config, include_cfg)
        duplicates_error.extend(merge_dup_error)

    config, merge_dup_warning = _merge_dicts(config, direct_config)
    duplicates_warning.extend(merge_dup_warning)
    return config, duplicates_warning, duplicates_error


def _resolve_path(value, config_dir: Path):
    if value is None:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((config_dir / path).resolve())


def resolve_paths(config: dict, config_path: str | Path):
    config = copy.deepcopy(config)
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent

    def _resolve_dataset_entry(entry):
        if isinstance(entry, dict) and "src" in entry:
            entry = copy.deepcopy(entry)
            entry["src"] = _resolve_path(entry["src"], config_dir)
        return entry

    for key in ["dataset", "val_dataset", "test_dataset"]:
        if key not in config or config[key] is None:
            continue
        value = config[key]
        if isinstance(value, list):
            config[key] = [_resolve_dataset_entry(item) for item in value]
        else:
            config[key] = _resolve_dataset_entry(value)

    if "dataset" in config and isinstance(config["dataset"], dict):
        for split_key in ["train", "val", "test"]:
            if split_key not in config["dataset"] or config["dataset"][split_key] is None:
                continue
            split_value = config["dataset"][split_key]
            if isinstance(split_value, list):
                config["dataset"][split_key] = [
                    _resolve_dataset_entry(item) for item in split_value
                ]
            else:
                config["dataset"][split_key] = _resolve_dataset_entry(split_value)

    if config.get("checkpoint"):
        config["checkpoint"] = _resolve_path(config["checkpoint"], config_dir)
    if config.get("run_dir"):
        config["run_dir"] = _resolve_path(config["run_dir"], config_dir)

    backbone = config.get("model", {}).get("backbone", {})
    if isinstance(backbone, dict) and backbone.get("scale_file"):
        backbone["scale_file"] = _resolve_path(backbone["scale_file"], config_dir)

    config["config_path"] = str(config_path)
    config["config_dir"] = str(config_dir)
    return config


def validate_config(config: dict):
    required = ["trainer", "model", "task", "optim"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")
    config.setdefault("loss_weighting", {"mode": "static"})
    loss_weighting = config["loss_weighting"]
    if not isinstance(loss_weighting, dict):
        raise TypeError("'loss_weighting' must be a mapping when provided.")
    loss_weighting.setdefault("mode", "static")
    return config


def load_config(path: str | Path, overrides: list[str] | None = None):
    config, duplicates_warning, duplicates_error = _load_config_recursive(Path(path))
    if duplicates_error:
        raise ValueError(
            "Conflicting config keys across includes: "
            + ", ".join(duplicates_error)
        )
    config = apply_overrides(config, overrides)
    config = resolve_paths(config, path)
    config["config_duplicates_warning"] = duplicates_warning
    return validate_config(config)
