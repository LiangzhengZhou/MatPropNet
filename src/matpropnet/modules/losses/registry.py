from __future__ import annotations

from collections.abc import Callable


BASE_LOSS_REGISTRY: dict[str, Callable] = {}
WEIGHT_REGISTRY: dict[str, Callable] = {}
ASYMMETRY_REGISTRY: dict[str, Callable] = {}


def register_base_loss(name: str):
    def decorator(obj):
        BASE_LOSS_REGISTRY[name] = obj
        return obj

    return decorator


def register_weight(name: str):
    def decorator(obj):
        WEIGHT_REGISTRY[name] = obj
        return obj

    return decorator


def register_asymmetry(name: str):
    def decorator(obj):
        ASYMMETRY_REGISTRY[name] = obj
        return obj

    return decorator

