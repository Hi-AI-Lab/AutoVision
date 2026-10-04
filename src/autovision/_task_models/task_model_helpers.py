# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""load_model: rebuild a TaskModel from an exported model file."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Literal

import torch

from autovision._env import Env  # noqa: F401  (cache dir contract)
from autovision.errors import CheckpointError
from autovision._accelerator import npu_available
from autovision.types import PathLike

__all__ = ["load_model"]

logger = logging.getLogger("autovision")

# Class-path migration table: old exports keep loading after refactors.
_CLASS_PATH_MIGRATIONS: dict[str, str] = {}


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if npu_available():
        return torch.device("npu")
    return torch.device("cpu")


def load_task_model(
    model: PathLike, device: str | torch.device | None = None
) -> Any:
    """Load an exported task model (``exported_models/*.pt``).

    The exported file carries ``model_class_path`` + ``model_init_args`` +
    ``train_model`` (EMA-prefixed state dict); the class is imported
    dynamically and the state dict loaded through the model's loader
    (EMA prefix stripped by the TaskModel).
    """
    device = _resolve_device(device)
    path = Path(model)
    if not path.exists():
        raise CheckpointError(f"Exported model does not exist: '{path}'")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise CheckpointError(
            f"Failed to load exported model '{path}': {error}"
        ) from error
    if not isinstance(checkpoint, dict) or "model_class_path" not in checkpoint:
        raise CheckpointError(
            f"File '{path}' is not an exported Autovision task model."
        )
    class_path: str = _CLASS_MIGRATIONS.get(
        checkpoint["model_class_path"], checkpoint["model_class_path"]
    )
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
    except (ImportError, AttributeError) as error:
        raise CheckpointError(
            f"Cannot import '{class_path}' from the installed autovision "
            f"version: {error}"
        ) from error

    init_args = dict(checkpoint.get("model_init_args", {}))
    init_args.pop("load_weights", None)
    model = model_class(**init_args)
    state_dict = checkpoint["train_model"]
    if hasattr(model, "load_train_state_dict"):
        model.load_train_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


_CLASS_MIGRATIONS = _CLASS_PATH_MIGRATIONS
