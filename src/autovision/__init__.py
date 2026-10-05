# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Autovision: enterprise visual model training framework.

Public API:

- :func:`autovision.pretrain` — self-supervised pretraining / distillation.
- :func:`autovision.list_models` — all models exposed by registered packages.
- :func:`autovision.list_methods` — all registered pretraining methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from autovision import errors
from autovision._checkpoint import CHECKPOINT_AUTOVISION_KEY
from autovision._commands import (
    PretrainConfig,
    PretrainResult,
    TaskTrainResult,
    pretrain,
    train_image_classification,
    train_instance_segmentation,
    train_object_detection,
    train_oriented_object_detection,
    train_panoptic_segmentation,
    train_semantic_segmentation,
)
from autovision._commands.extract_video_frames import extract_video_frames
from autovision._compat import install_graphmodule_deepcopy_fix
from autovision._embedding import embed
from autovision._export import benchmark
from autovision._methods import list_methods
from autovision._models import list_models
from autovision._task_models.task_model_helpers import load_task_model
from autovision._version import __version__
from autovision.errors import AutovisionError

# Third-party defect shims must be installed before any model construction.
install_graphmodule_deepcopy_fix()

__all__ = [
    "AutovisionError",
    "PretrainConfig",
    "PretrainResult",
    "__version__",
    "errors",
    "list_methods",
    "list_models",
    "embed",
    "benchmark",
    "extract_video_frames",
    "load_embedding_model",
    "load_task_model",
    "pretrain",
    "train_image_classification",
    "train_instance_segmentation",
    "train_object_detection",
    "train_oriented_object_detection",
    "train_panoptic_segmentation",
    "train_semantic_segmentation",
]


def load_embedding_model(path: str | Path, device: str | None = None) -> Any:
    """Load an exported embedding model (``exported_models/*.pt``).

    Rebuilds the ``EmbeddingModel`` from the exported metadata. The backbone
    must be resolvable via ``model_name`` recorded in the export — the
    export stores the state dict and normalization args; the backbone name
    is part of the run config, so this loader accepts it explicitly.
    """
    from autovision._models import EmbeddingModel, get_wrapped_model
    from autovision._models._model_helpers import load_state_dict_compat

    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or "kind" not in raw:
        raise ValueError(f"File '{path}' is not an exported Autovision model.")
    model_name = raw.get("model_name")
    if model_name is None:
        raise ValueError(
            f"Export '{path}' does not record 'model_name'; pass the model "
            f"specification to rebuild the backbone."
        )
    wrapper = get_wrapped_model(model_name)
    embedding_model = EmbeddingModel(
        wrapped_model=wrapper, embed_dim=raw.get("embed_dim")
    )
    load_state_dict_compat(embedding_model, raw["state_dict"])
    if device is not None:
        embedding_model.to(device)
    embedding_model.eval()
    return embedding_model
