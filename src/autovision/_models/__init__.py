# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Model layer: wrapper protocols, package registry, embedding model."""

from autovision._models import (
    dinov2,  # noqa: F401  (registers the package)
    dinov3,  # noqa: F401  (registers the package)
    ecvit,  # noqa: F401  (registers the package)
    rfdetr,  # noqa: F401  (registers the package)
    super_gradients,  # noqa: F401  (registers the package)
    timm,  # noqa: F401  (registers the package)
    torchvision,  # noqa: F401  (registers the package)
    ultralytics,  # noqa: F401  (registers the package)
)
from autovision._models.embedding_model import EmbeddingModel
from autovision._models.model_wrapper import (
    ArchitectureInfo,
    ArchitectureInfoGettable,
    ModelWrapper,
    missing_model_wrapper_attrs,
)
from autovision._models.package_helpers import (
    get_package,
    get_package_from_model,
    get_wrapped_model,
    list_model_names,
    parse_model_name,
    register_package,
)

__all__ = [
    "ArchitectureInfo",
    "ArchitectureInfoGettable",
    "EmbeddingModel",
    "ModelWrapper",
    "get_package",
    "get_package_from_model",
    "get_wrapped_model",
    "list_model_names",
    "missing_model_wrapper_attrs",
    "parse_model_name",
    "register_package",
]

def list_models() -> list[str]:
    """Public alias for :func:`list_model_names`."""
    return list_model_names()
