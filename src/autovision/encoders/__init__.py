# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Model layer: wrapper protocols, package registry, embedding model."""

from autovision.encoders import (
    dinov2,  # noqa: F401  (registers the package)
    dinov3,  # noqa: F401  (registers the package)
    ecvit,  # noqa: F401  (registers the package)
    rfdetr,  # noqa: F401  (registers the package)
    super_gradients,  # noqa: F401  (registers the package)
    timm,  # noqa: F401  (registers the package)
    torchvision,  # noqa: F401  (registers the package)
    ultralytics,  # noqa: F401  (registers the package)
)
from autovision.encoders.embedding_model import FeatureExtractor
from autovision.encoders.model_wrapper import (
    ArchitectureInfo,
    ArchitectureInfoGettable,
    EncoderProtocol,
    missing_model_wrapper_attrs,
)
from autovision.encoders.package_helpers import (
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
    "FeatureExtractor",
    "EncoderProtocol",
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
