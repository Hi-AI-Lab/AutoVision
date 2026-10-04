# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Model asset registry (filesystem + MLflow backends) + downloadable cache."""

from autovision._registry.downloadable import (
    DOWNLOADABLE_MODEL_URL_AND_HASH,
    DownloadableModel,
    DownloadableModelError,
    register_downloadable,
    resolve_downloadable,
)
from autovision._registry.mlflow_registry import MLflowModelRegistry
from autovision._registry.model_registry import (
    MODEL_STAGES,
    ModelRegistry,
    RegistryEntry,
    RegistryError,
)

__all__ = [
    "DOWNLOADABLE_MODEL_URL_AND_HASH",
    "DownloadableModel",
    "DownloadableModelError",
    "MLflowModelRegistry",
    "MODEL_STAGES",
    "ModelRegistry",
    "RegistryEntry",
    "RegistryError",
    "register_downloadable",
    "resolve_downloadable",
]
