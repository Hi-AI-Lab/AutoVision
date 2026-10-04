# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Configuration subsystem: base class, validation gateway, registries."""

from autovision._configs.base import PydanticConfig
from autovision._configs.registry import Registry
from autovision._configs.validate import (
    assert_config_resolved,
    no_auto,
    prettify_validation_errors,
    pydantic_model_validate,
    suggest_key,
)

__all__ = [
    "PydanticConfig",
    "Registry",
    "assert_config_resolved",
    "no_auto",
    "prettify_validation_errors",
    "pydantic_model_validate",
    "suggest_key",
]
