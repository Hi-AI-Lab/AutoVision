# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Package protocols for model providers.

A package owns a set of backbone models: it lists names, builds raw modules,
and adapts them to the :class:`~autovision._models.model_wrapper.ModelWrapper`
protocol. Registration is in
:mod:`autovision._models.package_helpers`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

__all__ = ["BasePackage", "Package"]


class BasePackage(ABC):
    """A model provider that can export models it built."""

    name: str

    @classmethod
    @abstractmethod
    def export_model(
        cls, model: Any, out: Path, log_example: bool = True
    ) -> None:
        """Export a model built by this package to ``out``."""

    @classmethod
    @abstractmethod
    def is_supported_model(cls, model: Any) -> bool:
        """True if ``model`` (instance or name) belongs to this package."""


class Package(BasePackage):
    """A model provider with a listable set of ``"<package>/<model>"`` names."""

    @classmethod
    @abstractmethod
    def list_model_names(cls) -> list[str]:
        """All supported model names as ``"<package>/<model>"`` strings."""

    @classmethod
    @abstractmethod
    def get_model(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> Any:
        """Build the raw model module."""

    @classmethod
    @abstractmethod
    def get_model_wrapper(cls, model: Any) -> Any:
        """Wrap a model built by :meth:`get_model` into a ``ModelWrapper``."""
