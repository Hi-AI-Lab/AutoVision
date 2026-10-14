# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Package registry: name parsing, dispatch, and model instantiation.

Dispatch order matters: packages are checked in registration order and the
custom package (accepting raw ``EncoderProtocol`` instances) must always be
last — custom is the last resort.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from autovision.encoders.model_wrapper import (
    EncoderProtocol,
    missing_model_wrapper_attrs,
)
from autovision.encoders.package import BasePackage, Package
from autovision.errors import UnknownModelError

__all__ = [
    "CUSTOM_PACKAGE",
    "BasePackage",
    "Package",
    "custom_package",
    "get_package",
    "get_package_from_model",
    "get_wrapped_model",
    "list_base_packages",
    "list_model_names",
    "list_packages",
    "parse_model_name",
    "register_package",
]

_BASE_PACKAGES: list[BasePackage] = []
_CUSTOM_PACKAGE_LIST: list[BasePackage] = []


def register_package(package: BasePackage) -> None:
    """Register a model package (built-ins at import time, plugins at runtime)."""
    if any(p.name == package.name for p in _BASE_PACKAGES):
        raise ValueError(f"Package '{package.name}' is already registered.")
    _BASE_PACKAGES.append(package)


def list_base_packages() -> list[BasePackage]:
    """All registered packages, custom always last."""
    return _BASE_PACKAGES + _CUSTOM_PACKAGE_LIST


def list_packages() -> list[Package]:
    """Packages with listable models (excludes custom)."""
    return [p for p in list_base_packages() if isinstance(p, Package)]


def get_package(package_name: str) -> BasePackage:
    for package in list_base_packages():
        if package.name == package_name:
            return package
    supported = ", ".join(p.name for p in list_base_packages())
    raise ValueError(
        f"Unknown package: '{package_name}'. Supported packages: {supported}"
    )


def list_model_names() -> list[str]:
    """Union of all package model names, sorted."""
    names: list[str] = []
    for package in list_packages():
        names.extend(package.list_model_names())
    return sorted(names)


def parse_model_name(model: str) -> tuple[str, str]:
    """Split ``"package/model_name"`` (first slash only)."""
    if "/" not in model:
        raise ValueError(
            f"The model name '{model}' must have the format '<package>/<model>'."
        )
    package_name, model_name = model.split("/", maxsplit=1)
    return package_name, model_name


def get_wrapped_model(
    model: str | EncoderProtocol | nn.Module,
    num_input_channels: int = 3,
    model_args: dict[str, Any] | None = None,
    load_weights: bool = True,
) -> EncoderProtocol:
    """Resolve any model specification to a ``EncoderProtocol``.

    - ``EncoderProtocol`` instance: returned as-is.
    - ``"package/model"`` string: built by the named package.
    - other ``nn.Module``: attributed to a package via ``is_supported_model``.
    """
    # Duck-typed check first: any object implementing the full protocol is
    # returned unchanged (structural subtyping, no base class required).
    if not isinstance(model, str) and not missing_model_wrapper_attrs(model):
        return model  # type: ignore[return-value]
    if isinstance(model, str):
        package_name, model_name = parse_model_name(model)
        package = get_package(package_name)
        if not isinstance(package, Package):
            raise ValueError(f"Package '{package_name}' does not support named models.")
        raw_model = package.get_model(
            model_name,
            num_input_channels=num_input_channels,
            model_args=model_args,
            load_weights=load_weights,
        )
        return package.get_model_wrapper(raw_model)
    if isinstance(model, nn.Module):
        package = get_package_from_model(
            model, include_custom=False, fallback_custom=False
        )
        if not isinstance(package, Package):
            raise UnknownModelError(
                f"Cannot attribute model of type '{type(model).__name__}' to any package."
            )
        return package.get_model_wrapper(model)
    raise UnknownModelError(
        f"Cannot resolve model specification of type '{type(model).__name__}'."
    )


def get_package_from_model(
    model: Any,
    include_custom: bool = True,
    fallback_custom: bool = True,
) -> BasePackage | None:
    """Find the package a built model belongs to via ``is_supported_model``."""
    packages = list_base_packages() if include_custom else _BASE_PACKAGES
    for package in packages:
        if package.is_supported_model(model):
            return package
    if fallback_custom:
        return custom_package()
    raise UnknownModelError(
        f"Unknown model of type '{type(model).__name__}'. If it is a custom "
        f"model, make it implement the EncoderProtocol protocol. Missing "
        f"attributes: {missing_model_wrapper_attrs(model, exclude_module_attrs=True)}"
    )


# Custom is provided by its own module and must always be registered LAST
# (named packages are preferred; custom is the last resort).
from autovision.encoders.custom.custom_package import CustomPackage  # noqa: E402

_CUSTOM_PACKAGE_LIST.append(CustomPackage())
CUSTOM_PACKAGE = _CUSTOM_PACKAGE_LIST[0]
custom_package = lambda: CUSTOM_PACKAGE  # noqa: E731
