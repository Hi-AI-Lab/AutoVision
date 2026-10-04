# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""DenseCL: single-image contrastive learning with dense correspondence."""

from autovision._methods.densecl.densecl import (
    DenseCL,
    DenseCLArgs,
    DenseCLSGDArgs,
)

__all__ = ["DenseCL", "DenseCLArgs", "DenseCLSGDArgs"]
