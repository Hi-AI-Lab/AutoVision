# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""EdgeCrafter (ECViT) backbone package (registers itself on import)."""

from autovision._models.ecvit.ecvit import ECViTModelWrapper
from autovision._models.ecvit.ecvit_package import ECVIT_PACKAGE

__all__ = ["ECVIT_PACKAGE", "ECViTModelWrapper"]
