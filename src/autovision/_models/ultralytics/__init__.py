# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Ultralytics (YOLO/RT-DETR) package. AGPL-3.0 — extras-only dependency."""

from autovision._models.ultralytics.ultralytics import UltralyticsModelWrapper
from autovision._models.ultralytics.ultralytics_package import ULTRALYTICS_PACKAGE

__all__ = ["ULTRALYTICS_PACKAGE", "UltralyticsModelWrapper"]
