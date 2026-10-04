# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""SAHI slicing-aided inference for detection task models."""

from autovision._task_models.object_detection_components.tiling_utils import (
    combine_object_detection_tiles,
    tile_image,
)

__all__ = ["combine_object_detection_tiles", "tile_image"]
