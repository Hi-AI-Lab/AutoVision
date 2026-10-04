# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""DINOv3: DINOv2 post-training + PaKA dense-relational auxiliary loss."""

from autovision._methods.dinov3.dinov3 import (
    DINOv3,
    DINOv3Args,
    DINOv3Transform,
    DINOv3TransformArgs,
)

__all__ = ["DINOv3", "DINOv3Args", "DINOv3Transform", "DINOv3TransformArgs"]
