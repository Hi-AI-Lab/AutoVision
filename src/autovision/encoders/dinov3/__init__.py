# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""DINOv3 package backed by HuggingFace transformers (gated)."""

from autovision.encoders.dinov3.dinov3_package import DINOV3_PACKAGE
from autovision.encoders.dinov3.dinov3_wrapper import (
    TransformersDinov3ConvNextWrapper,
    TransformersDinov3ViTWrapper,
)

__all__ = ["DINOV3_PACKAGE", "TransformersDinov3ConvNextWrapper", "TransformersDinov3ViTWrapper"]
