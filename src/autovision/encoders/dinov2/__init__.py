# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""DINOv2 package backed by HuggingFace transformers."""

from autovision.encoders.dinov2.dinov2_package import DINOV2_PACKAGE
from autovision.encoders.dinov2.dinov2_wrapper import TransformersDinov2Wrapper

__all__ = ["DINOV2_PACKAGE", "TransformersDinov2Wrapper"]
