# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""EOMT shared core (HF-backed): model construction, label conversion,
semantic decode, optimizer and transforms. Task packages import from here.
"""

from autovision.tasks.eomt.labels import semantic_masks_to_instances
from autovision.tasks.eomt.modeling import adapt_eomt_head, build_eomt_model
from autovision.tasks.eomt.postprocess import semantic_segmentation_decode
from autovision.tasks.eomt.scheduler import build_eomt_optimizer

__all__ = [
    "adapt_eomt_head",
    "build_eomt_model",
    "build_eomt_optimizer",
    "semantic_masks_to_instances",
    "semantic_segmentation_decode",
]
