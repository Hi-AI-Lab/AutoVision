# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Object detection building blocks shared by task model families."""

from autovision.tasks.detection.components.box_ops import (
    box_cxcywh_norm_to_xyxy,
    box_xyxy_to_cxcywh_norm,
    generalized_box_iou_loss,
)
from autovision.tasks.detection.components.ema import ModelEMA
from autovision.tasks.detection.components.flat_cosine import (
    FlatCosineLRScheduler,
)
from autovision.tasks.detection.components.hybrid_encoder import (
    HybridEncoder,
)
from autovision.tasks.detection.components.matcher import (
    HungarianMatcher,
)
from autovision.tasks.detection.components.rtdetrv2_criterion import (
    RTDETRCriterionV2,
)
from autovision.tasks.detection.components.rtdetrv2_decoder import (
    RTDETRv2Decoder,
    inverse_sigmoid,
)
from autovision.tasks.detection.components.sahi import (
    combine_object_detection_tiles,
    tile_image,
)

__all__ = [
    "FlatCosineLRScheduler",
    "HungarianMatcher",
    "HybridEncoder",
    "ModelEMA",
    "RTDETRCriterionV2",
    "RTDETRv2Decoder",
    "combine_object_detection_tiles",
    "tile_image",
    "box_cxcywh_norm_to_xyxy",
    "box_xyxy_to_cxcywh_norm",
    "generalized_box_iou_loss",
    "inverse_sigmoid",
]
