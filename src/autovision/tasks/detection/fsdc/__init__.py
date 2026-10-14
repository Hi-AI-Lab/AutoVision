# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""FSDC-DETR object detection family (design doc 22).

Native re-implementation of the FSDC-DETR paper's frequency-spatial
stack on the DEIMv2 trunk: dual-branch DINOv3+STA backbone with DBFSAF
fusion (Eq.1-6) and the FSC hybrid encoder with SFS-FF/FSD-Down at the
PAN junction (Eq.10-13). Decoder + criterion remain the official HF
objects.
"""

from autovision.tasks.detection.fsdc.config import (
    FSDC_MODEL_REGISTRY,
    FsdcArgs,
    FsdcObjectDetectionTrainArgs,
    resolve_fsdc_variant,
)
from autovision.tasks.detection.fsdc.dual_branch_backbone import (
    FsdcDinoDualBranchEncoder,
    FsdcSpatialTuningAdapter,
)
from autovision.tasks.detection.fsdc.fsc_encoder import (
    FsdcHybridEncoder,
)
from autovision.tasks.detection.fsdc.task_model import (
    FsdcObjectDetection,
)
from autovision.tasks.detection.fsdc.train_model import (
    FsdcObjectDetectionTrain,
)

__all__ = [
    "FSDC_MODEL_REGISTRY",
    "FsdcArgs",
    "FsdcDinoDualBranchEncoder",
    "FsdcHybridEncoder",
    "FsdcObjectDetection",
    "FsdcObjectDetectionTrain",
    "FsdcObjectDetectionTrainArgs",
    "FsdcSpatialTuningAdapter",
    "resolve_fsdc_variant",
]
