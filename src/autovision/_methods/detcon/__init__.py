# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""DetCon: masked region-contrastive pretraining (S = SupCon / B = BYOL)."""

from autovision._methods.detcon.detcon import (
    DetConB,
    DetConBArgs,
    DetConBLoss,
    DetConBSGDArgs,
    DetConS,
    DetConSArgs,
    DetConSGDArgs,
    DetConSLoss,
)

__all__ = [
    "DetConB",
    "DetConBArgs",
    "DetConBLoss",
    "DetConBSGDArgs",
    "DetConS",
    "DetConSArgs",
    "DetConSGDArgs",
    "DetConSLoss",
]
