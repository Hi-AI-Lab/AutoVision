# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Optimizer selection, parameter grouping, and LR scheduling."""

from autovision.optim.optimizer_args import (
    AdamWArgs,
    LARSArgs,
    OptimizerArgs,
    SGDArgs,
    get_optimizer_args_cls,
)
from autovision.optim.optimizer_helpers import (
    build_adamw,
    get_optimizer,
    get_weight_decay_parameters,
)
from autovision.optim.optimizer_type import OptimizerType
from autovision.optim.schedulers import CosineWarmupScheduler
from autovision.optim.trainable_modules import TrainableModules

__all__ = [
    "AdamWArgs",
    "CosineWarmupScheduler",
    "LARSArgs",
    "OptimizerArgs",
    "OptimizerType",
    "SGDArgs",
    "TrainableModules",
    "build_adamw",
    "get_optimizer",
    "get_optimizer_args_cls",
    "get_weight_decay_parameters",
]
