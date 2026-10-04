# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Optimizer selection, parameter grouping, and LR scheduling."""

from autovision._optim.optimizer_args import (
    AdamWArgs,
    LARSArgs,
    OptimizerArgs,
    SGDArgs,
    get_optimizer_args_cls,
)
from autovision._optim.optimizer_helpers import get_optimizer, get_weight_decay_parameters
from autovision._optim.trainable_modules import TrainableModules
from autovision._optimizer_type import OptimizerType
from autovision._scheduler import CosineWarmupScheduler

__all__ = [
    "AdamWArgs",
    "CosineWarmupScheduler",
    "LARSArgs",
    "OptimizerArgs",
    "OptimizerType",
    "SGDArgs",
    "TrainableModules",
    "get_optimizer",
    "get_optimizer_args_cls",
    "get_weight_decay_parameters",
]
