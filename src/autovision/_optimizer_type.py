# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Supported optimizer types."""

from __future__ import annotations

from enum import Enum, unique

__all__ = ["OptimizerType"]


@unique
class OptimizerType(str, Enum):
    ADAMW = "adamw"
    ADAMW8BIT = "adamw8bit"
    SGD = "sgd"
    LARS = "lars"
