# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Modules that participate in optimization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from torch import nn

__all__ = ["TrainableModules"]


@dataclass
class TrainableModules:
    """Modules whose parameters are optimized.

    ``modules_no_weight_decay`` modules are always excluded from weight decay
    regardless of the name/shape rules.
    """

    modules: Iterable[nn.Module]
    modules_no_weight_decay: Iterable[nn.Module] = field(default_factory=tuple)
