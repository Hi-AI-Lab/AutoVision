# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""PretextRecipe argument base class."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from autovision.config.base import PydanticConfig
from autovision.pretext.knn_eval import KnnEvalArgs

__all__ = ["PretextArgs"]


class PretextArgs(PydanticConfig):
    """Base class for method arguments.

    ``default_steps`` and ``default_epochs`` are mutually exclusive class
    variables consumed by the ``epochs="auto"`` resolution: if the method
    defines a step-based recipe, epochs are derived from it; otherwise the
    fixed epoch count is used.

    ``knn_eval`` configures the optional in-training kNN probe (§7.5); it is
    resolved and executed outside the method itself (see
    :mod:`autovision.pretext.knn_eval`).
    """

    default_steps: ClassVar[int | None] = None
    default_epochs: ClassVar[int | None] = 100

    knn_eval: KnnEvalArgs | None = None

    lr_scale_method: Literal["linear", "sqrt"] = "linear"
    reference_batch_size: int = 256

    def resolve_auto(
        self,
        scaling_info: Any = None,
        optimizer_args: Any = None,
        wrapped_model: Any = None,
    ) -> None:
        """Resolve any ``"auto"`` fields (default: nothing to resolve)."""
        if self.knn_eval is not None and scaling_info is not None:
            self.knn_eval.resolve_auto(dataset_size=scaling_info.dataset_size)
