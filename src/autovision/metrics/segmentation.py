# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Semantic segmentation metrics: confusion-matrix mIoU and pixel accuracy."""

from __future__ import annotations

from typing import Any

import torch

from autovision.errors import AutoVisionError

__all__ = ["SegmentationMetricCollector"]


class SegmentationMetricCollector:
    """mIoU + pixel accuracy over an accumulated confusion matrix.

    Label 255 marks ignored pixels (cityscapes convention) and never counts
    towards either metric.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one batch: class-index maps of shape (B, H, W)."""
        predictions = predictions.detach().cpu().long().flatten()
        targets = targets.detach().cpu().long().flatten()
        valid = targets != self.ignore_index
        predictions = predictions[valid]
        targets = targets[valid]
        # METRIC-09: the TARGET domain is annotation truth — a negative
        # value (some datasets use -1 as ignore instead of 255) used to
        # crash bincount with a raw RuntimeError, and a value at/above
        # num_classes crashed the reshape; both are annotation corruption
        # and must be named. (PREDICTION out-of-range stays defensive: a
        # misconfigured head is a recoverable training-time condition.)
        target_bad = (targets < 0) | (targets >= self.num_classes)
        if bool(target_bad.any()):
            raise AutoVisionError(
                f"Segmentation target(s) contain {int(target_bad.sum())} "
                f"value(s) outside [0, {self.num_classes}) other than the "
                f"ignore_index={self.ignore_index} — annotation corruption "
                f"(min={int(targets.min())}, max={int(targets.max())}). "
                "Set ignore_index to the dataset's convention or fix the "
                "labels."
            )
        targets = targets[~target_bad]
        predictions = predictions[~target_bad]
        # Drop predictions of out-of-range classes defensively.
        in_range = (predictions >= 0) & (predictions < self.num_classes)
        predictions = predictions[in_range]
        targets = targets[in_range]
        indices = targets * self.num_classes + predictions
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        self._matrix += counts.reshape(self.num_classes, self.num_classes)

    def reset(self) -> None:
        self._matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )

    def compute(self) -> dict[str, float]:
        matrix = self._matrix.float()
        intersection = matrix.diag()
        union = matrix.sum(dim=0) + matrix.sum(dim=1) - intersection
        present = union > 0
        if not present.any():
            return {"miou": 0.0, "pixel_accuracy": 0.0}
        miou = float((intersection[present] / union[present]).mean())
        total = matrix.sum()
        pixel_accuracy = float(intersection.sum() / total) if total > 0 else 0.0
        return {"miou": miou, "pixel_accuracy": pixel_accuracy}

    # ── distributed validation support ──────────────────────────────────
    def export_state(self) -> dict[str, Any]:
        return {"matrix": self._matrix.clone()}

    def merge_state(self, state: dict[str, Any]) -> None:
        self._matrix += state["matrix"].to(self._matrix.dtype)
