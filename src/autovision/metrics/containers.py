# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Lazy per-batch metric buffering (design doc D27).

``TaskMetrics`` is the aggregation container the metric collectors
converge on: per-batch stats accumulate as raw lists (no per-batch
reduction — cheap in the loop, and DDP ranks can merge raw shards), and
the expensive reduction happens once in :meth:`process`. Existing
``Collector``/``Evaluator`` classes keep their public behavior; new task
faces and progressive refactors adopt this container instead.

Two shared conventions live here so every collector inherits them:

- :meth:`export_state` / :meth:`merge_state` keep the established duck
  protocol the engines already call;
- :func:`empty_empty_score` fixes the per-image convention "no ground
  truth + no predictions = perfect", avoiding 0/0 ``nan`` leaking into
  error analysis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["TaskMetrics", "empty_empty_score"]


def empty_empty_score(
    num_targets: int, num_predictions: int, *, perfect: float = 1.0
) -> float | None:
    """Per-image score convention for the empty/empty corner (D27).

    Args:
        num_targets: Ground-truth instance count for one image.
        num_predictions: Prediction count for the same image.
        perfect: The score assigned to the perfect-empty case.

    Returns:
        ``perfect`` when both counts are zero (an image with nothing to
        find and nothing predicted is fully correct), ``0.0`` for the
        missed-everything corners (targets without predictions, or
        predictions without targets), and ``None`` when the caller has
        real work to score (both counts nonzero) — the caller computes
        that case itself.
    """
    if num_targets == 0 and num_predictions == 0:
        return perfect
    if num_targets == 0 or num_predictions == 0:
        return 0.0
    return None


class TaskMetrics:
    """Base container: lazy stats buffer + deferred reduction.

    Subclasses append per-batch rows via :meth:`update` (arbitrary named
    columns, typically numpy arrays or scalars) and implement
    :meth:`process` to reduce the concatenated columns into the final
    metric dict. The buffer holds raw references until ``process`` —
    per-batch work stays O(append) and cross-rank merging is a dict of
    list concatenations.

    Args:
        columns: Stat column names this collector accumulates.
    """

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns: tuple[str, ...] = tuple(columns)
        self.stats: dict[str, list[Any]] = {name: [] for name in self.columns}

    def update(self, **items: Any) -> None:
        """Append one per-batch row; every declared column must be given."""
        missing = [name for name in self.columns if name not in items]
        if missing:
            raise ValueError(
                f"update() missing column(s) {missing}; expected "
                f"{list(self.columns)}."
            )
        unknown = [name for name in items if name not in self.stats]
        if unknown:
            raise ValueError(
                f"update() got undeclared column(s) {unknown}; declared "
                f"{list(self.columns)}."
            )
        for name, value in items.items():
            self.stats[name].append(value)

    def _concatenate(self, values: list[Any]) -> np.ndarray:
        """Concatenate one column's rows; callers can override the stacking.

        METRIC audit P3-22: scalar rows (the docstring promises "numpy
        arrays or scalars") crashed np.concatenate — they join as
        one-element arrays.
        """
        rows = [np.asarray(value) for value in values]
        if not rows:
            return np.zeros(0)
        if all(row.ndim == 0 for row in rows):
            return np.stack(rows)
        rows = [row.reshape(1) if row.ndim == 0 else row for row in rows]
        return np.concatenate(rows, axis=0)

    def process(self) -> dict[str, float]:
        """Reduce the buffered stats into final metrics (subclass hook)."""
        raise NotImplementedError(
            "TaskMetrics subclasses implement process(); the base container "
            "only buffers."
        )

    def reset(self) -> None:
        """Drop all buffered rows (fresh evaluation pass)."""
        for name in self.stats:
            self.stats[name] = []

    def export_state(self) -> dict[str, list[Any]]:
        """Picklable raw-buffer snapshot for DDP gather (established duck)."""
        return {name: list(rows) for name, rows in self.stats.items()}

    def merge_state(self, state: Mapping[str, Sequence[Any]]) -> None:
        """Merge another rank's exported raw buffer into this one."""
        unknown = [name for name in state if name not in self.stats]
        if unknown:
            raise ValueError(
                f"merge_state() got unknown column(s) {unknown}; declared "
                f"{list(self.columns)}."
            )
        for name, rows in state.items():
            self.stats[name].extend(rows)
