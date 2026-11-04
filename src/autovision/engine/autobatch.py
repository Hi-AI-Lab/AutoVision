# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Automatic train-batch probing (design D4).

``probe_train_batch_size`` answers "what is the largest batch this
device can train with" by MEASURING: each candidate batch runs a real
step (forward + backward through the task's own loss/assigner, so the
memory profile includes what a forward-only probe would miss) under a
peak-memory reset, and the survivors are extrapolated to a target
fraction of free device memory. A candidate that OOMs marks the ceiling:
the last safe candidate wins outright (a linear fit through the memory
curve is only an extrapolation, but an OOM is a measurement).

The pure decision logic lives in :func:`fit_batch_size` (numpy only) so
the fitting/fallback policy is unit-testable without a GPU; the
measurement loop is a thin, GPU-only shell around it.

CLI integration note: ``batch_size="auto"`` is wired per task surface
(each task supplies its own ``step_fn`` closure); until a surface wires
it, callers pass an explicit batch size. See design D4
(docs/provenance/LEDGER.md).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from autovision.errors import AutoVisionError

__all__ = ["BatchProbe", "fit_batch_size", "probe_train_batch_size"]

logger = logging.getLogger(__name__)

#: Extra candidates offered on large-memory devices (>= 16 GiB free).
_LARGE_MEMORY_CANDIDATES = (32, 64)
_LARGE_MEMORY_FREE_BYTES = 16 * 2**30


@dataclass(frozen=True)
class BatchProbe:
    """Probe outcome, kept for logs and reproducibility records."""

    batch_size: int
    fraction: float
    #: Candidate -> peak reserved bytes (``None`` where the candidate OOMed).
    measured: dict[int, int | None]
    #: First candidate that OOMed after a successful one (``None`` if none).
    oom_ceiling: int | None
    #: True when the answer came from the linear fit, False from OOM
    #: fallback or a degenerate measurement set.
    fitted: bool


def fit_batch_size(
    measurements: Mapping[int, int | None],
    *,
    free_bytes: int,
    fraction: float = 0.6,
    max_batch: int = 1024,
) -> tuple[int, bool]:
    """Pure decision: measurements -> (batch size, was-fitted).

    Policy (in order):
    1. An OOM after successful candidates caps the search — the answer is
       the last successful candidate below it, not an extrapolation.
    2. Otherwise the monotonic non-decreasing memory prefix is fitted with
       a first-order polynomial and solved for
       ``target = free_bytes * fraction``; extrapolation past the measured
       range is the point (candidates are sparse by design).
    3. Fewer than two usable points degenerate to the largest success.
    The result is clamped to ``[1, max_batch]``.
    """
    successes = sorted(b for b, memory in measurements.items() if memory is not None)
    ooms = sorted(b for b, memory in measurements.items() if memory is None)
    if not successes:
        raise AutoVisionError(
            "autobatch: every candidate batch hit OOM; the model or input "
            "is too large for this device. Reduce the input size or batch."
        )
    oom_ceiling = next((b for b in ooms if b > successes[0]), None)
    if oom_ceiling is not None:
        safe = max(b for b in successes if b < oom_ceiling)
        logger.info(
            "autobatch: OOM at batch %s; falling back to the last safe "
            "candidate %s (no extrapolation past a measured OOM).",
            oom_ceiling, safe,
        )
        return min(safe, max_batch), False

    # Monotonic non-decreasing prefix: allocator reuse can make a larger
    # batch report less memory; such points are untrustworthy for fitting.
    prefix: list[tuple[int, int]] = []
    last = -1
    for batch in successes:
        memory = int(measurements[batch] or 0)
        if prefix and memory < last:
            break
        prefix.append((batch, memory))
        last = memory
    if len(prefix) < 2:
        return min(prefix[-1][0], max_batch), False

    batches = np.array([b for b, _ in prefix], dtype=np.float64)
    memory = np.array([m for _, m in prefix], dtype=np.float64)
    slope, intercept = np.polyfit(batches, memory, 1)
    # A usable curve grows by at least one byte per batch item; anything
    # flatter is a numerically-zero fit (identical measurements), and
    # extrapolating it would return the clamp ceiling.
    if slope < 1.0:
        return min(prefix[-1][0], max_batch), False
    target = float(free_bytes) * float(fraction)
    estimate = int((target - float(intercept)) / float(slope))
    return max(1, min(estimate, max_batch)), True


def probe_train_batch_size(
    step_fn: Callable[[int], None],
    *,
    device: torch.device | str = "cuda",
    fraction: float = 0.6,
    candidates: Sequence[int] | None = None,
    max_batch: int = 1024,
) -> BatchProbe:
    """Measure a real training step per candidate and fit the batch size.

    ``step_fn(batch_size)`` must run ONE full training step (data fetch,
    forward, backward) on ``device`` and clean up after itself; the probe
    resets peak-memory counters around each call, so the step must not
    span threads or leave a background CUDA stream running. ``fraction``
    is the share of currently FREE device memory the fitted step may
    occupy (0.6 leaves room for fragmentation and the allocator's slack).
    """
    device = torch.device(device)
    if device.type not in ("cuda", "npu"):
        raise AutoVisionError(
            f"autobatch probing needs a CUDA/NPU device, got '{device.type}'; "
            "pass an explicit batch_size instead."
        )
    if torch.backends.cudnn.benchmark:
        raise AutoVisionError(
            "autobatch probing is unreliable with cudnn.benchmark=True "
            "(the benchmark allocator skews peak memory); disable it or "
            "pass an explicit batch_size."
        )
    free_bytes, _total = torch.cuda.mem_get_info(device)
    if candidates is None:
        candidates = [1, 2, 4, 8, 16]
        if free_bytes >= _LARGE_MEMORY_FREE_BYTES:
            candidates = [*candidates, *_LARGE_MEMORY_CANDIDATES]

    oom_error: type[BaseException] = getattr(
        torch.cuda, "OutOfMemoryError", RuntimeError
    )
    measurements: dict[int, int | None] = {}
    try:
        for candidate in candidates:
            batch = int(candidate)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            try:
                step_fn(batch)
                torch.cuda.synchronize(device)
                measurements[batch] = int(torch.cuda.max_memory_reserved(device))
            except (oom_error, RuntimeError) as error:
                if not _is_oom(error):
                    raise
                measurements[batch] = None
                logger.info("autobatch: batch %d hit OOM during probing.", batch)
    finally:
        torch.cuda.empty_cache()

    batch_size, fitted = fit_batch_size(
        measurements, free_bytes=free_bytes, fraction=fraction, max_batch=max_batch
    )
    probe = BatchProbe(
        batch_size=batch_size,
        fraction=fraction,
        measured=dict(measurements),
        oom_ceiling=next(
            (b for b, memory in measurements.items() if memory is None
             and any(s < b for s, m in measurements.items() if m is not None)),
            None,
        ),
        fitted=fitted,
    )
    logger.info(
        "autobatch: batch_size=%d (%s, fraction=%.2f, free=%.1f GiB, "
        "measured=%s).",
        probe.batch_size,
        "fitted" if probe.fitted else "OOM fallback",
        fraction,
        free_bytes / 2**30,
        probe.measured,
    )
    return probe


def _is_oom(error: BaseException) -> bool:
    """True for CUDA/host out-of-memory failures, False for real bugs."""
    if isinstance(error, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    text = str(error).lower()
    return "out of memory" in text or "not enough memory" in text
