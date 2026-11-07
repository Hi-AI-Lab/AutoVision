# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""autobatch (design D4): pure fitter + monkeypatched CUDA probe loop.

The GPU measurement shell is exercised through fakes for
``torch.cuda.mem_get_info`` / ``max_memory_reserved`` etc., so the
decision policy (fit vs OOM fallback vs degenerate) is verified on CPU.
"""

from __future__ import annotations

import pytest
import torch

from autovision.engine.autobatch import fit_batch_size, probe_train_batch_size
from autovision.errors import AutoVisionError

_GIB = 2**30


# ── pure fitter ─────────────────────────────────────────────────────────


def test_fit_extrapolates_to_the_target_fraction() -> None:
    """A linear 1 GiB/batch curve against 24 GiB free lands on batch 14."""
    measurements = {2: 2 * _GIB, 4: 4 * _GIB, 8: 8 * _GIB, 16: 16 * _GIB}

    batch, fitted = fit_batch_size(measurements, free_bytes=24 * _GIB, fraction=0.6)

    assert (batch, fitted) == (14, True)


def test_fit_falls_back_to_the_last_safe_candidate_after_an_oom() -> None:
    """A measured OOM caps the search — no extrapolation past it."""
    batch, fitted = fit_batch_size(
        {1: _GIB, 2: 2 * _GIB, 8: 8 * _GIB, 16: None},
        free_bytes=100 * _GIB,
    )

    assert (batch, fitted) == (8, False)


def test_fit_uses_only_the_monotonic_prefix() -> None:
    """Allocator-reuse dips break the curve; the trustworthy prefix wins."""
    batch, fitted = fit_batch_size(
        {2: 2 * _GIB, 4: 1 * _GIB, 8: 8 * _GIB}, free_bytes=100 * _GIB
    )

    assert (batch, fitted) == (2, False)


def test_fit_degenerates_to_a_flat_curve() -> None:
    """A zero slope has no solution; the largest measured point wins."""
    batch, fitted = fit_batch_size(
        {4: 1 * _GIB, 8: 1 * _GIB}, free_bytes=100 * _GIB
    )

    assert (batch, fitted) == (8, False)


def test_fit_clamps_and_guards_degenerate_inputs() -> None:
    """A single point returns unfitted; a sub-1 fit clamps to batch 1."""
    assert fit_batch_size({8: 1 * _GIB}, free_bytes=4 * _GIB) == (8, False)
    # slope 1 GiB/batch, intercept 0.5 GiB: target 2.4 GiB -> estimate 1.
    measurements = {2: int(2.5 * _GIB), 4: int(4.5 * _GIB)}
    assert fit_batch_size(
        measurements, free_bytes=4 * _GIB, fraction=0.6
    ) == (1, True)


def test_fit_raises_when_every_candidate_oomed() -> None:
    """No measurement survives: the error tells the user what to change."""
    with pytest.raises(AutoVisionError, match="every candidate"):
        fit_batch_size({1: None, 2: None}, free_bytes=24 * _GIB)


# ── measurement shell (monkeypatched CUDA) ──────────────────────────────


class _FakeCuda:
    """Records the probed batch sizes and simulates a 1 GiB/batch curve."""

    def __init__(self, *, free_bytes: int, oom_at: int) -> None:
        self.free_bytes = free_bytes
        self.oom_at = oom_at
        self.probed: list[int] = []
        self._peak = 0

    def mem_get_info(self, device=None):  # noqa: ANN001, ANN202
        return self.free_bytes, self.free_bytes * 2

    def empty_cache(self, *args) -> None:  # noqa: ANN002
        return None

    def reset_peak_memory_stats(self, *args) -> None:  # noqa: ANN002
        self._peak = 0

    def max_memory_reserved(self, *args) -> int:  # noqa: ANN002
        return self._peak

    def synchronize(self, *args) -> None:  # noqa: ANN002
        return None

    def step(self, batch: int) -> None:
        """The 'real' training step: one batch of `batch` GiB, OOM past cap."""
        if batch >= self.oom_at:
            raise torch.cuda.OutOfMemoryError
        self.probed.append(batch)
        self._peak = batch * _GIB


def _install(fake: _FakeCuda, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "mem_get_info", fake.mem_get_info)
    monkeypatch.setattr(torch.cuda, "empty_cache", fake.empty_cache)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", fake.reset_peak_memory_stats)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", fake.max_memory_reserved)
    monkeypatch.setattr(torch.cuda, "synchronize", fake.synchronize)


def test_probe_fits_when_no_candidate_ooms(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean curve yields the fitted batch and a populated probe record."""
    fake = _FakeCuda(free_bytes=24 * _GIB, oom_at=10_000)
    _install(fake, monkeypatch)

    probe = probe_train_batch_size(
        fake.step, device="cuda", candidates=[2, 4, 8, 16]
    )

    assert (probe.batch_size, probe.fitted) == (14, True)
    assert probe.oom_ceiling is None
    assert probe.measured == {2: 2 * _GIB, 4: 4 * _GIB, 8: 8 * _GIB, 16: 16 * _GIB}


def test_probe_falls_back_when_a_candidate_ooms(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OOM ceiling switches the answer to the last safe candidate."""
    fake = _FakeCuda(free_bytes=24 * _GIB, oom_at=32)
    _install(fake, monkeypatch)

    probe = probe_train_batch_size(fake.step, device="cuda")

    assert (probe.batch_size, probe.fitted) == (16, False)
    assert probe.oom_ceiling == 32
    # The default candidate set gains 32/64 on >= 16 GiB devices.
    assert set(probe.measured) == {1, 2, 4, 8, 16, 32, 64}


def test_probe_rejects_cpu_and_cudnn_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards fail closed with actionable messages."""
    with pytest.raises(AutoVisionError, match="CUDA/NPU"):
        probe_train_batch_size(lambda batch: None, device="cpu")
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    with pytest.raises(AutoVisionError, match="cudnn.benchmark"):
        probe_train_batch_size(lambda batch: None, device="cuda")
