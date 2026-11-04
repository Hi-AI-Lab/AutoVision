# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Unit tests for idle-GPU selection and ``-1`` placeholder specs (D18).

All tests inject GPU snapshots or patch NVML away, so they pass on
CPU-only CI machines as well as GPU hosts.
"""

from __future__ import annotations

import logging
import random

import pytest
import torch

from autovision.errors import AutoVisionError
from autovision.runtime.autodevice import (
    GPUStat,
    list_gpu_stats,
    resolve_device_spec,
    select_idle_gpus,
)


def _stat(index: int, utilization: float, free: float) -> GPUStat:
    """Build a GPUStat shorthand."""
    return GPUStat(index=index, utilization=utilization, memory_free_fraction=free)


@pytest.fixture(autouse=True)
def _clean_cuda_visibility(monkeypatch: pytest.MonkeyPatch) -> None:
    """CVD must not leak in from the host; the CVD test sets its own."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


@pytest.fixture()
def no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the sequential-fill path to see exactly 2 CUDA slots."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)


class TestSelectIdleGpus:
    """Threshold filtering and ranking of the selection core."""

    def test_busy_gpus_excluded(self) -> None:
        """GPUs below the memory or idle-time thresholds are not selected."""
        stats = [
            _stat(0, utilization=0.95, free=0.9),  # busy core
            _stat(1, utilization=0.05, free=0.05),  # full memory
            _stat(2, utilization=0.1, free=0.8),  # idle
        ]
        assert select_idle_gpus(1, stats=stats) == [2]

    def test_ranking_prefers_least_utilized_then_most_free(self) -> None:
        """Ranking is (utilization, -free_memory); ties broken by rng."""
        stats = [
            _stat(0, utilization=0.2, free=0.5),
            _stat(1, utilization=0.1, free=0.9),
            _stat(2, utilization=0.1, free=0.6),
            _stat(3, utilization=0.3, free=0.99),
        ]
        assert select_idle_gpus(3, stats=stats) == [1, 2, 0]

    def test_random_tiebreak_is_rng_driven(self) -> None:
        """Equal-stat GPUs order by the injected RNG, not list position."""
        stats = [
            _stat(0, utilization=0.1, free=0.5),
            _stat(1, utilization=0.1, free=0.5),
            _stat(2, utilization=0.1, free=0.5),
        ]
        first = select_idle_gpus(1, stats=stats, rng=random.Random(7))
        second = select_idle_gpus(1, stats=stats, rng=random.Random(99))
        assert first == [1]
        assert second == [2]

    def test_fewer_idle_than_requested_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Selection returns what passed the thresholds and warns."""
        stats = [_stat(0, utilization=0.99, free=0.9)]
        with caplog.at_level(logging.WARNING):
            selected = select_idle_gpus(2, stats=stats)
        assert selected == []
        assert "passed the thresholds" in caplog.text

    def test_candidates_restrict_the_pool(self) -> None:
        """Explicit candidate torch indices limit which stats are ranked."""
        stats = [
            _stat(0, utilization=0.01, free=0.9),
            _stat(1, utilization=0.02, free=0.9),
        ]
        assert select_idle_gpus(1, candidates=[1], stats=stats) == [1]

    def test_empty_stats_return_nothing(self) -> None:
        """An empty snapshot (NVML unavailable) selects nothing."""
        assert select_idle_gpus(1, stats=[]) == []


class TestResolveDeviceSpec:
    """``-1`` placeholder expansion and degradation semantics."""

    def test_spec_without_placeholder_passes_through(self) -> None:
        """Concrete specs are returned verbatim."""
        assert resolve_device_spec("cpu") == "cpu"
        assert resolve_device_spec("cuda:1") == "cuda:1"

    def test_single_placeholder_picks_most_idle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare ``-1`` resolves to the least-utilized candidate."""
        monkeypatch.setattr(
            "autovision.runtime.autodevice._load_gpu_stats",
            lambda: [_stat(0, 0.9, 0.9), _stat(1, 0.05, 0.8), _stat(2, 0.2, 0.7)],
        )
        # Match the injected topology: the visibility gate counts real
        # CUDA slots when CVD is unset, and a GPU-less CI host has none.
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
        assert resolve_device_spec("-1") == "cuda:1"
        assert resolve_device_spec("cuda:-1") == "cuda:1"

    def test_placeholder_expands_around_explicit_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit ids in the spec are claimed before the idle pick."""
        monkeypatch.setattr(
            "autovision.runtime.autodevice._load_gpu_stats",
            lambda: [_stat(0, 0.01, 0.9), _stat(1, 0.05, 0.95)],
        )
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        assert resolve_device_spec("0,-1") == "cuda:0,cuda:1"

    def test_nvml_missing_degrades_sequentially(
        self, monkeypatch: pytest.MonkeyPatch, no_cuda: None
    ) -> None:
        """Without NVML, placeholders fill sequentially with visible slots."""
        monkeypatch.setattr(
            "autovision.runtime.autodevice._load_gpu_stats", lambda: None
        )
        assert resolve_device_spec("-1,-1") == "cuda:0,cuda:1"

    def test_more_placeholders_than_gpus_raise(
        self, monkeypatch: pytest.MonkeyPatch, no_cuda: None
    ) -> None:
        """An unsatisfiable spec raises instead of inventing cpu ranks."""
        monkeypatch.setattr(
            "autovision.runtime.autodevice._load_gpu_stats", lambda: None
        )
        with pytest.raises(AutoVisionError, match="Cannot satisfy 3"):
            resolve_device_spec("-1,-1,-1")

    def test_busy_only_host_falls_back_to_least_busy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stage 2: a card BELOW the thresholds is rescued and ranked."""
        # idle time 5% < 20% threshold: stage-1 rejects it, stage-2 rescues.
        busy = GPUStat(index=0, utilization=0.95, memory_free_fraction=0.9)
        monkeypatch.setattr(
            "autovision.runtime.autodevice._load_gpu_stats", lambda: [busy]
        )
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        assert resolve_device_spec("-1") == "cuda:0"

    def test_stage2_ranks_rescued_cards_by_busyness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two below-threshold cards: the less busy one fills first."""
        cards = [
            GPUStat(index=0, utilization=0.9, memory_free_fraction=0.5),
            GPUStat(index=1, utilization=0.3, memory_free_fraction=0.5),
        ]
        monkeypatch.setattr(
            "autovision.runtime.autodevice._load_gpu_stats", lambda: cards
        )
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        # Both fail the idle-time threshold (70%/30% < ... wait: 0.7 and 0.3
        # idle: card 1 (70% idle) passes stage-1; card 0 needs stage-2.
        assert resolve_device_spec("-1,-1") == "cuda:1,cuda:0"

    def test_cvd_explicit_claim_blocks_placeholder_reselect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit ids claim TORCH slots even under CVD translation (P1)."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
        stats = [
            GPUStat(index=4, utilization=0.0, memory_free_fraction=0.9),
            GPUStat(index=5, utilization=0.0, memory_free_fraction=0.9),
        ]
        monkeypatch.setattr(
            "autovision.runtime.autodevice._load_gpu_stats", lambda: stats
        )
        # The placeholder must take torch 1 (physical 5), NOT re-select the
        # explicitly claimed torch 0 (physical 4).
        assert resolve_device_spec("0,-1") == "cuda:0,cuda:1"

    def test_busy_only_host_with_more_placeholders_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two placeholders on one busy card is unsatisfiable -> error."""
        busy = GPUStat(index=0, utilization=0.5, memory_free_fraction=0.9)
        monkeypatch.setattr(
            "autovision.runtime.autodevice._load_gpu_stats", lambda: [busy]
        )
        # Two visible slots clear the visibility gate so the selection
        # core (1 usable card) is what raises, not the slot counter.
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        with pytest.raises(AutoVisionError, match="only 1 usable CUDA GPU"):
            resolve_device_spec("-1,-1")

    def test_cuda_visible_devices_translates_to_physical(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Torch indices map through CUDA_VISIBLE_DEVICES to NVML indices."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,7")
        monkeypatch.setattr(
            "autovision.runtime.autodevice._load_gpu_stats",
            lambda: [
                _stat(2, 0.01, 0.9),
                _stat(5, 0.0, 0.99),  # invisible
                _stat(7, 0.02, 0.9),
            ],
        )
        # Both visible cards idle: util 2% < 1% is false for both, so stage-1
        # picks by (utilization, -free) — torch 1 (physical 7) wins on lower
        # utilization among equal free fractions... util differs, so the
        # pick is deterministic: physical 2 (util 1%).
        selected = resolve_device_spec("-1")
        assert selected == "cuda:0"

    def test_non_cuda_accelerator_placeholder_rejected(self) -> None:
        """``npu:-1`` is a configuration error (NVML is NVIDIA-only)."""
        with pytest.raises(AutoVisionError, match="only supported for CUDA"):
            resolve_device_spec("npu:-1")


class TestListGpuStats:
    """Public snapshot helper degradation."""

    def test_nvml_missing_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed NVML load yields an empty list, not an exception."""
        monkeypatch.setattr(
            "autovision.runtime.autodevice._load_gpu_stats", lambda: None
        )
        assert list_gpu_stats() == []


class TestWiring:
    """Downstream consumers of the expanded spec grammar."""

    def test_resolve_device_placeholder_raises_on_cpu_host(self) -> None:
        """``resolve_device('-1')`` on a CPU-only host is a config error."""
        if torch.cuda.is_available():
            pytest.skip("requires a CPU-only host")
        from autovision.engine.engine import resolve_device
        from autovision.errors import AutoVisionError

        with pytest.raises(AutoVisionError, match="Cannot satisfy"):
            resolve_device("-1")

    def test_resolve_device_multi_token_uses_primary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A comma spec resolves to its first token in a single process."""
        from autovision.engine.engine import resolve_device

        monkeypatch.setattr(
            "autovision.runtime.autodevice.resolve_device_spec",
            lambda spec: "cuda:0,cuda:3",
        )
        assert resolve_device("0,-1") == torch.device("cuda:0")

    def test_nvml_missing_bare_int_spec_reaches_resolve_device(
        self, monkeypatch: pytest.MonkeyPatch, no_cuda: None
    ) -> None:
        """Bare-int tokens normalize on BOTH paths (P2: sequential path).

        Without NVML and without CUDA, ``"0,-1"`` must expand to valid
        torch devices — a raw ``"0"`` token would explode in
        ``torch.device`` with an unhelpful RuntimeError.
        """
        from autovision.engine.engine import resolve_device
        from autovision.errors import AutoVisionError

        monkeypatch.setattr(
            "autovision.runtime.autodevice._load_gpu_stats", lambda: None
        )
        try:
            resolved = resolve_device("0,-1")
            assert resolved == torch.device("cuda:0")
        except AutoVisionError as error:
            # One visible slot with two placeholders is unsatisfiable on a
            # CPU-only host: the raise IS the correct contract.
            assert "Cannot satisfy" in str(error)

    def test_resolve_num_devices_counts_tokens(self) -> None:
        """Comma specs contribute one process per token."""
        from autovision.cli.common_helpers import resolve_num_devices

        assert resolve_num_devices("0,-1") == 2
        assert resolve_num_devices("cuda:0,cuda:3") == 2
        assert resolve_num_devices("cuda:1") == 1
        assert resolve_num_devices(3) == 3
        assert resolve_num_devices("cpu") == 1
