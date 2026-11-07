# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Unit tests for D29a (storage speed probe) and D29b (maxDets auto-raise)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from autovision.data.quality import probe_storage_speed
from autovision.metrics.map_evaluator import COCOMAPEvaluator


@pytest.fixture()
def fast_dir(tmp_path: Path) -> Path:
    """A local tmp dir with one large and a few small files."""
    large = tmp_path / "large.bin"
    large.write_bytes(b"\0" * (4 * 1024 * 1024))
    for index in range(3):
        (tmp_path / f"small_{index}.bin").write_bytes(b"\0" * 1024)
    return tmp_path


class TestProbeStorageSpeed:
    """Sampling semantics and threshold warnings (design D29a)."""

    def test_reports_statistics(self, fast_dir: Path) -> None:
        """A readable sample returns ping/throughput/file counts."""
        paths = sorted(fast_dir.iterdir())
        stats = probe_storage_speed(paths, samples=4)
        assert stats is not None
        assert stats["files_sampled"] == 4.0
        assert stats["stat_ms_mean"] >= 0.0
        assert stats["throughput_mb_s_mean"] > 0.0

    def test_slow_storage_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch
    ) -> None:
        """Pings over the threshold produce exactly one advisory warning."""
        slow = tmp_path / "slow.bin"
        slow.write_bytes(b"\0" * (2 * 1024 * 1024))
        import autovision.data.quality as quality

        real_stat = quality.os.stat

        def slow_stat(path, *args, **kwargs):
            import time

            time.sleep(0.02)  # 20 ms: above the 10 ms threshold
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(quality.os, "stat", slow_stat)
        with caplog.at_level(logging.WARNING):
            stats = probe_storage_speed([slow], samples=1)
        assert stats is not None
        assert "non-local storage" in caplog.text

    def test_missing_files_skipped_and_empty_returns_none(
        self, tmp_path: Path
    ) -> None:
        """Unreadable paths drop out of the sample; no paths return None."""
        assert probe_storage_speed([]) is None
        ghost = tmp_path / "ghost.bin"
        assert probe_storage_speed([ghost]) is None

    def test_small_files_exempt_from_throughput(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch
    ) -> None:
        """Sub-16KB files measure ping only — no bogus throughput warning."""
        tiny = tmp_path / "tiny.bin"
        tiny.write_bytes(b"\0" * 64)
        import autovision.data.quality as quality

        recorded: list[int] = []

        real_stat = quality.os.stat

        def recording_stat(path, *args, **kwargs):
            recorded.append(1)
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(quality.os, "stat", recording_stat)
        with caplog.at_level(logging.WARNING):
            stats = probe_storage_speed([tiny], samples=1)
        assert stats is not None
        assert "throughput_mb_s_mean" not in stats
        assert "slow storage will bottleneck" not in caplog.text


class TestMaxDetsAutoRaise:
    """D29b: densest-image tracking and opt-in cap lift."""

    @staticmethod
    def _evaluator(**kwargs) -> COCOMAPEvaluator:
        """Build an evaluator over one class."""
        return COCOMAPEvaluator(categories={1: "cat"}, **kwargs)

    def test_tracks_densest_image(self) -> None:
        """``update_targets`` records the largest instance count seen."""
        evaluator = self._evaluator()
        evaluator.update_targets(
            [{"boxes": torch.zeros(150, 4), "labels": torch.ones(150, dtype=torch.long)}],
            [(100, 100)],
        )
        evaluator.update_targets(
            [{"boxes": torch.zeros(3, 4), "labels": torch.ones(3, dtype=torch.long)}],
            [(100, 100)],
        )
        assert evaluator._max_instances_seen == 150

    def test_auto_raise_lifts_cap_at_compute_time(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Opt-in flag evaluates with a lifted cap, warns, and restores.

        The empty-predictions early return in ``compute()`` sits after the
        cap check, so no pycocotools machinery is needed to observe the
        warning. The lift itself is per-call (H3-6c): the attribute keeps
        the caller's pinned cap so a second round inherits no protocol.
        """
        evaluator = self._evaluator(auto_raise=True)
        evaluator.update_targets(
            [{"boxes": torch.zeros(200, 4), "labels": torch.ones(200, dtype=torch.long)}],
            [(100, 100)],
        )
        with caplog.at_level(logging.WARNING):
            metrics = evaluator.compute()
        assert metrics["map"] == 0.0  # early return, no predictions
        assert evaluator.max_dets == 100  # restored — never mutated (H3-6c)
        assert "raised to 200" in caplog.text

    def test_default_keeps_cap_fixed(self) -> None:
        """Without opt-in, an explicit protocol cap is never touched."""
        evaluator = self._evaluator(max_dets=500)
        evaluator.update_targets(
            [{"boxes": torch.zeros(900, 4), "labels": torch.ones(900, dtype=torch.long)}],
            [(100, 100)],
        )
        assert evaluator.max_dets == 500
