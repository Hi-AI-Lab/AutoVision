# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Engine/checkpoint small items: conditional cache clearing (D34) and
transient-failure retry on the atomic checkpoint write (D34).

Cache contract: clear accelerator memory only when the caching allocator's
reserved share exceeds the threshold (memory-saturated GPUs — the reason
the unconditional clear existed — always clear); MPS always; CPU never; a
failed occupancy query degrades to always-clear. Write contract: every
attempt gets a fresh temp file, the rename publishes only a fsynced file,
and exhaustion raises CheckpointError with the previous checkpoint intact.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autovision.engine.engine import _maybe_clear_accelerator_cache
from autovision.errors import CheckpointError
from autovision.runtime import checkpoint as checkpoint_module
from autovision.runtime.checkpoint import Checkpoint, CheckpointMetadata


@pytest.fixture(autouse=True)
def _zero_retry_delays(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse checkpoint-write backoff (transient-failure tests only)."""
    monkeypatch.setattr(checkpoint_module, "_SAVE_RETRY_BASE_DELAY_S", 0.0)


@pytest.fixture()
def _fake_cuda(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch torch.cuda memory introspection with a controllable fake."""
    state = {"reserved": 0, "total": 1000, "cleared": 0}

    def fake_empty_cache() -> None:
        state["cleared"] += 1

    monkeypatch.setattr(
        torch.cuda, "memory_reserved", lambda device=None: state["reserved"]
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: type(
            "Props", (), {"total_memory": state["total"]}
        )()
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)
    return state


class TestConditionalCacheClearing:
    """Reserved-fraction threshold rule (D34)."""

    def test_low_reserved_share_skips_the_clear(self, _fake_cuda: dict) -> None:
        """A small training run pays no cudaFree + re-cudaMalloc round trip."""
        _fake_cuda["reserved"] = 100
        _maybe_clear_accelerator_cache(torch.device("cuda:0"))
        assert _fake_cuda["cleared"] == 0

    def test_high_reserved_share_clears(self, _fake_cuda: dict) -> None:
        """Memory-saturated GPUs clear every time (the old behavior)."""
        _fake_cuda["reserved"] = 600
        _maybe_clear_accelerator_cache(torch.device("cuda:0"))
        assert _fake_cuda["cleared"] == 1

    def test_exact_threshold_does_not_clear(self, _fake_cuda: dict) -> None:
        """The rule is strict ``>``, matching "above threshold" semantics."""
        _fake_cuda["reserved"] = 500
        _maybe_clear_accelerator_cache(torch.device("cuda:0"))
        assert _fake_cuda["cleared"] == 0

    def test_failed_occupancy_query_degrades_to_clearing(
        self, _fake_cuda: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An introspection failure must not risk an OOM'd validation."""

        def boom(device) -> int:
            raise RuntimeError("no NVML")

        monkeypatch.setattr(torch.cuda, "memory_reserved", boom)
        _maybe_clear_accelerator_cache(torch.device("cuda:0"))
        assert _fake_cuda["cleared"] == 1

    def test_cpu_device_never_touches_the_cache(self, _fake_cuda: dict) -> None:
        _maybe_clear_accelerator_cache(torch.device("cpu"))
        assert _fake_cuda["cleared"] == 0

    def test_mps_always_clears(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MPS allocators leak; the clear is unconditional (D34)."""
        cleared = []
        monkeypatch.setattr(
            torch, "mps", type("Mps", (), {"empty_cache": lambda: cleared.append(1)})
        )
        _maybe_clear_accelerator_cache(torch.device("mps"))
        assert cleared == [1]


class TestCheckpointWriteRetry:
    """Transient-failure retry on the atomic checkpoint write (D34)."""

    def test_transient_failures_are_retried(self, tmp_path: Path) -> None:
        """Two failed torch.save calls then success: the file is complete."""
        checkpoint = Checkpoint(
            state_dict={"w": torch.ones(2)},
            autovision=CheckpointMetadata(epoch=1, global_step=1),
        )
        calls = {"n": 0}
        real_save = torch.save

        def flaky_save(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise OSError("device flush interrupted the write")
            return real_save(*args, **kwargs)

        target = tmp_path / "last.ckpt"
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch, "save", flaky_save)
            checkpoint.save(target)
        assert calls["n"] == 3
        reloaded = Checkpoint.from_path(target).state_dict
        assert torch.equal(reloaded["w"], torch.ones(2))

    def test_exhausted_retries_raise_checkpoint_error(self, tmp_path: Path) -> None:
        """Three failing attempts surface CheckpointError; nothing is left
        behind except the (untouched) previous checkpoint."""
        checkpoint = Checkpoint(
            state_dict={"w": torch.ones(2)},
            autovision=CheckpointMetadata(epoch=1, global_step=1),
        )
        target = tmp_path / "last.ckpt"
        previous = Checkpoint(
            state_dict={"w": torch.zeros(2)},
            autovision=CheckpointMetadata(epoch=0, global_step=0),
        )
        previous.save(target)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("disk gone")))
            with pytest.raises(CheckpointError, match="disk gone"):
                checkpoint.save(target)
        # The previous checkpoint survived every failed attempt.
        reloaded = Checkpoint.from_path(target).state_dict
        assert list(reloaded) == ["w"]
        assert torch.equal(reloaded["w"], torch.zeros(2))
        assert not list(tmp_path.glob("*.tmp"))

    def test_save_serialized_retries_transient_write_failures(
        self, tmp_path: Path
    ) -> None:
        """The archive path shares the retry contract."""
        checkpoint = Checkpoint(
            state_dict={"w": torch.ones(2)},
            autovision=CheckpointMetadata(epoch=1, global_step=1),
        )
        payload = checkpoint.serialize()
        target = tmp_path / "epoch1.ckpt"
        real_open = Path.open
        calls = {"n": 0}

        def flaky_open(self, *args, **kwargs):
            if "wb" in args and calls["n"] < 1:
                calls["n"] += 1
                raise OSError("antivirus scan locked the file")
            return real_open(self, *args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Path, "open", flaky_open)
            checkpoint.save_serialized(target, payload)
        reloaded = Checkpoint.from_path(target).state_dict
        assert torch.equal(reloaded["w"], torch.ones(2))
