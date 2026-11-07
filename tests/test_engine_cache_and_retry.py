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
from torch.utils.data import DataLoader

from autovision.engine.engine import TaskEngine, _maybe_clear_accelerator_cache
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


class TestExportWeightsAtomicity:
    """H1-5: export_weights publishes via tmp+fsync+os.replace.

    A failed export write must surface (through the rank-zero write barrier
    in DDP) while the previous artifacts and any tmp files stay intact.
    """

    @staticmethod
    def _fit_with_exports(tmp_path: Path, **kwargs):
        from tests.test_early_stopping_callbacks import _Dataset, _MetricModel

        model = _MetricModel([0.5, 0.6, 0.7])
        loader = DataLoader(_Dataset(), batch_size=2)
        return model, TaskEngine().fit_task(
            model, loader, loader,
            out_dir=tmp_path, steps=3, val_every_num_steps=1,
            save_every_num_steps=100, save_last=True, save_best=True,
            devices="cpu", visualize_examples=False, **kwargs,
        )

    def test_successful_export_publishes_a_complete_payload(
        self, tmp_path: Path
    ) -> None:
        """The happy path still writes a loadable payload, without tmp residue."""
        _, result = self._fit_with_exports(tmp_path)
        exported = result.exported_model_path
        assert exported is not None and exported.exists()
        payload = torch.load(exported, weights_only=True)
        assert payload["kind"] == "task_model"
        assert payload["schema_version"] == 1
        assert "train_model" in payload
        assert not list((tmp_path / "exported_models").glob("*.tmp"))

    def test_failed_export_keeps_previous_artifacts_and_no_tmp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A disk failure during the export write surfaces as CheckpointError
        after the atomic-write retries; previous artifacts keep their bytes
        and no tmp file survives (checkpoint writes are unaffected)."""
        from tests.test_early_stopping_callbacks import _Dataset, _MetricModel

        export_dir = tmp_path / "exported_models"
        export_dir.mkdir(parents=True)
        (export_dir / "exported_best.pt").write_bytes(b"old-best-bytes")
        (export_dir / "exported_last.pt").write_bytes(b"old-last-bytes")

        real_save = torch.save

        def _exploding_save(obj, path, *args, **kwargs):
            # Only the task-model export payload fails; the Checkpoint
            # payloads (no "kind" key) go through untouched.
            if isinstance(obj, dict) and obj.get("kind") == "task_model":
                raise OSError("disk full")
            return real_save(obj, path, *args, **kwargs)

        monkeypatch.setattr(torch, "save", _exploding_save)

        model = _MetricModel([0.5, 0.6, 0.7])
        loader = DataLoader(_Dataset(), batch_size=2)
        with pytest.raises(CheckpointError, match="disk full"):
            TaskEngine().fit_task(
                model, loader, loader,
                out_dir=tmp_path, steps=3, val_every_num_steps=1,
                save_every_num_steps=100, save_last=True, save_best=True,
                devices="cpu", visualize_examples=False,
            )
        assert (export_dir / "exported_best.pt").read_bytes() == b"old-best-bytes"
        assert (export_dir / "exported_last.pt").read_bytes() == b"old-last-bytes"
        assert not list(export_dir.glob("*.tmp"))
        assert (tmp_path / "checkpoints" / "best.ckpt").exists()

    def test_early_stop_export_still_publishes(self, tmp_path: Path) -> None:
        """The restructured early-stop export (try/except + barrier) still
        publishes the last artifact when stopping before the final step."""
        from tests.test_early_stopping_callbacks import _Dataset, _MetricModel

        model = _MetricModel([0.5] * 10)
        loader = DataLoader(_Dataset(), batch_size=2)
        result = TaskEngine().fit_task(
            model, loader, loader,
            out_dir=tmp_path, steps=6, val_every_num_steps=1,
            save_every_num_steps=100, save_last=True, save_best=False,
            devices="cpu", visualize_examples=False,
            early_stopping_patience=2,
        )
        assert result.stopped_early is True
        assert result.exported_model_path is not None
        assert result.exported_model_path.exists()
        assert not list((tmp_path / "exported_models").glob("*.tmp"))
