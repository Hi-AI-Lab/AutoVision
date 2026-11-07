# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pretraining resume: drift guards, scheduler two-stage restore, atomic export.

Covers the TRAIN-01..05 / 09 / 12 / 15 hardening designs. The
load-bearing invariant is that a
resume WITHOUT parameter changes reproduces the uninterrupted LR schedule
bit for bit; every behavior change (extended max_epochs, drift warnings)
is asserted on top of that anchor.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from torch.nn import Linear
from torch.utils.data import DataLoader, Dataset

from autovision.engine.engine import Engine
from autovision.errors import AutoVisionError
from autovision.optim.schedulers import CosineWarmupScheduler
from autovision.runtime import checkpoint as _checkpoint_module
from autovision.runtime.checkpoint import CheckpointMetadata

pytestmark = pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # the fixture steps a bare scheduler without an optimizer (review P3-6)



class _CosineStub(torch.nn.Module):
    """Minimal pretraining method with a real cosine schedule."""

    def __init__(self) -> None:
        super().__init__()
        self.model = Linear(4, 2)
        self.global_step = 0
        self.total_steps = 0

    def forward(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self.model(batch["x"]).sum()

    def configure_optimizers(self, total_steps: int, num_epochs: int):
        self.total_steps = total_steps
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-1)
        return optimizer, CosineWarmupScheduler(
            optimizer,
            warmup_epochs=max(1, num_epochs // 4),
            max_epochs=num_epochs,
        )

    def clip_gradients(self, optimizer) -> float | None:
        return None

    def on_before_optimizer_step(self, optimizer) -> None:
        return None

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        return None

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        return None

    def consume_log_dict(self) -> dict | None:
        return None


class _TinySet(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index):
        return {"x": torch.arange(4, dtype=torch.float32) + index}


def _loader() -> DataLoader:
    return DataLoader(_TinySet(), batch_size=2)


def _make_checkpoint(
    tmp_path: Path, max_epochs: int, epochs_trained: int
) -> tuple[Path, float]:
    """Step a max_epochs schedule `epochs_trained` times and save last.ckpt.

    Manual construction keeps the fixture deterministic: the schedule shape,
    the step counter and the fit_args snapshot are exactly what the resume
    tests reason about (a real fit would couple the schedule to its own
    max_epochs, which is the variable under test). Returns
    ``(path, lr_at_save)`` — the resumed optimizer must restore exactly
    this LR BEFORE its first post-resume step (review P3-1).
    """
    method = _CosineStub()
    optimizer, scheduler = method.configure_optimizers(
        total_steps=max_epochs * 4, num_epochs=max_epochs
    )
    for _ in range(epochs_trained):
        scheduler.step()
    lr_at_save = optimizer.param_groups[0]["lr"]
    path = tmp_path / "last.ckpt"
    Engine.save_checkpoint(
        path, method, optimizer, scheduler,
        epoch=epochs_trained - 1, global_step=epochs_trained,
        fit_args={
            "max_epochs": max_epochs,
            "steps_per_epoch": 4,
            "seed": 0,
            "precision": "32",
            "save_period_num_epochs": 0,
            "log_every_n_steps": 50,
            "data_fingerprint": None,
        },
    )
    return path, lr_at_save


def _resume_scheduler(
    path: Path, max_epochs: int, *, expected_lr_at_save: float | None = None
):
    """Resume the checkpoint's optimizer/scheduler with a given schedule.

    ``expected_lr_at_save`` pins the review P3-1 invariant: the restored
    optimizer's param-group LR equals the checkpoint's saved value BEFORE
    any post-resume scheduler step (the optimizer restore sets it; the
    scheduler restore must not clobber it).
    """
    method = _CosineStub()
    optimizer, scheduler = method.configure_optimizers(
        total_steps=max_epochs * 4, num_epochs=max_epochs
    )
    Engine._resume(path, method, optimizer, scheduler, torch.device("cpu"))
    if expected_lr_at_save is not None:
        assert optimizer.param_groups[0]["lr"] == expected_lr_at_save
    return optimizer, scheduler


class TestSchedulerTwoStageRestore:
    """TRAIN-01: the step counter rides the state dict; the schedule
    shape always comes from the CURRENT call."""

    def test_resume_with_same_args_reproduces_the_lr_trajectory(self, tmp_path):
        """Golden anchor: resuming with unchanged max_epochs must continue
        the exact LR sequence an uninterrupted schedule would produce."""
        path, lr_at_save = _make_checkpoint(tmp_path, max_epochs=6, epochs_trained=2)
        optimizer, scheduler = _resume_scheduler(
            path, max_epochs=6, expected_lr_at_save=lr_at_save
        )
        assert scheduler.max_epochs == 6

        # Reference: a fresh max_epochs=6 schedule advanced to the same
        # step (fit stepped once per epoch; the checkpoint sits at 2).
        reference_parameter = torch.nn.Parameter(torch.zeros(1))
        reference_optimizer = torch.optim.SGD([reference_parameter], lr=1e-1)
        reference = CosineWarmupScheduler(
            reference_optimizer, warmup_epochs=1, max_epochs=6
        )
        for _ in range(2):
            reference.step()

        resumed_lrs, reference_lrs = [], []
        for _ in range(4):
            scheduler.step()
            resumed_lrs.append(optimizer.param_groups[0]["lr"])
            reference.step()
            reference_lrs.append(reference_optimizer.param_groups[0]["lr"])
        # Bit-exact: the golden guarantee for unchanged-args resumes.
        assert resumed_lrs == reference_lrs

    def test_resume_with_extended_max_epochs_continues_the_decay(self, tmp_path):
        """The fix's target behavior: extending max_epochs on resume takes
        effect — the old implementation replayed the checkpoint's frozen
        max_epochs and parked the LR at end_value for the added epochs."""
        path, lr_at_save = _make_checkpoint(tmp_path, max_epochs=6, epochs_trained=2)
        optimizer, scheduler = _resume_scheduler(
            path, max_epochs=12, expected_lr_at_save=lr_at_save
        )
        assert scheduler.max_epochs == 12  # the CURRENT call governs

        # Step far past the OLD endpoint (6): the decay must still be
        # alive, not frozen at end_value.
        for _ in range(2 + 6):
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] > 0.0

        # The extended schedule must equal a fresh max_epochs=12 curve at
        # the same step, i.e. resume rewound nothing and overrode nothing.
        # The reference mirrors _resume_scheduler's construction (12//4=3
        # warmup epochs) and is pre-advanced to the SAME position the
        # resumed scheduler now sits at (checkpoint step 2 + the 8 freeze
        # check steps above).
        reference_parameter = torch.nn.Parameter(torch.zeros(1))
        reference_optimizer = torch.optim.SGD([reference_parameter], lr=1e-1)
        reference = CosineWarmupScheduler(
            reference_optimizer, warmup_epochs=3, max_epochs=12
        )
        for _ in range(2 + 8):
            reference.step()
        for _ in range(8):
            reference.step()
            scheduler.step()
            assert optimizer.param_groups[0]["lr"] == pytest.approx(
                reference_optimizer.param_groups[0]["lr"]
            )
        assert scheduler.last_epoch == reference.last_epoch == 18


class TestResumeStatePayload:
    """The identity snapshot rides every checkpoint for the resume check."""

    def test_checkpoint_carries_fit_args(self, tmp_path):
        path, _lr = _make_checkpoint(tmp_path, max_epochs=6, epochs_trained=1)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        fit_args = payload["autovision"]["engine_state"]["fit_args"]
        assert fit_args["max_epochs"] == 6
        assert "data_fingerprint" in fit_args

    def test_metadata_step_field_roundtrips_and_stays_backward_compatible(self):
        """TRAIN-09: the unambiguous ``step`` field; old payloads load."""
        metadata = CheckpointMetadata(epoch=7, global_step=7, step=7)
        restored = CheckpointMetadata.from_dict(metadata.to_dict())
        assert restored.step == 7
        legacy = CheckpointMetadata.from_dict({"epoch": 3, "global_step": 9})
        assert legacy.step is None


class TestResumeDriftGuards:
    """TRAIN-01: fit-args drift on the pretraining path is loud."""

    def test_fit_args_drift_check_emits_the_warning(self, caplog):
        """The D17 helper reports out-of-whitelist drift at warning level
        (the pretraining resume wires this helper against its snapshot)."""
        from autovision.engine.engine import check_resume_fit_args

        with caplog.at_level(logging.WARNING, logger="autovision"):
            check_resume_fit_args({"precision": "16-mixed"}, precision="32")
        assert any(
            "Resume drift" in record.getMessage() for record in caplog.records
        )

    def test_whitelisted_schedule_extension_is_info_level(self, caplog):
        """max_epochs is the documented resume extension (TRAIN-01)."""
        from autovision.engine.engine import check_resume_fit_args

        with caplog.at_level(logging.INFO, logger="autovision"):
            check_resume_fit_args({"max_epochs": 6}, max_epochs=12)
        assert not [
            record for record in caplog.records if "Resume drift" in record.getMessage()
        ]


class TestConsistencyLedger:
    """TRAIN-03: cross-rank resume divergence fails with a named error."""

    def test_single_process_is_a_noop(self):
        from autovision.engine.engine import _verify_resume_consistency

        _verify_resume_consistency(torch.device("cpu"), 1, 4, 0.5)

    def test_diverged_ranks_fail_with_filesystem_guidance(self, monkeypatch):
        import torch.distributed

        from autovision.engine.engine import _verify_resume_consistency

        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

        def diverged_all_reduce(tensor, op=None):
            # Simulate one rank at step 0 and one at step 4: MIN sees 0,
            # MAX sees 4, so the ledger must report the divergence.
            tensor.fill_(0.0 if op == torch.distributed.ReduceOp.MIN else 4.0)

        monkeypatch.setattr(torch.distributed, "all_reduce", diverged_all_reduce)
        with pytest.raises(AutoVisionError, match="shared") as excinfo:
            _verify_resume_consistency(torch.device("cpu"), 2, 4, 0.5)
        assert "start_step" in str(excinfo.value)

    def test_consistent_ranks_pass(self, monkeypatch):
        import torch.distributed

        from autovision.engine.engine import _verify_resume_consistency

        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(
            torch.distributed, "all_reduce", lambda tensor, op=None: None
        )
        _verify_resume_consistency(torch.device("cpu"), 2, 4, 0.5)


class TestFriendlyStateDictLoad:
    """TRAIN-05: a configuration-changed checkpoint names the keys."""

    def test_shape_mismatch_reports_key_diff(self):
        from autovision.engine.engine import _load_state_dict_with_context

        model = Linear(4, 2)
        other = Linear(8, 2)
        with pytest.raises(AutoVisionError, match="warm start") as excinfo:
            _load_state_dict_with_context(
                model, other.state_dict(), context="warm start"
            )
        message = str(excinfo.value)
        assert "1 shape-mismatched key(s)" in message
        assert "weight" in message

    def test_missing_and_unexpected_keys_are_counted(self):
        from autovision.engine.engine import _load_state_dict_with_context

        model = torch.nn.Sequential(Linear(4, 2), Linear(2, 2))
        checkpoint_model = Linear(4, 2)
        with pytest.raises(AutoVisionError, match="missing") as excinfo:
            _load_state_dict_with_context(
                model, checkpoint_model.state_dict(), context="resume"
            )
        assert "4 missing" in str(excinfo.value)

    def test_unrelated_runtime_error_reraises_untouched(self):
        from autovision.engine.engine import _load_state_dict_with_context

        class _Exploding(torch.nn.Module):
            def load_state_dict(self, state_dict, strict=True):
                raise RuntimeError("some unrelated failure")

        with pytest.raises(RuntimeError, match="unrelated"):
            _load_state_dict_with_context(_Exploding(), {}, context="warm start")


class TestEntryGuards:
    """TRAIN-15: batch_sampler loaders fail with guidance, not arithmetic."""

    def test_batch_size_none_loader_is_rejected(self, tmp_path):
        engine = Engine()

        class _BatchSampler:
            def __iter__(self):
                return iter([[0, 1], [2, 3]])

            def __len__(self) -> int:
                return 2

        loader = DataLoader(_TinySet(), batch_sampler=_BatchSampler())
        with pytest.raises(AutoVisionError, match="batch_size"):
            engine.fit(
                _CosineStub(), loader, out_dir=tmp_path / "run", max_epochs=1,
                devices="cpu", precision="32",
            )


class TestPretrainExportAtomicity:
    """TRAIN-02: a failed export never leaves a torn exported_last.pt."""

    def test_export_writes_through_the_atomic_path(self, tmp_path):
        from autovision.cli.train import _export_embedding_model

        class _Model:
            embed_dim = 8

            def state_dict(self):
                return {"w": torch.zeros(2)}

        path = _export_embedding_model(
            tmp_path, _Model(), model_name="m", normalize_args={"mean": 0.0}
        )
        assert path.exists()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["kind"] == "embedding_model"
        assert list(payload["state_dict"]) == ["w"]
        assert torch.equal(payload["state_dict"]["w"], torch.zeros(2))

    def test_failed_export_keeps_the_previous_file_and_no_tmp(
        self, tmp_path, monkeypatch
    ):
        from autovision.cli.train import _export_embedding_model

        monkeypatch.setattr(_checkpoint_module, "_SAVE_RETRY_BASE_DELAY_S", 0.0)

        class _Model:
            embed_dim = 8

            def state_dict(self):
                return {"w": torch.zeros(2)}

        path = _export_embedding_model(
            tmp_path, _Model(), model_name="m", normalize_args={}
        )
        previous_bytes = path.read_bytes()

        real_save = torch.save

        def failing_save(obj, target, *args, **kwargs):
            if "exported_last" in str(target):
                raise OSError("disk full")
            return real_save(obj, target, *args, **kwargs)

        monkeypatch.setattr(torch, "save", failing_save)
        with pytest.raises(Exception):
            _export_embedding_model(
                tmp_path, _Model(), model_name="m", normalize_args={}
            )
        # The previous export is byte-identical and no tmp residue remains.
        assert path.read_bytes() == previous_bytes
        assert not [
            p for p in path.parent.iterdir() if p.name.endswith(".tmp")
        ]
