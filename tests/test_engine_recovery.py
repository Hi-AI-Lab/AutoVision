# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Unit tests for D20/D21/D22 engine additions.

Covers the checkpoint archive/hygiene path (save_period serialization),
OOM detection/loader cloning (single-process recovery), and the callback
registry (registration styles + per-handler isolation).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from torch.nn import Linear
from torch.utils.data import DataLoader, Dataset

from autovision.engine.callbacks import CallbackRegistry
from autovision.engine.engine import (
    Engine,
    TaskEngine,
    _build_grad_scaler,
    _clone_loader_with_batch_size,
    _is_oom_error,
    _prune_epoch_archives,
    _sanitize_state_dict_for_serialization,
    make_dataloader_generator,
)
from autovision.runtime.checkpoint import Checkpoint, CheckpointMetadata
from autovision.tasks.train_model import TaskStepResult, TrainModel


class TestSanitizeStateDict:
    """D20 serialization hygiene: fp16 clamp + contiguity."""

    def test_fp16_overflow_clamped_on_a_clone(self) -> None:
        """Non-finite fp16 tensors are clamped; live tensors stay intact."""
        raw = torch.tensor([1.0, float("inf"), 3.0], dtype=torch.float16)
        state = {"w": raw}
        sanitized = _sanitize_state_dict_for_serialization(state)
        assert bool(torch.isfinite(sanitized["w"]).all())
        assert float(raw[1]) == float("inf")  # live tensor untouched

    def test_fp32_non_finite_left_alone(self) -> None:
        """Only fp16 tensors are clamped (fp32 handling is the NaN guard's)."""
        raw = torch.tensor([float("nan"), 1.0], dtype=torch.float32)
        state = {"w": raw}
        sanitized = _sanitize_state_dict_for_serialization(state)
        assert sanitized["w"] is raw

    def test_four_dim_weights_normalized_contiguous(self) -> None:
        """channels-last 4-d weights come back standard-contiguous."""
        raw = torch.randn(2, 3, 4, 4).to(memory_format=torch.channels_last)
        assert not raw.is_contiguous()
        sanitized = _sanitize_state_dict_for_serialization({"w": raw})
        assert sanitized["w"].is_contiguous()

    def test_non_tensors_pass_through(self) -> None:
        """Optimizer dicts and scalars survive the sanitizer untouched."""
        payload = {"optimizer": {"lr": 0.1}, "step": 3}
        assert _sanitize_state_dict_for_serialization(payload) == payload


class TestCheckpointArchive:
    """D20 single-serialize multi-write and rolling retention."""

    def test_serialize_roundtrip_matches_save(self, tmp_path: Path) -> None:
        """``serialize()`` payload loads back identical to a streamed save."""
        state = {"w": torch.arange(6.0).reshape(2, 3)}
        metadata = CheckpointMetadata(epoch=2, global_step=7)
        checkpoint = Checkpoint(state_dict=state, autovision=metadata)
        streamed = tmp_path / "streamed.ckpt"
        checkpoint.save(streamed)
        payload = checkpoint.serialize()
        archived = tmp_path / "archived.ckpt"
        checkpoint.save_serialized(archived, payload)
        loaded = Checkpoint.from_path(archived)
        assert loaded.autovision.epoch == 2
        assert loaded.autovision.global_step == 7
        assert torch.equal(loaded.state_dict["w"], state["w"])
        assert torch.equal(
            Checkpoint.from_path(streamed).state_dict["w"], state["w"]
        )

    def test_prune_epoch_archives_keeps_newest(self, tmp_path: Path) -> None:
        """Rolling retention removes oldest ``epochN`` files past ``keep``."""
        for epoch in (1, 2, 3, 4, 5):
            (tmp_path / f"epoch{epoch}.ckpt").write_bytes(b"ckpt")
        (tmp_path / "last.ckpt").write_bytes(b"kept")  # not an archive
        _prune_epoch_archives(tmp_path, keep=2)
        remaining = sorted(p.name for p in tmp_path.glob("epoch*.ckpt"))
        assert remaining == ["epoch4.ckpt", "epoch5.ckpt"]
        assert (tmp_path / "last.ckpt").exists()

    def test_prune_ignores_non_numeric_names(self, tmp_path: Path) -> None:
        """``epochX.ckpt``-style names never crash the pruning walk."""
        (tmp_path / "epochX.ckpt").write_bytes(b"weird")
        (tmp_path / "epoch3.ckpt").write_bytes(b"ckpt")
        _prune_epoch_archives(tmp_path, keep=3)
        assert (tmp_path / "epochX.ckpt").exists()

    def test_build_checkpoint_calls_hook_and_metadata(self) -> None:
        """The pretraining builder fires ``on_save_checkpoint`` and metadata.

        Hook extras (e.g. EMA sidecars) stay outside ``engine_state`` at the
        Engine level — the historical contract this builder preserves.
        """
        calls: list[dict] = []

        class _Method(Linear):
            def on_save_checkpoint(self, checkpoint: dict) -> None:
                calls.append(checkpoint)
                checkpoint["extra"] = 1  # consumed by task-level closures only

        method = _Method(2, 2)
        optimizer = torch.optim.SGD(method.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        checkpoint = Engine._build_checkpoint(
            method=method,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            global_step=3,
        )
        assert len(calls) == 1
        assert checkpoint.autovision.global_step == 3
        assert checkpoint.autovision.epoch == 1


class TestOomDetection:
    """D21 error classification and loader cloning."""

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            pytest.param(
                torch.OutOfMemoryError("CUDA out of memory"), True, id="typed-oom"
            ),
            pytest.param(
                RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling"), True,
                id="cublas",
            ),
            pytest.param(
                RuntimeError("cudnn status: CUDNN_STATUS_INTERNAL_ERROR"), True,
                id="cudnn",
            ),
            pytest.param(
                RuntimeError("unable to find an engine to execute computation"),
                True, id="engine",
            ),
            pytest.param(RuntimeError("shape mismatch"), False, id="plain-error"),
            pytest.param(ValueError("bad value"), False, id="non-runtime"),
        ],
    )
    def test_is_oom_error(self, error: Exception, expected: bool) -> None:
        """Typed OOM and the driver-message variants classify correctly."""
        assert _is_oom_error(error) is expected

    def test_clone_loader_preserves_configuration(self) -> None:
        """Cloned loaders keep dataset/collate/batch semantics, new size."""

        class _Data(Dataset):
            def __len__(self) -> int:
                return 8

            def __getitem__(self, index: int) -> int:
                return index

        def _collate(batch: list[int]) -> torch.Tensor:
            return torch.tensor(batch)

        loader = DataLoader(
            _Data(), batch_size=4, collate_fn=_collate, num_workers=0
        )
        halved = _clone_loader_with_batch_size(loader, 2)
        assert halved.batch_size == 2
        assert halved.collate_fn is _collate
        batch = next(iter(halved))
        assert batch.shape[0] == 2

    def test_clone_carries_worker_seeding(self) -> None:
        """Rebuilt loaders keep the D23a generator/worker_init_fn wiring."""

        class _Data(Dataset):
            def __len__(self) -> int:
                return 8

            def __getitem__(self, index: int) -> int:
                return index

        from autovision.cli.common_helpers import build_task_dataloader

        seeded = build_task_dataloader(
            _Data(), batch_size=4, num_workers=0, train=True, seed=5
        )
        halved = _clone_loader_with_batch_size(seeded, 2)
        assert halved.generator is seeded.generator
        assert halved.worker_init_fn is seeded.worker_init_fn


class TestOomRecoveryLoop:
    """The in-loop recovery: one scripted OOM halves the batch, run survives."""

    def test_run_recovers_from_a_scripted_oom(self, tmp_path) -> None:
        """An OOM on the first step is recovered; the run completes."""
        from tests.test_early_stopping_callbacks import _Dataset

        class _OomOnceModel(TrainModel):
            task = "unit_test"

            def __init__(self) -> None:
                super().__init__(task_model=Linear(4, 2))
                self.calls = 0
                self.batch_sizes: list[int] = []

            def training_step(self, batch: dict, step: int) -> TaskStepResult:
                self.calls += 1
                self.batch_sizes.append(int(batch["x"].shape[0]))
                if self.calls == 1:
                    raise torch.OutOfMemoryError("CUDA out of memory")
                return TaskStepResult(
                    loss=self.model(batch["x"]).sum(),
                    log_dict={"loss_component": torch.tensor(1.0)},
                )

            def validation_step(self, batch: dict, step: int) -> TaskStepResult:
                return TaskStepResult(loss=torch.ones(()), log_dict={})

            def get_optimizer(
                self, total_steps: int, global_batch_size: int,
                train_num_batches: int = 0, gradient_accumulation_steps: int = 1,
            ):
                optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
                return optimizer, torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda _: 1.0
                )

        model = _OomOnceModel()
        loader = DataLoader(_Dataset(), batch_size=2)
        result = TaskEngine().fit_task(
            model, loader, None,
            out_dir=tmp_path, steps=4, save_every_num_steps=100,
            save_last=False, save_best=False, devices="cpu",
            visualize_examples=False,
        )
        assert result.steps_run == 4
        assert 2 in model.batch_sizes  # original batch before the OOM
        assert 1 in model.batch_sizes  # halved batch after recovery

    def test_non_oom_runtime_error_still_raises(self, tmp_path) -> None:
        """Ordinary RuntimeErrors are not swallowed by the recovery path."""
        from tests.test_early_stopping_callbacks import _Dataset


        class _BoomModel(TrainModel):
            task = "unit_test"

            def __init__(self) -> None:
                super().__init__(task_model=Linear(4, 2))

            def training_step(self, batch: dict, step: int) -> TaskStepResult:
                raise RuntimeError("shape mismatch")

            def validation_step(self, batch: dict, step: int) -> TaskStepResult:
                return TaskStepResult(loss=torch.ones(()), log_dict={})

            def get_optimizer(
                self, total_steps: int, global_batch_size: int,
                train_num_batches: int = 0, gradient_accumulation_steps: int = 1,
            ):
                optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
                return optimizer, torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda _: 1.0
                )

        loader = DataLoader(_Dataset(), batch_size=2)
        with pytest.raises(RuntimeError, match="shape mismatch"):
            TaskEngine().fit_task(
                _BoomModel(), loader, None,
                out_dir=tmp_path, steps=2, save_every_num_steps=100,
                save_last=False, save_best=False, devices="cpu",
                visualize_examples=False,
            )


class TestCallbackRegistry:
    """D22 registration styles, dispatch, and per-handler isolation."""

    def test_unknown_event_rejected(self) -> None:
        """Registration to an event outside the catalogue is a typo error."""
        registry = CallbackRegistry()
        with pytest.raises(ValueError, match="Unknown callback event"):
            registry.add("on_full_moon", lambda: None)

    def test_add_set_clear_lifecycle(self) -> None:
        """add appends, set replaces, clear drops (one event or all)."""
        registry = CallbackRegistry()
        seen: list[str] = []
        registry.add("on_train_start", lambda *a: seen.append("a1"))
        registry.add("on_train_start", lambda *a: seen.append("a2"))
        registry.run("on_train_start")
        assert seen == ["a1", "a2"]
        registry.set("on_train_start", lambda *a: seen.append("s1"))
        registry.run("on_train_start")
        assert seen[-1] == "s1"
        registry.clear("on_train_start")
        registry.run("on_train_start")
        registry.add("on_step_end", lambda *a: seen.append("step"))
        registry.clear()
        registry.run("on_step_end")
        assert seen[-1] == "s1"

    def test_adapt_duck_objects(self) -> None:
        """Duck-typed objects register exactly the hooks they define."""

        class _Partial:
            def on_train_start(self, steps: int, out_dir) -> None:
                pass

        class _Silent:
            pass

        registry = CallbackRegistry()
        registry.adapt([_Partial(), _Silent()])
        assert len(registry.handlers("on_train_start")) == 1
        assert registry.handlers("on_step_end") == []

    def test_handler_failure_is_isolated(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One broken handler warns and the next one still runs."""
        seen: list[str] = []

        def _boom() -> None:
            raise RuntimeError("callback exploded")

        registry = CallbackRegistry()
        registry.add("on_validation_end", _boom)
        registry.add("on_validation_end", lambda *a: seen.append("after"))
        with caplog.at_level(logging.WARNING):
            registry.run("on_validation_end")
        assert seen == ["after"]
        assert "failed" in caplog.text

    def test_checkpoint_save_event_fires_in_fit_task(self, tmp_path) -> None:
        """``on_checkpoint_save`` reaches adapted duck callbacks (D22)."""
        from tests.test_early_stopping_callbacks import _Dataset, _MetricModel

        events: list[tuple] = []

        class _Recorder:
            def on_checkpoint_save(self, step: int, kind: str) -> None:
                events.append((step, kind))

        model = _MetricModel([0.5, 0.6, 0.7])
        loader = DataLoader(_Dataset(), batch_size=2)
        TaskEngine().fit_task(
            model, loader, loader,
            out_dir=tmp_path, steps=3, val_every_num_steps=1,
            save_every_num_steps=2, save_last=True, save_best=False,
            devices="cpu", visualize_examples=False, callbacks=[_Recorder()],
        )
        assert events and events[0][1] == "last"


class TestSavePeriodArchive:
    """D20 archive sequence end to end (serialize once, write many)."""

    def test_fit_writes_and_prunes_epoch_archives(self, tmp_path: Path) -> None:
        """``save_period_num_epochs`` produces rolling epochN archives.

        Uses a duck-typed minimal pretraining method — the repo has no
        PretextRecipe test harness, and the archive path only needs the
        fit-loop contract (forward/configure_optimizers/clip hooks).
        """

        class _TinyPretext(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.Linear(4, 2)
                self.total_steps = 0

            def forward(self, batch: dict, batch_idx: int) -> torch.Tensor:
                return self.layer(batch["x"]).sum() * 0.01

            def configure_optimizers(self, total_steps: int, num_epochs: int):
                optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)
                return optimizer, torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda _: 1.0
                )

            def clip_gradients(self, optimizer) -> None:
                return None

            def on_before_optimizer_step(self, optimizer) -> None:
                return None

            def consume_log_dict(self) -> dict:
                return {}

            def on_save_checkpoint(self, checkpoint: dict) -> None:
                return None

        class _Data(Dataset):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int) -> dict:
                return {"x": torch.randn(4)}

        from autovision.engine.engine import Engine

        result = Engine().fit(
            _TinyPretext(),
            DataLoader(_Data(), batch_size=2),
            out_dir=tmp_path,
            max_epochs=4,
            devices="cpu",
            save_period_num_epochs=2,
        )
        assert result.epochs_run == 4
        checkpoints = tmp_path / "checkpoints"
        archives = sorted(p.name for p in checkpoints.glob("epoch*.ckpt"))
        # Epochs 2 and 4 hit the period; retention keeps the newest 2.
        assert archives == ["epoch2.ckpt", "epoch4.ckpt"]
        archived = Checkpoint.from_path(checkpoints / "epoch4.ckpt")
        assert archived.autovision.epoch == 4  # 1-based, matches the name
        # On archive epochs last.ckpt shares the archive payload (the D20
        # single-serialize multi-write point).
        last = Checkpoint.from_path(checkpoints / "last.ckpt")
        assert last.autovision.epoch == archived.autovision.epoch


class TestOomRecoveryLimits:
    """D21 guard rails: attempt budget and the batch=1 floor."""

    @staticmethod
    def _run_with_repeated_oom(tmp_path, failures: int):
        from tests.test_early_stopping_callbacks import _Dataset

        class _RepeatOomModel(TrainModel):
            task = "unit_test"

            def __init__(self, failures: int) -> None:
                super().__init__(task_model=Linear(4, 2))
                self.remaining = failures

            def training_step(self, batch: dict, step: int) -> TaskStepResult:
                if self.remaining > 0:
                    self.remaining -= 1
                    raise torch.OutOfMemoryError("CUDA out of memory")
                return TaskStepResult(
                    loss=self.model(batch["x"]).sum(),
                    log_dict={"loss_component": torch.tensor(1.0)},
                )

            def validation_step(self, batch: dict, step: int) -> TaskStepResult:
                return TaskStepResult(loss=torch.ones(()), log_dict={})

            def get_optimizer(
                self, total_steps: int, global_batch_size: int,
                train_num_batches: int = 0, gradient_accumulation_steps: int = 1,
            ):
                optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
                return optimizer, torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda _: 1.0
                )

        model = _RepeatOomModel(failures)
        loader = DataLoader(_Dataset(), batch_size=2)
        result = TaskEngine().fit_task(
            model, loader, None,
            out_dir=tmp_path, steps=5, save_every_num_steps=100,
            save_last=False, save_best=False, devices="cpu",
            visualize_examples=False,
        )
        return model, result

    def test_three_failures_exhaust_the_budget_and_raise(self, tmp_path) -> None:
        """After _OOM_MAX_ATTEMPTS recoveries the original error surfaces."""
        with pytest.raises(torch.OutOfMemoryError):
            self._run_with_repeated_oom(tmp_path, failures=10)

    def test_single_failure_still_completes(self, tmp_path) -> None:
        """One OOM inside the budget recovers and finishes the run."""
        model, result = self._run_with_repeated_oom(tmp_path, failures=1)
        assert result.steps_run == 5


class TestDataloaderSeeding:
    """D23a generator/worker_init_fn wiring on the shared builders."""

    def test_seed_attaches_generator_and_init_fn(self) -> None:
        """``seed=...`` wires reproducible workers; ``None`` stays clean."""
        from autovision.cli.common_helpers import build_task_dataloader

        class _Data(Dataset):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int) -> int:
                return index

        seeded = build_task_dataloader(
            _Data(), batch_size=2, num_workers=0, train=True, seed=7
        )
        assert seeded.generator is not None
        assert seeded.worker_init_fn is not None
        plain = build_task_dataloader(_Data(), batch_size=2, num_workers=0, train=True)
        assert plain.generator is None
        assert plain.worker_init_fn is None

    def test_generator_seed_deterministic(self) -> None:
        """Same run seed -> same generator stream (per-rank math aside)."""
        assert (
            make_dataloader_generator(11).initial_seed()
            == make_dataloader_generator(11).initial_seed()
        )

    def test_grad_scaler_untouched_by_d20(self) -> None:
        """Guard against accidental coupling: scaler builder unchanged."""
        assert _build_grad_scaler("32-true", "cpu") is None
