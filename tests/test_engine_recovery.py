# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Unit tests for D20/D21/D22 engine additions and H1-4/H3-3a/H3-3b/H3-3d/
H3-3f/H3-3g fixes.

Covers the checkpoint archive/hygiene path (save_period serialization),
OOM detection/loader cloning (single-process recovery), the callback
registry (registration styles + per-handler isolation), the DDP loader
rebuild / seed wiring (H1-4), the sampler-epoch resume plumbing (H3-3b),
the ranked validation-loss merge (H3-3d), the per-rank RNG checkpoint
shards (H3-3a), the dataset-drift resume check (H3-3g) and the alerting
step watchdog (H3-3f).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

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
    _dataset_fingerprint,
    _InfiniteCycleIterator,
    _is_oom_error,
    _merge_ranked_val_loss,
    _prune_epoch_archives,
    _rng_state_for_this_rank,
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
        # Metadata epochs are 0-BASED everywhere (the resume/rollback
        # convention adds 1); only the epoch{N}.ckpt filename is 1-based.
        assert archived.autovision.epoch == 3
        # On archive epochs last.ckpt shares the archive payload (the D20
        # single-serialize multi-write point) — so a resume/rollback from
        # last.ckpt lands on the next epoch, never skipping one.
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

    def test_generator_rank_offset_changes_the_stream(self) -> None:
        """H1-4: same seed, two ranks -> different streams; same rank stable."""
        rank0 = make_dataloader_generator(123, rank=0)
        rank1 = make_dataloader_generator(123, rank=1)
        assert rank0.initial_seed() != rank1.initial_seed()
        assert (
            make_dataloader_generator(123, rank=1).initial_seed()
            == rank1.initial_seed()
        )
        sample0 = torch.rand(8, generator=rank0)
        sample1 = torch.rand(8, generator=rank1)
        assert not torch.equal(sample0, sample1)

    def test_grad_scaler_untouched_by_d20(self) -> None:
        """Guard against accidental coupling: scaler builder unchanged."""
        assert _build_grad_scaler("32-true", "cpu") is None


# ── shared fixtures for the H1-4 / H3-3b / H3-3d test classes ──────────


class _SeedProbeDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict:
        return {"x": torch.arange(4, dtype=torch.float32) + index}


class _SeedProbePretext(torch.nn.Module):
    """Duck-typed minimal pretraining method (fit-loop contract only)."""

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


class _PlainTaskModel(TrainModel):
    task = "unit_test"

    def __init__(self) -> None:
        super().__init__(task_model=Linear(4, 2))

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
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


def _mock_two_rank_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub a 2-rank process group for single-process engine checks.

    The engine paths under test only query rank/world and post the verdict
    and write-barrier collectives; a passthrough DDP wrapper and no-op
    all_reduce keep the run deterministic and in-process.
    """
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, *a, **k: None)
    monkeypatch.setattr(
        torch.distributed, "destroy_process_group", lambda *a, **k: None
    )
    monkeypatch.setattr(
        torch.nn.parallel, "DistributedDataParallel", lambda model, **kw: model
    )


class TestWithSamplerFieldCopy:
    """H1-4: the shared rebuild helper carries every per-rank field."""

    def test_rebuilt_loader_keeps_worker_and_seeding_fields(self) -> None:
        """worker_init_fn/generator/prefetch_factor/persistent_workers pass
        through; drop_last follows the source loader unless overridden.

        Constructing a DataLoader never spawns workers (that happens at the
        first ``iter()``), so ``num_workers=2`` attributes are safe here.
        """
        dataset = _SeedProbeDataset()
        generator = torch.Generator().manual_seed(7)

        def _worker_init(worker_id: int) -> None:
            return None

        loader = DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=2,
            worker_init_fn=_worker_init, generator=generator,
            prefetch_factor=3, persistent_workers=True,
        )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True, num_replicas=1, rank=0
        )
        rebuilt = Engine._with_sampler(loader, sampler)
        assert rebuilt.sampler is sampler
        assert rebuilt.worker_init_fn is _worker_init
        assert rebuilt.generator is generator
        assert rebuilt.prefetch_factor == 3
        assert rebuilt.persistent_workers is True
        assert rebuilt.num_workers == 2
        assert rebuilt.drop_last is False  # taken from the source loader
        assert Engine._with_sampler(loader, sampler, drop_last=True).drop_last


class TestFitDdpEpochLoaderRebuild:
    """H1-4: Engine.fit's DDP per-epoch rebuild keeps the seed wiring."""

    def test_rebuild_carries_seeding_fields_and_run_seed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rebuilt epoch_loader receives worker_init_fn/generator/
        prefetch_factor/persistent_workers from the injected loader and a
        DistributedSampler seeded with the run seed.

        Single-process, mocked-distributed check (no real data, no gloo):
        the distributed bits are stubbed and DataLoader construction is
        recorded instead of executed with worker processes.
        """
        import autovision.engine.engine as engine_mod

        generator = torch.Generator().manual_seed(99)

        def _worker_init(worker_id: int) -> None:
            return None

        loader = DataLoader(
            _SeedProbeDataset(), batch_size=2, shuffle=True, num_workers=2,
            worker_init_fn=_worker_init, generator=generator,
            prefetch_factor=3, persistent_workers=True,
        )
        _mock_two_rank_group(monkeypatch)

        recorded: list[dict] = []
        real_data_loader = engine_mod.DataLoader

        def _recording_data_loader(*args: Any, **kwargs: Any) -> DataLoader:
            recorded.append(dict(kwargs))
            # Run the epoch in-process: num_workers=0 forbids the worker
            # pool fields; the assertions target the recorded kwargs.
            shim = dict(kwargs)
            shim["num_workers"] = 0
            shim["prefetch_factor"] = None
            shim["persistent_workers"] = False
            return real_data_loader(*args, **shim)

        monkeypatch.setattr(engine_mod, "DataLoader", _recording_data_loader)

        result = Engine().fit(
            _SeedProbePretext(), loader,
            out_dir=tmp_path, max_epochs=1, devices="cpu", seed=42,
        )
        assert result.epochs_run == 1
        assert len(recorded) == 1
        kwargs = recorded[0]
        assert kwargs["worker_init_fn"] is _worker_init
        assert kwargs["generator"] is generator
        assert kwargs["prefetch_factor"] == 3
        assert kwargs["persistent_workers"] is True
        assert kwargs["num_workers"] == 2
        assert kwargs["drop_last"] is True
        sampler = kwargs["sampler"]
        assert isinstance(
            sampler, torch.utils.data.distributed.DistributedSampler
        )
        assert sampler.seed == 42
        assert sampler.num_replicas == 2


class TestCycleIteratorStartEpoch:
    """H3-3b: the cycle iterator continues the saved sampler epoch."""

    def test_start_epoch_syncs_the_sampler(self) -> None:
        """``start_epoch=N`` seeds both the counter and the sampler epoch."""
        dataset = _SeedProbeDataset()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True, num_replicas=1, rank=0
        )
        loader = DataLoader(dataset, batch_size=2, sampler=sampler, drop_last=True)
        iterator = _InfiniteCycleIterator(loader, sampler=sampler, start_epoch=3)
        assert iterator._epoch == 3
        assert sampler.epoch == 3
        next(iterator)  # 4 items / batch 2: consume the first pass
        next(iterator)
        next(iterator)  # StopIteration -> restart advances to epoch 4
        assert iterator._epoch == 4
        assert sampler.epoch == 4

    def test_default_start_does_not_touch_the_sampler(self) -> None:
        """A default (epoch 0) start behaves exactly as before the fix."""
        recorded: list[int] = []

        class _RecordingSampler:
            def set_epoch(self, epoch: int) -> None:
                recorded.append(epoch)

        loader = DataLoader(_SeedProbeDataset(), batch_size=2)
        iterator = _InfiniteCycleIterator(loader, sampler=_RecordingSampler())
        assert iterator._epoch == 0
        assert recorded == []  # no spurious set_epoch(0) at construction
        iterator.reset()
        assert recorded == [1]


class TestResumeSamplerEpoch:
    """H3-3b: resume restores the iterator's sampler epoch from state.

    Runs ``fit_task`` under the mocked 2-rank group: only with a real
    (distributed) sampler does the iterator's epoch counter advance, which
    is the value the checkpoint must carry. Single-process runs have no
    sampler, so their epoch stays 0 by contract.
    """

    @staticmethod
    def _run(model, tmp_path: Path, *, steps: int, **kwargs):
        loader = DataLoader(_SeedProbeDataset(), batch_size=2)
        return TaskEngine().fit_task(
            model, loader, None,
            out_dir=tmp_path, steps=steps, save_every_num_steps=100,
            save_last=True, save_best=False, devices="cpu",
            visualize_examples=False, **kwargs,
        )

    @staticmethod
    def _capture_iterator_start(
        monkeypatch: pytest.MonkeyPatch,
    ) -> list[int]:
        import autovision.engine.engine as engine_mod

        captured: list[int] = []
        real_iterator = engine_mod._InfiniteCycleIterator

        def _recording_iterator(loader, sampler=None, start_epoch=0):
            captured.append(int(start_epoch))
            return real_iterator(loader, sampler=sampler, start_epoch=start_epoch)

        monkeypatch.setattr(engine_mod, "_InfiniteCycleIterator", _recording_iterator)
        return captured

    def test_resume_continues_the_saved_sampler_epoch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under the mocked 2-rank sampler (1 batch per cycle) a 5-step run
        advances the iterator to epoch 4; the resume rebuilds the iterator
        at exactly that epoch."""
        _mock_two_rank_group(monkeypatch)
        model = _PlainTaskModel()
        self._run(model, tmp_path, steps=5)
        last = tmp_path / "checkpoints" / "last.ckpt"
        saved_epoch = int(
            Checkpoint.from_path(last).autovision.engine_state["sampler_epoch"]
        )
        assert saved_epoch == 4  # 5 steps x 1 batch/cycle -> 4 restarts

        captured = self._capture_iterator_start(monkeypatch)
        self._run(model, tmp_path, steps=6, resume_interrupted=True)
        assert captured == [saved_epoch]

    def test_old_checkpoint_without_the_key_resumes_at_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A checkpoint written before ``sampler_epoch`` existed resumes at
        epoch 0 instead of erroring (schema-version compatibility)."""
        _mock_two_rank_group(monkeypatch)
        model = _PlainTaskModel()
        self._run(model, tmp_path, steps=5)
        last = tmp_path / "checkpoints" / "last.ckpt"
        loaded = Checkpoint.from_path(last)
        payload = {
            "state_dict": loaded.state_dict,
            "autovision": loaded.autovision.to_dict(),
        }
        payload["autovision"]["engine_state"].pop("sampler_epoch", None)
        torch.save(payload, last)

        captured = self._capture_iterator_start(monkeypatch)
        self._run(model, tmp_path, steps=6, resume_interrupted=True)
        assert captured == [0]


class TestRankedValLossMerge:
    """H3-3d: empty val shards must not dilute the global mean.

    Note: there is no torchrun/gloo harness in this file (multi-process DDP
    lives in the gated tests/test_torchrun_distributed.py), so the cross-rank
    reduction is simulated: the helper reduces the numerator then the
    denominator, and the stub returns pre-computed cross-rank totals per
    call in that order.
    """

    @staticmethod
    def _simulate_peers(monkeypatch: pytest.MonkeyPatch, totals: list[float]) -> None:
        stream = iter(totals)

        def _stub_all_reduce(tensor, *args, **kwargs):
            tensor.fill_(next(stream))

        monkeypatch.setattr(torch.distributed, "all_reduce", _stub_all_reduce)

    def test_empty_shard_does_not_dilute_the_mean(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """rank0 = 3.0 over 2 batches, rank1 = 0 batches -> mean 1.5 (the
        old world_size division would have reported 0.75)."""
        self._simulate_peers(monkeypatch, [3.0, 2.0])
        merged = _merge_ranked_val_loss(3.0, 2, torch.device("cpu"))
        assert merged == pytest.approx(1.5)

    def test_balanced_shards_match_the_plain_mean(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Equal batch counts per rank reduce to the classic mean."""
        self._simulate_peers(monkeypatch, [4.0, 2.0])
        merged = _merge_ranked_val_loss(2.0, 1, torch.device("cpu"))
        assert merged == pytest.approx(2.0)

    def test_all_shards_empty_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every rank empty -> 0.0 without a ZeroDivisionError."""
        self._simulate_peers(monkeypatch, [0.0, 0.0])
        merged = _merge_ranked_val_loss(0.0, 0, torch.device("cpu"))
        assert merged == 0.0


# ── H3-3a: per-rank RNG shards in checkpoints ──────────────────────────


class TestRngShardSelection:
    """H3-3a: each rank must resume its OWN RNG shard from engine_state."""

    @staticmethod
    def _mock_rank(monkeypatch: pytest.MonkeyPatch, rank: int, world: int) -> None:
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world)

    def test_each_rank_selects_its_own_shard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """rank 0 reads shard 0, rank 1 reads shard 1 of the same state."""
        state = {
            "rng": "legacy-stream",
            "rng_per_rank": ["shard-0", "shard-1"],
        }
        self._mock_rank(monkeypatch, rank=0, world=2)
        assert _rng_state_for_this_rank(state) == "shard-0"
        self._mock_rank(monkeypatch, rank=1, world=2)
        assert _rng_state_for_this_rank(state) == "shard-1"

    def test_shard_count_mismatch_falls_back_to_the_legacy_stream(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A checkpoint written under a different world size warns and
        restores the shared stream instead of a mismatched shard."""
        state = {"rng": "legacy-stream", "rng_per_rank": ["s0", "s1", "s2"]}
        self._mock_rank(monkeypatch, rank=1, world=2)
        with caplog.at_level(logging.WARNING, logger="autovision"):
            assert _rng_state_for_this_rank(state) == "legacy-stream"
        assert any("RNG shards" in record.message for record in caplog.records)

    def test_no_shards_returns_the_legacy_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Old checkpoints (no ``rng_per_rank``) keep the old behavior."""
        self._mock_rank(monkeypatch, rank=0, world=2)
        assert _rng_state_for_this_rank({"rng": "legacy-stream"}) == "legacy-stream"

    def test_outside_a_process_group_returns_the_legacy_stream(self) -> None:
        """Single-process resume of a multi-rank checkpoint is safe."""
        assert not torch.distributed.is_initialized()
        state = {"rng": "legacy-stream", "rng_per_rank": ["s0", "s1"]}
        assert _rng_state_for_this_rank(state) == "legacy-stream"


def _mock_two_rank_group_with_rng_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extend the 2-rank mock with a per-rank RNG gather stub (H3-3a).

    ``all_gather_object`` receives rank 0's own capture and fabricates a
    DISTINCT shard for the peer rank, so the saved shards are provably not
    two copies of one stream.
    """
    _mock_two_rank_group(monkeypatch)

    def fake_all_gather_object(
        gathered: list[Any], value: Any, *args: Any, **kwargs: Any
    ) -> None:
        gathered[0] = value
        gathered[1] = {"torch": torch.ByteTensor(list(range(16)))}

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)


class TestPerRankRngCheckpoint:
    """H3-3a: checkpoints carry one RNG shard per rank, restored per rank."""

    @staticmethod
    def _run(model, tmp_path: Path, steps: int = 2, **kwargs):
        loader = DataLoader(_SeedProbeDataset(), batch_size=2)
        return TaskEngine().fit_task(
            model, loader, None,
            out_dir=tmp_path, steps=steps, save_every_num_steps=100,
            save_last=True, save_best=False, devices="cpu",
            visualize_examples=False, **kwargs,
        )

    def test_task_checkpoint_carries_two_distinct_shards(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mocked 2-rank save stores ``rng_per_rank`` with DIFFERENT shards
        while the legacy ``rng`` key stays for backward compatibility."""
        _mock_two_rank_group_with_rng_gather(monkeypatch)
        self._run(_PlainTaskModel(), tmp_path)
        engine_state = Checkpoint.from_path(
            tmp_path / "checkpoints" / "last.ckpt"
        ).autovision.engine_state
        shards = engine_state["rng_per_rank"]
        assert isinstance(shards, list) and len(shards) == 2
        assert not torch.equal(shards[0]["torch"], shards[1]["torch"])
        assert "rng" in engine_state  # legacy key preserved

    def test_pretraining_checkpoint_carries_rng_per_rank(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``Engine.fit`` gathers through the same helper (archive-free save)."""
        _mock_two_rank_group_with_rng_gather(monkeypatch)
        Engine().fit(
            _SeedProbePretext(),
            DataLoader(_SeedProbeDataset(), batch_size=2),
            out_dir=tmp_path, max_epochs=1, devices="cpu",
        )
        engine_state = Checkpoint.from_path(
            tmp_path / "checkpoints" / "last.ckpt"
        ).autovision.engine_state
        shards = engine_state["rng_per_rank"]
        assert len(shards) == 2
        assert not torch.equal(shards[0]["torch"], shards[1]["torch"])

    def test_resume_restores_this_ranks_own_shard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resuming as rank 1 restores shard 1, not the shared stream."""
        import autovision.engine.engine as engine_mod

        _mock_two_rank_group_with_rng_gather(monkeypatch)
        model = _PlainTaskModel()
        self._run(model, tmp_path)
        saved = Checkpoint.from_path(
            tmp_path / "checkpoints" / "last.ckpt"
        ).autovision.engine_state["rng_per_rank"]

        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
        restored: list[Any] = []
        monkeypatch.setattr(
            engine_mod, "_restore_rng_state", lambda state: restored.append(state)
        )
        self._run(model, tmp_path, resume_interrupted=True, steps=3)
        assert len(restored) == 1
        assert torch.equal(restored[0]["torch"], saved[1]["torch"])

    def test_checkpoint_without_rng_per_rank_still_resumes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An old checkpoint (no shard list) resumes via the legacy key."""
        _mock_two_rank_group_with_rng_gather(monkeypatch)
        model = _PlainTaskModel()
        self._run(model, tmp_path)
        last = tmp_path / "checkpoints" / "last.ckpt"
        loaded = Checkpoint.from_path(last)
        payload = {
            "state_dict": loaded.state_dict,
            "autovision": loaded.autovision.to_dict(),
        }
        payload["autovision"]["engine_state"].pop("rng_per_rank", None)
        torch.save(payload, last)

        result = self._run(model, tmp_path, resume_interrupted=True, steps=3)
        assert result.steps_run == 1  # steps 2 -> 3 from the saved step


# ── H3-3f: alerting-only per-step watchdog ──────────────────────────────


class TestStepTimeoutWatchdog:
    """H3-3f: AUTOVISION_STEP_TIMEOUT_SEC warns once per over-threshold
    entry and never cancels the step."""

    @staticmethod
    def _run(model, tmp_path: Path, steps: int = 4):
        loader = DataLoader(_SeedProbeDataset(), batch_size=2)
        return TaskEngine().fit_task(
            model, loader, None,
            out_dir=tmp_path, steps=steps, save_every_num_steps=100,
            save_last=False, save_best=False, devices="cpu",
            visualize_examples=False,
        )

    def test_warning_fires_on_entering_the_over_threshold_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Slow steps 0 and 2 around fast steps: exactly one warning per
        over-threshold ENTRY (edge-triggered, no per-step spam), and the
        message says the step was not cancelled."""
        monkeypatch.setenv("AUTOVISION_STEP_TIMEOUT_SEC", "0.02")

        class _Sleeper(_PlainTaskModel):
            def training_step(self, batch: dict, step: int) -> TaskStepResult:
                if step in (0, 2):
                    time.sleep(0.05)
                return super().training_step(batch, step)

        with caplog.at_level(logging.WARNING, logger="autovision"):
            result = self._run(_Sleeper(), tmp_path)
        assert result.steps_run == 4  # the watchdog never cancels a step
        warnings = [
            record for record in caplog.records if "watchdog" in record.message
        ]
        assert len(warnings) == 2  # entries at steps 0 and 2 only
        assert "Step 1 took" in warnings[0].message  # 1-based step numbering
        assert "NOT cancelled" in warnings[0].message

    def test_fast_steps_stay_silent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Steps well under the threshold never warn."""
        monkeypatch.setenv("AUTOVISION_STEP_TIMEOUT_SEC", "60")
        with caplog.at_level(logging.WARNING, logger="autovision"):
            self._run(_PlainTaskModel(), tmp_path)
        assert not [
            record for record in caplog.records if "watchdog" in record.message
        ]


# ── H3-3g: dataset-drift detection on resume ────────────────────────────


class _SpecProbeDataset(_SeedProbeDataset):
    """Probe dataset carrying a classification-style filenames spec."""

    def __init__(self, filenames: list[str]) -> None:
        self.filenames = list(filenames)


class TestDatasetFingerprint:
    """H3-3g: the data-spec hash is stable across processes and spec only."""

    def test_same_spec_gives_the_same_fingerprint(self) -> None:
        """Fresh instances with equal content hash identically."""
        first = _dataset_fingerprint(_SpecProbeDataset(["a.png", "b.png"]))
        second = _dataset_fingerprint(_SpecProbeDataset(["a.png", "b.png"]))
        assert first == second
        assert first is not None

    def test_changed_spec_changes_the_fingerprint(self) -> None:
        """A swapped filename produces a different digest."""
        assert _dataset_fingerprint(
            _SpecProbeDataset(["a.png", "b.png"])
        ) != _dataset_fingerprint(_SpecProbeDataset(["a.png", "c.png"]))

    def test_paths_normalize_to_strings(self) -> None:
        """Path-based and string-based specs of the same data match."""

        class _PathPairs:
            pairs = [(Path("img/a.png"), Path("mask/a.png"))]

        class _StrPairs:
            pairs = [["img/a.png", "mask/a.png"]]

        assert _dataset_fingerprint(_PathPairs()) == _dataset_fingerprint(
            _StrPairs()
        )

    def test_datasets_without_a_spec_return_none(self) -> None:
        """Duck datasets disable the drift check (backward compatible)."""
        assert _dataset_fingerprint(_SeedProbeDataset()) is None

    def test_unserializable_attributes_are_dropped_not_reprd(self) -> None:
        """An address-bearing object in one attr never enters the hash —
        reprs differ per process and would false-alarm every resume."""
        fingerprint = _dataset_fingerprint(_SeedProbeDataset())
        assert fingerprint is None  # no spec attrs at all

        class _BadPair:
            pairs = [object()]  # unserializable → dropped

        assert _dataset_fingerprint(_BadPair()) is None


class TestResumeDatasetDrift:
    """H3-3g: resuming against a changed dataset warns; equal data is quiet."""

    @staticmethod
    def _run(model, tmp_path: Path, dataset, **kwargs):
        loader = DataLoader(dataset, batch_size=2)
        return TaskEngine().fit_task(
            model, loader, None,
            out_dir=tmp_path, steps=2, save_every_num_steps=100,
            save_last=True, save_best=False, devices="cpu",
            visualize_examples=False, **kwargs,
        )

    def test_resume_with_a_changed_dataset_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A swapped filename between checkpoint and resume is loud."""
        model = _PlainTaskModel()
        self._run(model, tmp_path, _SpecProbeDataset(["a.png", "b.png", "c.png", "d.png"]))
        with caplog.at_level(logging.WARNING, logger="autovision"):
            self._run(
                model, tmp_path,
                _SpecProbeDataset(["a.png", "b.png", "c.png", "e.png"]),
                resume_interrupted=True,
            )
        assert any(
            "Resume dataset drift" in record.message
            for record in caplog.records
        )

    def test_resume_with_the_same_dataset_is_quiet(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Equal data specs produce no drift warning."""
        model = _PlainTaskModel()
        filenames = ["a.png", "b.png", "c.png", "d.png"]
        self._run(model, tmp_path, _SpecProbeDataset(filenames))
        with caplog.at_level(logging.WARNING, logger="autovision"):
            self._run(
                model, tmp_path, _SpecProbeDataset(filenames),
                resume_interrupted=True,
            )
        assert not any(
            "Resume dataset drift" in record.message
            for record in caplog.records
        )

    def test_datasets_without_a_spec_skip_the_check(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Fingerprint-less datasets (old harnesses) never warn."""
        model = _PlainTaskModel()
        self._run(model, tmp_path, _SeedProbeDataset())
        with caplog.at_level(logging.WARNING, logger="autovision"):
            self._run(model, tmp_path, _SeedProbeDataset(), resume_interrupted=True)
        assert not any(
            "Resume dataset drift" in record.message
            for record in caplog.records
        )
