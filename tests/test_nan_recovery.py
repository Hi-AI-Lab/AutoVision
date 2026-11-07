# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""NaN/Inf training self-healing: rollback recovery + EMA repair (D32/D33).

The recovery contract: a non-finite loss (or gradient norm on the non-AMP
path) rolls training back to ``last.ckpt`` — model, optimizer, scheduler,
scaler, RNG and, for the task engine, best-metric/early-stopping state —
and re-runs from its step, at most three consecutive times. A checkpoint
that is itself poisoned refuses to load. A run with finite losses is
byte-for-byte unaffected (the golden trajectory test anchors that).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from torch.nn import Linear
from torch.utils.data import DataLoader, Dataset

from autovision.engine.engine import (
    _NAN_RECOVERY_MAX_ATTEMPTS,
    Engine,
    TaskEngine,
    _assert_finite_gradients,
    _assert_state_dict_finite,
    _repair_ema_if_present,
    _sync_corruption_verdict,
    _TrainingCorruptedError,
)
from autovision.runtime import checkpoint as _checkpoint_module
from autovision.runtime.checkpoint import Checkpoint
from autovision.tasks.shared.ema import ModelEMA
from autovision.tasks.train_model import TaskStepResult, TrainModel


@pytest.fixture(autouse=True)
def _zero_retry_delays(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse checkpoint-write backoff (irrelevant to these scenarios)."""
    monkeypatch.setattr(_checkpoint_module, "_SAVE_RETRY_BASE_DELAY_S", 0.0)


class _Dataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index):
        return {"x": torch.arange(4, dtype=torch.float32) + index}


class TestCorruptionPrimitives:
    """The D32 helpers: exception typing, poisoned-checkpoint guard, verdict."""

    def test_finite_gradient_assert_raises_corrupted_error(self) -> None:
        """A NaN gradient norm surfaces as _TrainingCorruptedError, not RuntimeError."""
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        parameter = optimizer.param_groups[0]["params"][0]
        parameter.grad = torch.full_like(parameter, float("nan"))
        with pytest.raises(_TrainingCorruptedError):
            _assert_finite_gradients(optimizer, None)

    def test_assert_state_dict_finite_rejects_poisoned_checkpoint(
        self, tmp_path: Path
    ) -> None:
        """Recovery refuses to reinstall the very NaN it is escaping."""
        state = {"layer.weight": torch.tensor([1.0, float("nan")])}
        with pytest.raises(RuntimeError, match="corrupted with NaN/Inf"):
            _assert_state_dict_finite(state, tmp_path / "last.ckpt")

    def test_assert_state_dict_finite_passes_clean_and_integer_states(
        self, tmp_path: Path
    ) -> None:
        """Float-finite and non-float tensors load; non-dict input is ignored."""
        _assert_state_dict_finite(
            {"w": torch.ones(2), "counts": torch.zeros(3, dtype=torch.long)},
            tmp_path / "last.ckpt",
        )
        _assert_state_dict_finite({"nested": "not-a-tensor"}, tmp_path / "x.ckpt")

    def test_assert_state_dict_finite_walks_nested_engine_state(
        self, tmp_path: Path
    ) -> None:
        """The rollback guard covers the WHOLE payload, not just the top:
        NaN optimizer moments, EMA sidecars, and non-finite Python floats
        (best_metric) are refused; int/RNG-byte states pass."""
        with pytest.raises(RuntimeError, match="corrupted with NaN/Inf"):
            _assert_state_dict_finite(
                {"optimizer": {"state": {0: {"exp_avg": torch.zeros(3),
                                              "exp_avg_sq": torch.full((3,), float("nan"))}},
                 "param_groups": [{"lr": 0.1}]},
                 "step": 5},
                tmp_path / "last.ckpt",
            )
        with pytest.raises(RuntimeError, match="corrupted with NaN/Inf"):
            _assert_state_dict_finite(
                {"step": 1, "best_metric": float("nan")}, tmp_path / "x.ckpt"
            )
        # RNG states (byte tensors / int lists) and clean floats pass.
        _assert_state_dict_finite(
            {"rng": {"torch": torch.randint(0, 255, (8,), dtype=torch.uint8),
                     "python": [3, 1, 4]},
             "scaler": {"scale": torch.tensor([2.0])},
             "best_metric": 0.5},
            tmp_path / "y.ckpt",
        )

    @pytest.mark.parametrize(
        ("local", "expected"),
        [pytest.param(True, True, id="clean-local"), pytest.param(False, False, id="corrupt-local")],
    )
    def test_verdict_sync_passthrough_single_process(
        self, local: bool, expected: bool
    ) -> None:
        """Without a process group the local detection is the verdict."""
        assert _sync_corruption_verdict(torch.device("cpu"), 1, local) is expected


class _PretextStub(torch.nn.Module):
    """Minimal pretraining method: poisonable forward, plain SGD schedule."""

    def __init__(self) -> None:
        super().__init__()
        self.model = Linear(4, 2)
        self.global_step = 0
        self.total_steps = 0
        self.forward_calls = 0
        self.nan_forwards = 0
        #: Global-step counters at which the FIRST visit yields a NaN loss;
        #: re-visits after a rollback train normally (a transient fault).
        self.transient_nan_steps: set[int] = set()
        self._nan_fired_steps: set[int] = set()
        #: Global-step counters that are ALWAYS non-finite (a data defect).
        self.persistent_nan_steps: set[int] = set()

    def forward(self, batch: dict, batch_idx: int) -> torch.Tensor:
        self.forward_calls += 1
        loss = self.model(batch["x"]).sum()
        if (
            self.global_step in self.transient_nan_steps
            and self.global_step not in self._nan_fired_steps
        ):
            # First visit to the poisoned step only; the post-rollback
            # re-visit (restored RNG, same batch) trains normally.
            self._nan_fired_steps.add(self.global_step)
            self.nan_forwards += 1
            return torch.tensor(float("nan"), requires_grad=True)
        if self.global_step in self.persistent_nan_steps:
            self.nan_forwards += 1
            return torch.tensor(float("nan"), requires_grad=True)
        return loss

    def configure_optimizers(self, total_steps: int, num_epochs: int):
        self.total_steps = total_steps
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    def clip_gradients(self, optimizer) -> float | None:
        return None

    def on_before_optimizer_step(self, optimizer) -> None:
        return None

    def consume_log_dict(self) -> dict:
        return {}

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        return None

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        return None


class TestPretrainingRecovery:
    """Engine.fit: epoch-level rollback from last.ckpt (D32)."""

    def test_transient_nan_rolls_back_and_reruns_the_epoch(
        self, tmp_path: Path
    ) -> None:
        """A one-shot NaN in epoch 1 recovers from the epoch-0 checkpoint.

        The poisoned batch is re-encountered with restored RNG; the second
        visit trains normally, so the run completes with both epochs clean.
        """
        method = _PretextStub()
        # 2 epochs × 4 batches; poison the third step of epoch 1 (global 5).
        method.transient_nan_steps = {5}
        result = Engine().fit(
            method,
            DataLoader(_Dataset(), batch_size=1),
            out_dir=tmp_path,
            max_epochs=2,
            precision="32-true",
            devices="cpu",
            seed=0,
        )
        assert result.epochs_run == 2
        assert (tmp_path / "checkpoints" / "last.ckpt").exists()
        # Forward calls: 8 clean + 1 poisoned visit + 1 re-run visit.
        assert method.forward_calls == 10
        assert result.metrics == {}

    def test_nan_in_first_epoch_cannot_recover(self, tmp_path: Path) -> None:
        """No checkpoint exists yet — the run fails with actionable guidance."""
        method = _PretextStub()
        method.persistent_nan_steps = {0}
        with pytest.raises(RuntimeError, match="no last.ckpt exists"):
            Engine().fit(
                method,
                DataLoader(_Dataset(), batch_size=1),
                out_dir=tmp_path,
                max_epochs=2,
                precision="32-true",
                devices="cpu",
                seed=0,
            )

    def test_persistent_nan_exhausts_the_budget(self, tmp_path: Path) -> None:
        """A data-driven NaN reproduces after every rollback (restored RNG)
        and the engine declares the run dead after the allowed attempts."""
        method = _PretextStub()
        method.persistent_nan_steps = {5}
        with pytest.raises(RuntimeError, match="persisted"):
            Engine().fit(
                method,
                DataLoader(_Dataset(), batch_size=1),
                out_dir=tmp_path,
                max_epochs=2,
                precision="32-true",
                devices="cpu",
                seed=0,
            )
        # The poisoned step fires once per detection cycle: three allowed
        # recoveries plus the final detection that declares the run dead.
        assert method.nan_forwards == _NAN_RECOVERY_MAX_ATTEMPTS + 1

    def test_recovery_warning_names_the_checkpoint(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The rollback is loud: attempt count and last.ckpt path in the log."""
        method = _PretextStub()
        method.transient_nan_steps = {5}
        with caplog.at_level(logging.WARNING, logger="autovision"):
            Engine().fit(
                method,
                DataLoader(_Dataset(), batch_size=1),
                out_dir=tmp_path,
                max_epochs=2,
                precision="32-true",
                devices="cpu",
                seed=0,
            )
        warnings = [
            record for record in caplog.records if "recovery attempt" in record.message
        ]
        assert warnings, "expected a rollback warning"
        assert "last.ckpt" in warnings[0].message

    def test_resume_with_validate_finite_refuses_poisoned_file(
        self, tmp_path: Path
    ) -> None:
        """The exact guard the in-loop recovery uses: a NaN checkpoint is
        refused before any weight reaches the model (fp32 NaN passes the
        D16/D20 serialization clamps untouched, so this is reachable)."""
        path = tmp_path / "last.ckpt"
        torch.save(
            {
                "state_dict": {
                    "model.weight": torch.full((2, 4), float("nan")),
                    "model.bias": torch.zeros(2),
                },
                "autovision": {"epoch": 0, "global_step": 4, "engine_state": {}},
            },
            path,
        )
        method = _PretextStub()
        optimizer, scheduler = method.configure_optimizers(8, 2)
        with pytest.raises(RuntimeError, match="corrupted with NaN/Inf"):
            Engine._resume(
                path, method, optimizer, scheduler, torch.device("cpu"),
                validate_finite=True,
            )

    def test_resume_without_validate_finite_tolerates_poisoned_file(
        self, tmp_path: Path
    ) -> None:
        """Regular resumes keep their historical tolerance (no validation)."""
        path = tmp_path / "last.ckpt"
        torch.save(
            {
                "state_dict": {
                    "model.weight": torch.full((2, 4), float("nan")),
                    "model.bias": torch.zeros(2),
                },
                "autovision": {"epoch": 0, "global_step": 4, "engine_state": {}},
            },
            path,
        )
        method = _PretextStub()
        optimizer, scheduler = method.configure_optimizers(8, 2)
        epoch, step, _state = Engine._resume(
            path, method, optimizer, scheduler, torch.device("cpu")
        )
        assert (epoch, step) == (1, 4)


class _NanOnceModel(TrainModel):
    """Task model: first visit to a poisoned call index yields a NaN loss."""

    task = "unit_test"

    def __init__(self, poison_call: int) -> None:
        super().__init__(task_model=Linear(4, 2))
        self.calls = 0
        self.poison_call = poison_call

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        self.calls += 1
        if self.calls == self.poison_call:
            return TaskStepResult(
                loss=torch.tensor(float("nan"), requires_grad=True),
                log_dict={"loss": torch.tensor(0.0)},
            )
        return TaskStepResult(
            loss=self.model(batch["x"]).sum(),
            log_dict={"loss": self.model(batch["x"]).sum().detach()},
        )

    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        return TaskStepResult(loss=torch.ones(()), log_dict={})

    def compute_metrics(self, split: str) -> dict[str, float]:
        return {"val_metric/map": 0.5}

    def get_optimizer(
        self, total_steps: int, global_batch_size: int,
        train_num_batches: int = 0, gradient_accumulation_steps: int = 1,
    ):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


class TestTaskRecovery:
    """TaskEngine.fit_task: step-level rollback from last.ckpt (D32)."""

    def test_transient_nan_rolls_back_to_the_saved_step(
        self, tmp_path: Path
    ) -> None:
        """A NaN at step 2 (calls 0-indexed 3) rolls back to the step-1 save.

        Every step is checkpointed; the re-run covers the poisoned step and
        the run completes with the full step count.
        """
        model = _NanOnceModel(poison_call=3)
        result = TaskEngine().fit_task(
            model,
            DataLoader(_Dataset(), batch_size=1),
            None,
            out_dir=tmp_path,
            steps=4,
            save_every_num_steps=1,
            save_last=True,
            save_best=False,
            devices="cpu",
            visualize_examples=False,
        )
        assert result.steps_run == 4
        # Calls: steps 0-2 clean, step 2's first visit poisoned, then the
        # re-run of step 2 and step 3.
        assert model.calls == 5
        assert result.checkpoint_path == tmp_path / "checkpoints" / "last.ckpt"

    def test_nan_before_any_checkpoint_fails(self, tmp_path: Path) -> None:
        """With saves disabled, the first NaN has nothing to roll back to."""
        model = _NanOnceModel(poison_call=1)
        with pytest.raises(RuntimeError, match="no.*last.ckpt.*roll back"):
            TaskEngine().fit_task(
                model,
                DataLoader(_Dataset(), batch_size=1),
                None,
                out_dir=tmp_path,
                steps=4,
                save_every_num_steps=100,
                save_last=False,
                save_best=False,
                devices="cpu",
                visualize_examples=False,
            )

    def test_persistent_nan_exhausts_the_budget(self, tmp_path: Path) -> None:
        """A NaN at every step ≥ 2 reproduces after each rollback and the
        run dies with the original signal chained after attempt 3."""
        class _AlwaysNanAfterWarmup(_NanOnceModel):
            def training_step(self, batch: dict, step: int) -> TaskStepResult:
                self.calls += 1
                if self.calls >= 3:
                    return TaskStepResult(
                        loss=torch.tensor(float("nan"), requires_grad=True),
                        log_dict={"loss": torch.tensor(0.0)},
                    )
                return TaskStepResult(
                    loss=self.model(batch["x"]).sum(),
                    log_dict={"loss": self.model(batch["x"]).sum().detach()},
                )

        runaway = _AlwaysNanAfterWarmup(poison_call=99)
        runaway.calls = 0
        with pytest.raises(RuntimeError, match="persisted"):
            TaskEngine().fit_task(
                runaway,
                DataLoader(_Dataset(), batch_size=1),
                None,
                out_dir=tmp_path,
                steps=10,
                save_every_num_steps=1,
                save_last=True,
                save_best=False,
                devices="cpu",
                visualize_examples=False,
            )

    def test_clean_validations_restore_the_budget(self, tmp_path: Path) -> None:
        """Four scattered transients, each followed by a clean validation,
        ALL recover: the reset keeps every allowance full. Without the
        reset the fourth transient would exceed the budget of 3 and kill
        the run, so completion itself is the assertion of the reset."""
        class _FourTransients(_NanOnceModel):
            def __init__(self) -> None:
                super().__init__(poison_call=99)
                self._poisoned_steps: set[int] = set()

            def training_step(self, batch: dict, step: int) -> TaskStepResult:
                self.calls += 1
                if step in (2, 4, 6, 8) and step not in self._poisoned_steps:
                    # First visit to each poisoned step only; the post-
                    # rollback re-visit trains normally.
                    self._poisoned_steps.add(step)
                    return TaskStepResult(
                        loss=torch.tensor(float("nan"), requires_grad=True),
                        log_dict={"loss": torch.tensor(0.0)},
                    )
                return TaskStepResult(
                    loss=self.model(batch["x"]).sum(),
                    log_dict={"loss": self.model(batch["x"]).sum().detach()},
                )

        four = _FourTransients()
        result = TaskEngine().fit_task(
            four,
            DataLoader(_Dataset(), batch_size=1),
            DataLoader(_Dataset(), batch_size=1),
            out_dir=tmp_path,
            steps=10,
            val_every_num_steps=2,
            save_every_num_steps=1,
            save_last=True,
            save_best=False,
            devices="cpu",
            visualize_examples=False,
        )
        assert result.steps_run == 10
        # 10 step executions plus one poisoned (discarded) visit each for
        # steps 2/4/6/8; every recovery logged attempt 1/3 — the reset.
        assert four.calls == 14

    def test_scattered_transients_recover_without_a_validation_set(
        self, tmp_path: Path
    ) -> None:
        """Four scattered transients with NO validation set still all
        recover: the budget resets per clean step. (The previous reset
        point — a clean validation — never fires on this run, so the four
        attempts shared one budget and the fourth killed the run.)"""
        class _FourTransientsNoVal(_NanOnceModel):
            def __init__(self) -> None:
                super().__init__(poison_call=99)
                self._poisoned_steps: set[int] = set()

            def training_step(self, batch: dict, step: int) -> TaskStepResult:
                self.calls += 1
                if step in (2, 4, 6, 8) and step not in self._poisoned_steps:
                    self._poisoned_steps.add(step)
                    return TaskStepResult(
                        loss=torch.tensor(float("nan"), requires_grad=True),
                        log_dict={"loss": torch.tensor(0.0)},
                    )
                return TaskStepResult(
                    loss=self.model(batch["x"]).sum(),
                    log_dict={"loss": self.model(batch["x"]).sum().detach()},
                )

        four = _FourTransientsNoVal()
        result = TaskEngine().fit_task(
            four,
            DataLoader(_Dataset(), batch_size=1),
            None,
            out_dir=tmp_path,
            steps=10,
            save_every_num_steps=1,
            save_last=True,
            save_best=False,
            devices="cpu",
            visualize_examples=False,
        )
        assert result.steps_run == 10
        assert four.calls == 14

    def test_validation_nan_warns_without_rolling_back(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-finite VALIDATION loss is a measurement-side fault: it
        warns loudly, does not trigger rollback, and never becomes the
        best-metric baseline — the run itself completes."""
        class _NanValModel(_NanOnceModel):
            def validation_step(self, batch: dict, step: int) -> TaskStepResult:
                return TaskStepResult(
                    loss=torch.tensor(float("nan")), log_dict={}
                )

            def compute_metrics(self, split: str) -> dict[str, float]:
                return {"val_metric/map": float("nan")}

        model = _NanValModel(poison_call=99)
        with caplog.at_level(logging.WARNING, logger="autovision"):
            result = TaskEngine().fit_task(
                model,
                DataLoader(_Dataset(), batch_size=1),
                DataLoader(_Dataset(), batch_size=1),
                out_dir=tmp_path,
                steps=4,
                val_every_num_steps=2,
                save_every_num_steps=1,
                save_last=True,
                save_best=True,
                devices="cpu",
                visualize_examples=False,
            )
        assert result.steps_run == 4
        # Validations at steps 1 and 3, each warned once.
        assert sum(
            "Non-finite validation loss" in record.message
            for record in caplog.records
        ) == 2
        # The NaN watch value must not become the baseline: no best file.
        assert not (tmp_path / "checkpoints" / "best.ckpt").exists()


class _EmaHost(torch.nn.Module):
    """TrainModel-shaped host: ``model`` is the task model, ``ema_model``
    wraps IT (the production construction), so repair must resolve the
    task model through ``get_task_model`` to align state_dict keys."""

    def __init__(self) -> None:
        super().__init__()
        self.model = Linear(4, 2)
        self.ema_model = ModelEMA(self.model, decay=0.9, warmup_steps=0)

    def get_task_model(self):
        return self.model


class TestEmaRepair:
    """EMA poisoning repair before checkpoint writes (D33)."""

    def test_repair_resyncs_poisoned_tensors_from_live(self) -> None:
        """A NaN EMA tensor with a finite live counterpart is restored."""
        live = Linear(4, 2)
        ema = ModelEMA(live, decay=0.9, warmup_steps=0)
        with torch.no_grad():
            ema.model.weight.copy_(torch.tensor([[float("nan")] * 4] * 2))
            ema.model.bias.fill_(float("inf"))
        repaired = ema.repair_from(live)
        assert repaired == 2
        assert torch.isfinite(ema.model.weight).all()
        assert torch.isfinite(ema.model.bias).all()
        # The live model was never touched.
        assert torch.isfinite(live.weight).all()

    def test_repair_skips_when_live_has_no_clean_source(self) -> None:
        """Non-finite in BOTH copies stays untouched (no fabrication)."""
        live = Linear(4, 2)
        with torch.no_grad():
            live.weight.fill_(float("nan"))
        ema = ModelEMA(live, decay=0.9, warmup_steps=0)
        with torch.no_grad():
            ema.model.weight.fill_(float("nan"))
        assert ema.repair_from(live) == 0
        assert torch.isnan(ema.model.weight).all()

    def test_repair_returns_zero_for_a_healthy_ema(self) -> None:
        live = Linear(4, 2)
        ema = ModelEMA(live, decay=0.9, warmup_steps=0)
        ema.update(live)
        assert ema.repair_from(live) == 0

    def test_repair_fast_path_catches_low_precision_poison(self) -> None:
        """The batched-amax healthy check detects NaN/inf in fp16 tensors
        too (amax propagates NaN, inf stays inf) and still repairs them."""
        live = Linear(4, 2).half()
        ema = ModelEMA(live, decay=0.9, warmup_steps=0)
        with torch.no_grad():
            ema.model.weight.fill_(float("nan"))
            ema.model.bias.fill_(float("inf"))
        assert ema.repair_from(live) == 2
        assert bool(torch.isfinite(ema.model.weight.float()).all())
        assert bool(torch.isfinite(ema.model.bias.float()).all())

    def test_engine_helper_gates_by_allow_flag(self) -> None:
        """allow=False (FSDP) and EMA-less models are no-ops; the default
        path repairs and returns the tensor count."""
        host = _EmaHost()
        with torch.no_grad():
            host.ema_model.model.bias.fill_(float("nan"))
        assert _repair_ema_if_present(host, allow=False) == 0
        assert bool(torch.isnan(host.ema_model.model.bias).all())
        assert _repair_ema_if_present(host) == 1
        assert bool(torch.isfinite(host.ema_model.model.bias).all())
        plain = Linear(4, 2)
        assert _repair_ema_if_present(plain) == 0


class TestDDPVerdictBarrier:
    """Rank-collective alignment of the epoch-end verdict (D32, review P0).

    A rank whose epoch looked clean must STILL post the verdict all_reduce:
    if the post were conditional on the local flag, a rank-local NaN (the
    exact scenario D32 exists for) would leave the poisoned rank's verdict
    pairing with a clean rank's write barrier — every rank dies with a
    misleading "rank zero failed to write" error instead of recovering.
    These tests run one simulated rank (rank 0 of WORLD_SIZE=2, collectives
    faked with a MIN-reduce against a scripted remote verdict).
    """

    @staticmethod
    def _install_rank_zero_of_two(
        monkeypatch: pytest.MonkeyPatch, remote_verdict: int
    ) -> list[int]:
        """Simulate rank 0 of a 2-rank job; collectives pair with a fake peer."""
        import torch.distributed as distributed

        postings: list[int] = []

        def fake_all_reduce(flag, op=None, **kwargs):
            local = int(flag.item())
            postings.append(local)
            flag.fill_(min(local, remote_verdict))

        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setattr(distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(distributed, "get_world_size", lambda: 2)
        monkeypatch.setattr(distributed, "all_reduce", fake_all_reduce)
        monkeypatch.setattr(distributed, "destroy_process_group", lambda: None)
        monkeypatch.setattr(
            torch.nn.parallel, "DistributedDataParallel",
            lambda module, **kwargs: module,
        )
        return postings

    def test_verdict_posted_unconditionally_every_epoch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A healthy 2-epoch run posts verdict+barrier per epoch (4 posts).

        Under the conditional-post bug this sequence was [barrier, barrier]:
        the absence of verdict posts is exactly what desynced the ranks.
        """
        postings = self._install_rank_zero_of_two(monkeypatch, remote_verdict=1)
        result = Engine().fit(
            _PretextStub(),
            DataLoader(_Dataset(), batch_size=1),
            out_dir=tmp_path,
            max_epochs=2,
            precision="32-true",
            devices="cpu",
            seed=0,
        )
        assert result.epochs_run == 2
        assert postings == [1, 1, 1, 1]

    def test_remote_corruption_flips_clean_rank_into_recovery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A peer-rank NaN (remote verdict 0) must reach THIS rank's recovery
        path — not surface as the misleading write-barrier failure."""
        postings = self._install_rank_zero_of_two(monkeypatch, remote_verdict=0)
        with pytest.raises(RuntimeError, match="no last.ckpt exists"):
            Engine().fit(
                _PretextStub(),
                DataLoader(_Dataset(), batch_size=1),
                out_dir=tmp_path,
                max_epochs=2,
                precision="32-true",
                devices="cpu",
                seed=0,
            )
        # Exactly one collective posted (the verdict) and no barrier ever
        # followed: a clean local epoch entered the recovery path purely
        # because the peer reported NaN — the discriminator vs the old
        # conditional-post bug, whose failure was the write-barrier error.
        assert len(postings) == 1


class TestArchiveEpochResumeBoundary:
    """last.ckpt metadata epochs stay 0-based across archive saves (review P1)."""

    def test_resume_after_archive_does_not_skip_an_epoch(
        self, tmp_path: Path
    ) -> None:
        """A save_period archive at 0-based epoch 1 (file epoch2.ckpt) must
        resume into epoch 2 — a 1-based metadata value would skip it."""

        class _Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Linear(4, 2)
                self.global_step = 0
                self.total_steps = 0

            def forward(self, batch: dict, batch_idx: int) -> torch.Tensor:
                self.global_step = batch_idx
                return self.model(batch["x"]).sum()

            def configure_optimizers(self, total_steps: int, num_epochs: int):
                self.total_steps = total_steps
                optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
                return optimizer, torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda _: 1.0
                )

            def clip_gradients(self, optimizer) -> float | None:
                return None

            def on_before_optimizer_step(self, optimizer) -> None:
                return None

            def consume_log_dict(self) -> dict:
                return {}

            def on_save_checkpoint(self, checkpoint: dict) -> None:
                return None

            def on_load_checkpoint(self, checkpoint: dict) -> None:
                return None

        method = _Tiny()
        Engine().fit(
            method,
            DataLoader(_Dataset(), batch_size=2),
            out_dir=tmp_path,
            max_epochs=2,
            precision="32-true",
            devices="cpu",
            seed=0,
            save_period_num_epochs=2,
        )
        # The run ended at the archive epoch: last.ckpt metadata must say
        # epoch 1 (0-based, the epoch that just finished), so a resume — or
        # a D32 rollback — lands at epoch 2, not 3.
        last = Checkpoint.from_path(tmp_path / "checkpoints" / "last.ckpt")
        assert last.autovision.epoch == 1
        assert (tmp_path / "checkpoints" / "epoch2.ckpt").exists()


class _DecliningMetricModel(TrainModel):
    """Task model whose val metric declines per validation: the pre-NaN best
    can only survive a rollback if ``_restore_last_state`` restored it."""

    task = "unit_test"

    def __init__(self, poison_step: int) -> None:
        super().__init__(task_model=Linear(4, 2))
        self.calls = 0
        self.poison_step = poison_step
        self.validations = 0

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        self.calls += 1
        if step == self.poison_step and self.calls == self.poison_step + 1:
            return TaskStepResult(
                loss=torch.tensor(float("nan"), requires_grad=True),
                log_dict={"loss": torch.tensor(0.0)},
            )
        loss = self.model(batch["x"]).sum()
        return TaskStepResult(loss=loss, log_dict={"loss": loss.detach()})

    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        return TaskStepResult(loss=torch.ones(()), log_dict={})

    def compute_metrics(self, split: str) -> dict[str, float]:
        self.validations += 1
        return {"val_metric/map": max(0.1, 1.0 - 0.2 * (self.validations - 1))}

    def get_optimizer(
        self, total_steps: int, global_batch_size: int,
        train_num_batches: int = 0, gradient_accumulation_steps: int = 1,
    ):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


class TestRollbackRestoresBestMetric:
    """The rollback restores the best-metric baseline (review finding 8)."""

    def test_pre_nan_best_survives_the_rollback(self, tmp_path: Path) -> None:
        """Metric declines 1.0 → 0.8 → ...; a NaN at step 3 rolls back. The
        restored baseline (1.0) must keep governing best.ckpt — without the
        restore, the first post-rollback validation (0.6) would become the
        new best and the final best_metrics would say 0.6."""
        model = _DecliningMetricModel(poison_step=3)
        result = TaskEngine().fit_task(
            model,
            DataLoader(_Dataset(), batch_size=1),
            DataLoader(_Dataset(), batch_size=1),
            out_dir=tmp_path,
            steps=8,
            val_every_num_steps=2,
            save_every_num_steps=1,
            save_last=True,
            save_best=True,
            devices="cpu",
            visualize_examples=False,
        )
        # Validations: 1.0 (step 1), 0.8 (step 3) — then the NaN at step 3
        # happens AFTER that validation? No: NaN at step 3 precedes its
        # validation; rollback to the step-2 save; post-rollback vals see
        # 0.6, 0.4 — none may replace the restored 1.0 baseline.
        assert result.best_metrics["val_metric/map"] == pytest.approx(1.0)
        assert result.best_checkpoint_path == tmp_path / "checkpoints" / "best.ckpt"


class TestTaskRecoveryCheckpointFiniteness:
    """The task-engine rollback validates the checkpoint it re-enters from."""

    def test_poisoned_last_ckpt_refuses_task_recovery(self, tmp_path: Path) -> None:
        """A NaN-payload last.ckpt (reachable: D20 clamps fp16 only) aborts
        the recovery instead of reinstalling the poison."""
        model = _NanOnceModel(poison_call=2)
        checkpoints = tmp_path / "checkpoints"
        checkpoints.mkdir(parents=True)
        torch.save(
            {
                "state_dict": {
                    "model.weight": torch.full((2, 4), float("nan")),
                    "model.bias": torch.zeros(2),
                },
                "autovision": {
                    "epoch": 1,
                    "global_step": 1,
                    "engine_state": {"step": 1},
                },
            },
            checkpoints / "last.ckpt",
        )
        with pytest.raises(RuntimeError, match="corrupted with NaN/Inf"):
            TaskEngine().fit_task(
                model,
                DataLoader(_Dataset(), batch_size=1),
                None,
                out_dir=tmp_path,
                steps=4,
                save_every_num_steps=100,
                save_last=True,
                save_best=False,
                devices="cpu",
                visualize_examples=False,
            )


class TestRollbackValidatesEngineState:
    """The task-engine rollback walks the engine_state payload too: a NaN
    hiding beside clean weights (optimizer moments, EMA sidecar, best
    baseline) must abort the recovery instead of being restored silently."""

    @pytest.mark.parametrize(
        ("state_id", "engine_state"),
        [
            pytest.param(
                "optimizer-moments",
                {"optimizer": {"state": {0: {"exp_avg": torch.full((2, 4), float("nan")),
                                             "exp_avg_sq": torch.zeros((2, 4))}},
                 "param_groups": [{"lr": 1e-3}]},
                 "step": 1},
                id="optimizer",
            ),
            pytest.param(
                "ema-sidecar",
                {"ema": {"model": {"weight": torch.full((2, 4), float("nan")),
                                   "bias": torch.zeros(2)}, "updates": 3},
                 "step": 1},
                id="ema",
            ),
            pytest.param(
                "best-metric-float",
                {"step": 1, "best_metric": float("nan")},
                id="best-metric",
            ),
        ],
    )
    def test_poisoned_engine_state_refuses_task_recovery(
        self, tmp_path: Path, state_id: str, engine_state: dict
    ) -> None:
        """Clean weights + one poisoned engine_state entry → recovery aborts."""
        model = _NanOnceModel(poison_call=2)
        checkpoints = tmp_path / "checkpoints"
        checkpoints.mkdir(parents=True)
        torch.save(
            {
                "state_dict": {
                    "model.weight": torch.zeros((2, 4)),
                    "model.bias": torch.zeros(2),
                },
                "autovision": {
                    "epoch": 1,
                    "global_step": 1,
                    "engine_state": engine_state,
                },
            },
            checkpoints / "last.ckpt",
        )
        with pytest.raises(RuntimeError, match="corrupted with NaN/Inf"):
            TaskEngine().fit_task(
                model,
                DataLoader(_Dataset(), batch_size=1),
                None,
                out_dir=tmp_path,
                steps=4,
                save_every_num_steps=100,
                save_last=True,
                save_best=False,
                devices="cpu",
                visualize_examples=False,
            )


class _EmaTaskModel(TrainModel):
    """TrainModel with the production EMA construction (wraps self.model)."""

    task = "unit_test"

    def __init__(self) -> None:
        super().__init__(task_model=Linear(4, 2))
        self.ema_model = ModelEMA(self.model, decay=0.9, warmup_steps=0)

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        loss = self.model(batch["x"]).sum()
        return TaskStepResult(loss=loss, log_dict={"loss": loss.detach()})

    def on_train_batch_end(self) -> None:
        self.ema_model.update(self.model)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # The production EMA sidecar convention (rtdetr train_model).
        checkpoint["ema"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        ema_state = checkpoint.get("ema")
        if ema_state is not None:
            self.ema_model.load_state_dict(ema_state)

    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        return TaskStepResult(loss=torch.ones(()), log_dict={})

    def get_optimizer(
        self, total_steps: int, global_batch_size: int,
        train_num_batches: int = 0, gradient_accumulation_steps: int = 1,
    ):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


class TestEmaRepairWiring:
    """D33 is wired into the save path: a poisoned EMA never enters a file."""

    def test_checkpoint_ema_is_repaired_before_the_write(
        self, tmp_path: Path
    ) -> None:
        model = _EmaTaskModel()
        with torch.no_grad():
            # Poison the EMA AFTER its copy so the live model stays clean —
            # exactly the transient-NaN scenario the repair exists for.
            model.ema_model.model.bias.fill_(float("nan"))
        result = TaskEngine().fit_task(
            model,
            DataLoader(_Dataset(), batch_size=1),
            None,
            out_dir=tmp_path,
            steps=2,
            save_every_num_steps=1,
            save_last=True,
            save_best=False,
            devices="cpu",
            visualize_examples=False,
        )
        assert result.checkpoint_path is not None
        stored = Checkpoint.from_path(result.checkpoint_path)
        ema_state = stored.autovision.engine_state["ema"]["model"]
        assert torch.isfinite(ema_state["bias"]).all()
        # The live EMA object itself was repaired in place (the running
        # average continues from clean weights).
        assert torch.isfinite(model.ema_model.model.bias).all()
