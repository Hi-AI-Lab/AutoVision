# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Resume hardening and early-stop persistence contracts (design doc D17).

Covers: the patience counter surviving a resume, old checkpoints without
the persisted state starting fresh, the fit-args drift check (warning vs
whitelisted info), and the ``possible_stop`` early validation.
"""

from __future__ import annotations

import logging

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autovision.engine.engine import TaskEngine
from autovision.runtime.checkpoint import Checkpoint
from autovision.tasks.train_model import TaskStepResult, TrainModel


class _MetricModel(TrainModel):
    """Task model serving a scripted per-validation metric sequence."""

    task = "unit_test"
    watch_metric = "val_metric/map"

    def __init__(self, values: list[float]) -> None:
        super().__init__(task_model=nn.Linear(4, 2))
        self.values = values
        self.validation_calls = 0

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        return TaskStepResult(
            loss=self.model(batch["x"]).sum(),
            log_dict={"loss_component": torch.tensor(1.0)},
        )

    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        return TaskStepResult(loss=torch.ones(()), log_dict={})

    def compute_metrics(self, split: str) -> dict[str, float]:
        value = self.values[min(self.validation_calls, len(self.values) - 1)]
        self.validation_calls += 1
        return {"val_metric/map": value}

    def get_optimizer(
        self,
        total_steps: int,
        global_batch_size: int,
        train_num_batches: int = 0,
        gradient_accumulation_steps: int = 1,
    ):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        return optimizer, torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1.0
        )


class _Dataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict:
        return {"x": torch.arange(4, dtype=torch.float32) + index}


def _fit(model: _MetricModel, tmp_path, **kwargs):
    """Run ``fit_task`` with light defaults (see the early-stopping suite)."""
    loader = DataLoader(_Dataset(), batch_size=2)
    return TaskEngine().fit_task(
        model,
        loader,
        loader,
        out_dir=tmp_path,
        steps=kwargs.pop("steps", 6),
        val_every_num_steps=kwargs.pop("val_every_num_steps", 1),
        save_every_num_steps=100,
        save_last=kwargs.pop("save_last", True),
        save_best=False,
        devices="cpu",
        visualize_examples=False,
        **kwargs,
    )


class TestPatiencePersistence:
    """``no_improve`` survives the resume boundary (D17)."""

    def test_resume_continues_the_patience_counter(self, tmp_path) -> None:
        """A resumed run inherits the exhausted patience, not a fresh one."""
        first = _fit(
            _MetricModel([0.5, 0.2, 0.2, 0.2]),
            tmp_path,
            steps=6,
            early_stopping_patience=2,
        )
        assert first.stopped_early is True
        assert first.steps_run == 3  # patience exhausted within run one

        second = _fit(
            _MetricModel([0.2, 0.2, 0.2, 0.2]),
            tmp_path,
            steps=6,
            early_stopping_patience=2,
            resume_interrupted=True,
        )
        assert second.stopped_early is True
        assert second.steps_run == 1  # first post-resume validation stops it
        assert second.stop_step == 4

    def test_old_checkpoint_without_state_starts_fresh(self, tmp_path) -> None:
        """Checkpoints written before D17 resume with a zeroed counter."""
        _fit(
            _MetricModel([0.5, 0.2, 0.2, 0.2]),
            tmp_path,
            steps=6,
            early_stopping_patience=2,
        )
        ckpt_path = tmp_path / "checkpoints" / "last.ckpt"
        checkpoint = Checkpoint.from_path(ckpt_path, map_location="cpu")
        checkpoint.autovision.engine_state.pop("no_improve", None)
        checkpoint.save(ckpt_path)

        second = _fit(
            _MetricModel([0.2, 0.2, 0.2, 0.2]),
            tmp_path,
            steps=6,
            early_stopping_patience=2,
            resume_interrupted=True,
        )
        assert second.stopped_early is True
        assert second.steps_run == 2  # needs two fresh non-improvements


class TestResumeDriftCheck:
    """Fit-args drift is loud; whitelisted knobs stay info-level (D17)."""

    def test_out_of_whitelist_drift_warns(
        self, tmp_path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-schedule drift logs a warning naming both values."""
        _fit(_MetricModel([0.5, 0.2, 0.2, 0.2]), tmp_path, steps=6)
        with caplog.at_level(logging.WARNING):
            _fit(
                _MetricModel([0.2, 0.2, 0.2, 0.2]),
                tmp_path,
                steps=6,
                gradient_accumulation_steps=2,
                resume_interrupted=True,
            )
        assert "Resume drift" in caplog.text
        assert "gradient_accumulation_steps=2" in caplog.text
        assert "trained with 1" in caplog.text

    def test_whitelisted_drift_stays_info_level(
        self, tmp_path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Schedule-knob drift logs info and never a warning."""
        _fit(_MetricModel([0.5, 0.2, 0.2, 0.2]), tmp_path, steps=6)
        with caplog.at_level(logging.INFO):
            _fit(
                _MetricModel([0.2, 0.2, 0.2, 0.2]),
                tmp_path,
                steps=6,
                val_every_num_steps=2,
                resume_interrupted=True,
            )
        assert "Resuming with val_every_num_steps=2" in caplog.text
        assert "Resume drift" not in caplog.text


class TestPossibleStop:
    """The one-validation-ahead trigger (D17)."""

    def test_extra_validation_precedes_the_stop(self, tmp_path) -> None:
        """Close to the threshold, validation fires off-schedule once."""
        model = _MetricModel([0.5, 0.2, 0.2, 0.2, 0.2])
        result = _fit(
            model,
            tmp_path,
            steps=10,
            val_every_num_steps=2,
            early_stopping_patience=3,
            save_last=False,
        )
        assert result.stopped_early is True
        # Scheduled validations land at steps 2/4/6; the third non-improvement
        # arms possible_stop, so the deciding validation fires at step 7
        # instead of waiting for step 8.
        assert model.validation_calls == 4
        assert result.steps_run == 7
