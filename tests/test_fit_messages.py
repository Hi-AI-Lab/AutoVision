# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Fit-log semantics: best-metric wording and LR schedule phrasing.

The best-metric line distinguishes the first baseline from a later
improvement (a 0.0000 smoke-run metric is a baseline, not an improvement),
and the LR schedule line names only non-empty phases with the tail floor
correctly attributed to ``min_factor × peak`` instead of "peak LR".
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from torch.nn import Linear
from torch.utils.data import DataLoader, Dataset

from autovision.engine.engine import TaskEngine
from autovision.tasks.shared.flat_cosine import FlatCosineLRScheduler
from autovision.tasks.train_model import TaskStepResult, TrainModel


class _Dataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index):
        return {"x": torch.arange(4, dtype=torch.float32) + index}


class _RisingMetricModel(TrainModel):
    """Task model whose val metric rises on every validation."""

    task = "unit_test"

    def __init__(
        self, warmup: int = 0, flat: int = 0, no_aug: int = 0, total: int = 2
    ) -> None:
        super().__init__(task_model=Linear(4, 2))
        self.validations = 0
        self._windows = (warmup, flat, no_aug, total)

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        loss = self.model(batch["x"]).sum()
        return TaskStepResult(loss=loss, log_dict={"loss": loss.detach()})

    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        return TaskStepResult(loss=torch.ones(()), log_dict={})

    def compute_metrics(self, split: str) -> dict[str, float]:
        self.validations += 1
        return {"val_metric/map": 0.1 * self.validations}

    def get_optimizer(
        self, total_steps: int, global_batch_size: int,
        train_num_batches: int = 0, gradient_accumulation_steps: int = 1,
    ):
        warmup, flat, no_aug, total = self._windows
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        scheduler = FlatCosineLRScheduler(
            optimizer,
            total_steps=total,
            warmup_steps=warmup,
            flat_steps=flat,
            no_aug_steps=no_aug,
        )
        return optimizer, scheduler


def _fit(
    model: _RisingMetricModel, tmp_path: Path
):
    return TaskEngine().fit_task(
        model,
        DataLoader(_Dataset(), batch_size=1),
        DataLoader(_Dataset(), batch_size=2),
        out_dir=tmp_path,
        steps=2,
        save_last=False,
        save_best=False,
        val_every_num_steps=1,
        devices="cpu",
        visualize_examples=False,
    )


class TestBestMetricWording:
    """The first evaluation is a baseline; only later beats 'improve'."""

    def test_first_validation_is_a_baseline_not_an_improvement(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A fresh run logs 'baseline set to' once, then 'improved to'."""
        with caplog.at_level(logging.INFO, logger="autovision"):
            _fit(_RisingMetricModel(), tmp_path)
        messages = [record.getMessage() for record in caplog.records]
        baselines = [m for m in messages if "baseline set to" in m]
        improved = [m for m in messages if "improved to" in m]
        assert len(baselines) == 1
        assert "val_metric/map baseline set to 0.1000 (step 1)" in baselines[0]
        assert len(improved) == 1
        assert "val_metric/map improved to 0.2000 (step 2)" in improved[0]


class TestLRScheduleWording:
    """Only non-empty phases are named; the tail holds min_factor × peak."""

    def test_degenerate_two_step_run_names_warmup_only(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A smoke run whose warmup spans all steps logs one phase."""
        with caplog.at_level(logging.INFO, logger="autovision"):
            _fit(_RisingMetricModel(warmup=2, total=2), tmp_path)
        schedule = [
            record.getMessage() for record in caplog.records
            if record.getMessage().startswith("LR schedule:")
        ]
        assert schedule == ["LR schedule: warmup 2 (steps 2)"]

    def test_full_recipe_names_each_phase_with_the_tail_floor(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A warmup/flat/cosine/tail run lists every phase once."""
        with caplog.at_level(logging.INFO, logger="autovision"):
            _fit(
                _RisingMetricModel(warmup=2, flat=2, no_aug=2, total=10),
                tmp_path,
            )
        schedule = [
            record.getMessage() for record in caplog.records
            if record.getMessage().startswith("LR schedule:")
        ]
        assert schedule == [
            "LR schedule: warmup 2 → flat until 4 → cosine until 8 → "
            "tail at ×0.5 until 10 (steps 10)"
        ]

    def test_no_phase_is_described_as_peak_lr(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The tail holds min_factor × peak; 'peak LR' must not reappear."""
        with caplog.at_level(logging.INFO, logger="autovision"):
            _fit(
                _RisingMetricModel(warmup=2, flat=2, no_aug=2, total=10),
                tmp_path,
            )
        assert not any(
            "at peak LR" in record.getMessage() for record in caplog.records
        )
