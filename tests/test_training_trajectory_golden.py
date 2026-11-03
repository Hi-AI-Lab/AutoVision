# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Training-loop golden trajectory (enterprise acceptance L2 anchor).

A fixed-seed micro-run pins the EXACT per-step loss sequence and data
order on CPU. Same-seed determinism was verified empirically (2026-11-03
probe: two runs bit-identical), so any engine change that silently shifts
numerics — AMP gating, resume state, sanitization, OOM recovery, loader
seeding — fails here instead of surfacing as an unexplained metric drift
downstream. The fixture is platform/torch-version sensitive by design:
when it diverges after an *intentional* numeric change, delete
``tests/golden/training_trajectory.json`` to re-record and state the
change in the CHANGELOG.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autovision.engine.engine import TaskEngine
from autovision.tasks.train_model import TaskStepResult, TrainModel

GOLDEN_PATH = Path(__file__).parent / "golden" / "training_trajectory.json"

_SEED = 7
_STEPS = 20


class _SeqDataset(Dataset):
    """Samples carry their index as content so data ORDER is observable."""

    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int) -> dict:
        x = torch.zeros(4, dtype=torch.float32)
        x[0] = float(index)
        return {"x": x}


class _RecordingModel(TrainModel):
    """Tiny task model recording the loss trajectory and sample order."""

    task = "unit_test"

    def __init__(self) -> None:
        super().__init__(task_model=nn.Linear(4, 2))
        self.losses: list[float] = []
        self.seen_indices: list[int] = []

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        # Full per-batch order: a strictly stronger anchor than first-sample.
        self.seen_indices.extend(int(row[0]) for row in batch["x"])
        loss = self.model(batch["x"]).sum()
        self.losses.append(float(loss.detach()))
        return TaskStepResult(
            loss=loss, log_dict={"loss_component": loss.detach()}
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


def _run_trajectory(run_dir: Path) -> dict:
    """One deterministic pass; same-seed determinism verified empirically."""
    torch.manual_seed(_SEED)  # model init precedes the engine's reseed
    model = _RecordingModel()
    loader = DataLoader(_SeqDataset(), batch_size=4, shuffle=True, num_workers=0)
    TaskEngine().fit_task(
        model, loader, None,
        out_dir=run_dir, steps=_STEPS,
        save_every_num_steps=10_000, save_last=False, save_best=False,
        devices="cpu", visualize_examples=False, seed=_SEED,
    )
    return {
        "meta": {
            "seed": _SEED,
            "steps": _STEPS,
            "batch_size": 4,
            "dataset_size": 16,
            "note": "CPU-only trajectory; floats are json-repr exact",
        },
        "losses": model.losses,
        "data_order": model.seen_indices,
    }


class TestGoldenTrajectory:
    """The numeric anchor: exact losses and shuffle order per step."""

    def test_trajectory_matches_snapshot(self, tmp_path: Path) -> None:
        """Any silent numeric drift in the training loop fails here."""
        snapshot = _run_trajectory(tmp_path)
        if not GOLDEN_PATH.exists():
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write: a torn fixture file would poison every later
            # comparison with a decode error that looks like numeric drift.
            tmp_file = GOLDEN_PATH.with_suffix(".json.tmp")
            tmp_file.write_text(json.dumps(snapshot, indent=2))
            tmp_file.replace(GOLDEN_PATH)
            pytest.skip("golden trajectory recorded; re-run to compare")
        golden = json.loads(GOLDEN_PATH.read_text())
        # Completeness: 5 epochs over 16 samples, each seen exactly 5 times.
        counts = {i: snapshot["data_order"].count(i) for i in range(16)}
        assert set(counts.values()) == {5}
        assert len(snapshot["data_order"]) == 80
        assert golden == snapshot, (
            "Training numerics diverged from the golden trajectory; if this "
            f"change is intentional, delete {GOLDEN_PATH} to re-record and "
            "state the numeric change in the CHANGELOG."
        )
