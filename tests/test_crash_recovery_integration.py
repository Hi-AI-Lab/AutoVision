# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""SIGKILL crash-restart integration (enterprise acceptance L3 anchor).

Complements the friendly two-run resume tests with the hostile path: a
child process is SIGKILLed mid-training, and the test proves the
atomically written ``last.ckpt`` survives, carries the D17 early-stop
state, resumes at the killed step, and early-stops within the restored
patience — plus the D20 stale-tmp sweep on the next save.
"""

from __future__ import annotations

import glob
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from autovision.runtime.checkpoint import Checkpoint

_CHILD_TEMPLATE = '''
import sys
sys.path.insert(0, {repo!r})
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autovision.engine.engine import TaskEngine
from autovision.tasks.train_model import TaskStepResult, TrainModel

OUT = {out_dir!r}
MODE = sys.argv[1]


class _Data(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {{"x": torch.arange(4, dtype=torch.float32) + index}}


class _MetricModel(TrainModel):
    task = "unit_test"
    watch_metric = "val_metric/map"

    def __init__(self, values):
        super().__init__(task_model=nn.Linear(4, 2))
        self.values = values
        self.calls = 0

    def training_step(self, batch, step):
        return TaskStepResult(
            loss=self.model(batch["x"]).sum(),
            log_dict={{"loss_component": torch.tensor(1.0)}},
        )

    def validation_step(self, batch, step):
        return TaskStepResult(loss=torch.ones(()), log_dict={{}})

    def compute_metrics(self, split):
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return {{"val_metric/map": value}}

    def get_optimizer(self, total_steps, global_batch_size,
                      train_num_batches=0, gradient_accumulation_steps=1):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


loader = DataLoader(_Data(), batch_size=2)
if MODE == "train":
    values = [0.5] + [0.2] * 100000  # one improve, then steady decline
    result = TaskEngine().fit_task(
        _MetricModel(values), loader, loader,
        out_dir=OUT, steps=1000000, val_every_num_steps=5,
        save_every_num_steps=10, save_last=True, save_best=False,
        devices="cpu", visualize_examples=False, seed=0,
        early_stopping_patience=1000,
    )
    print(f"CHILD-DONE steps_run={{result.steps_run}} stopped={{result.stopped_early}}")
else:
    from autovision.runtime.checkpoint import Checkpoint

    killed_step = Checkpoint.from_path(
        OUT + "/checkpoints/last.ckpt"
    ).autovision.engine_state["step"]
    values = [0.2] * 100000
    result = TaskEngine().fit_task(
        _MetricModel(values), loader, loader,
        out_dir=OUT, steps=1000000, val_every_num_steps=5,
        save_every_num_steps=10, save_last=True, save_best=False,
        devices="cpu", visualize_examples=False, seed=0,
        early_stopping_patience=3,  # restored counter must stop this fast
        resume_interrupted=True,
    )
    print(
        f"CHILD-DONE resumed_step={{killed_step}} "
        f"steps_run={{result.steps_run}} stopped={{result.stopped_early}} "
        f"stop_step={{result.stop_step}}"
    )
'''

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="SIGKILL-based crash simulation is posix-only"
)


def _wait_for(condition, timeout_s: float, interval_s: float = 0.05) -> bool:
    """Poll ``condition`` until true or the timeout elapses."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(interval_s)
    return False


class TestSigkillCrashRestart:
    """The hostile path: kill -9 mid-run, resume, verify continuity."""

    def test_crash_resume_restores_state_and_stops_early(
        self, tmp_path: Path
    ) -> None:
        """SIGKILL mid-run leaves a loadable checkpoint; resume continues."""
        repo = Path(__file__).parent.parent
        out_dir = tmp_path / "run"
        child_script = tmp_path / "child.py"
        child_script.write_text(
            _CHILD_TEMPLATE.format(repo=str(repo), out_dir=str(out_dir))
        )
        python = sys.executable

        # ── phase 1: train, then SIGKILL once the early-stop state exists ──
        train = subprocess.Popen(
            [python, str(child_script), "train"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        last_ckpt = out_dir / "checkpoints" / "last.ckpt"
        try:

            def _has_decline_state() -> bool:
                try:
                    state = Checkpoint.from_path(last_ckpt).autovision.engine_state
                except Exception:
                    return False  # mid-write read: retry on the next poll
                return int(state.get("no_improve", 0)) >= 1

            if not _wait_for(last_ckpt.exists, timeout_s=60):
                # Kill BEFORE reading stdout: the live child never EOFs and
                # a read here would hang the whole CI job.
                train.send_signal(signal.SIGKILL)
                train.wait(timeout=10)
                pytest.fail(
                    "last.ckpt never appeared; child said: "
                    f"{train.stdout.read()!r}"
                )
            assert _wait_for(_has_decline_state, timeout_s=30), (
                "early-stop decline state never reached the checkpoint"
            )
        finally:
            if train.poll() is None:
                train.send_signal(signal.SIGKILL)
                train.wait(timeout=10)

        # ── durability: the atomically written checkpoint loads standalone ──
        state = Checkpoint.from_path(last_ckpt).autovision.engine_state
        assert int(state["step"]) > 0
        assert "no_improve" in state and "fit_args" in state

        # ── phase 2: resume; the restored patience must stop it quickly ──
        resumed = subprocess.run(
            [python, str(child_script), "resume"],
            capture_output=True, text=True, timeout=120,
        )
        assert resumed.returncode == 0, resumed.stdout[-2000:]
        done_line = [
            line for line in resumed.stdout.splitlines() if "CHILD-DONE" in line
        ]
        assert done_line, "child did not print its result line"
        fields = dict(
            part.split("=", 1)
            for part in done_line[-1].split("CHILD-DONE", 1)[1].split()
        )
        assert int(fields["resumed_step"]) == int(state["step"])
        assert fields["stopped"] == "True"
        assert int(fields["stop_step"]) > int(fields["resumed_step"])
        assert int(fields["steps_run"]) <= 5 * 3  # within restored patience ×3

        # ── D20 hygiene: no stale tmp survives the post-resume saves ──
        assert not glob.glob(str(out_dir / "checkpoints" / "*.tmp"))
