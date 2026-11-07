# Copyright (c) 2026 AutoVision contributors. Apache-2.0.

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autovision.engine.engine import (
    Engine,
    TaskEngine,
    _StridedValSampler,
)
from autovision.errors import AutoVisionError
from autovision.runtime import distributed as dist
from autovision.tasks.train_model import TaskStepResult, TrainModel


class TestRankDetection:
    def test_unset(self):
        assert dist.get_global_rank() is None
        assert dist.get_local_rank() is None
        assert dist.get_node_rank() is None

    def test_unset_means_zero(self):
        assert dist.is_global_rank_zero()
        assert dist.is_local_rank_zero()
        assert dist.is_node_rank_zero()

    def test_global_rank_chain(self):
        env = {"RANK": "3"}
        assert dist.get_global_rank(env) == 3
        assert not dist.is_global_rank_zero(env)
        env = {"LOCAL_RANK": "1"}  # fallback when RANK absent
        assert dist.get_global_rank(env) == 1
        env = {"SLURM_PROCID": "2"}
        assert dist.get_global_rank(env) == 2
        env = {"JSM_NAMESPACE_RANK": "4"}
        assert dist.get_global_rank(env) == 4

    def test_local_rank_chain(self):
        assert dist.get_local_rank({"LOCAL_RANK": "0"}) == 0
        assert dist.get_local_rank({"SLURM_LOCALID": "5"}) == 5
        assert dist.get_local_rank({"JSM_NAMESPACE_LOCAL_RANK": "2"}) == 2

    def test_node_rank_chain(self):
        assert dist.get_node_rank({"NODE_RANK": "1"}) == 1
        assert dist.get_node_rank({"GROUP_RANK": "2"}) == 2
        assert dist.get_node_rank({"SLURM_NODEID": "0"}) == 0
        assert dist.is_node_rank_zero({"SLURM_NODEID": "0"})
        assert not dist.is_node_rank_zero({"SLURM_NODEID": "1"})

    def test_global_zero_falls_back_to_node_zero(self):
        env = {"SLURM_NODEID": "0"}  # no global rank set
        assert dist.is_global_rank_zero(env)
        env = {"SLURM_NODEID": "1"}
        assert not dist.is_global_rank_zero(env)

    def test_invalid_values_skipped(self):
        env = {"RANK": "not-a-number", "LOCAL_RANK": "7"}
        assert dist.get_global_rank(env) == 7


class TestReduceDict:
    def test_noop_without_distributed(self):
        data = {"loss": __import__("torch").tensor(1.0)}
        assert dist.reduce_dict(data) is data


class TestDataloaderSeedRank:
    def test_zero_without_process_group(self):
        """H1-4: loader seed streams default to rank 0 outside a process group."""
        from autovision.engine.engine import dataloader_seed_rank

        assert not torch.distributed.is_initialized()
        assert dataloader_seed_rank() == 0


# ── shared fixtures for the mocked-2-rank engine checks (H3-3c/e/h) ─────


class _FlatDataset(Dataset):
    """Train-shaped samples (batch key ``x``) with a configurable size."""

    def __init__(self, num_samples: int = 4) -> None:
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict:
        return {"x": torch.arange(4, dtype=torch.float32) + index}


class _ValIdDataset(Dataset):
    """Val samples that carry their own index so counting is observable."""

    def __init__(self, num_samples: int = 7) -> None:
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict:
        return {
            "x": torch.arange(4, dtype=torch.float32) + index,
            "idx": index,
        }


class _ImageLikeDataset(Dataset):
    """(3, 2, 2) tensors so BatchNorm2d tolerates a batch of one."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict:
        return {
            "x": torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) + index
        }


class _SimpleTaskModel(TrainModel):
    """Minimal TrainModel with the optimizer hook fit_task requires."""

    task = "unit_test"

    def __init__(self, task_model: nn.Module | None = None) -> None:
        super().__init__(task_model=task_model or nn.Linear(4, 2))

    def training_step(self, batch: dict, step: int) -> TaskStepResult:
        loss = self.model(batch["x"]).sum()
        return TaskStepResult(
            loss=loss, log_dict={"unit/loss": float(loss.detach())}
        )

    def get_optimizer(
        self, total_steps: int, global_batch_size: int,
        train_num_batches: int = 0, gradient_accumulation_steps: int = 1,
    ):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        return optimizer, torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda _: 1.0
        )


class _CountingValModel(_SimpleTaskModel):
    """Task model whose val metric is the mean of the validated sample ids.

    The metric state is the sample-id list itself, so the rank-zero merged
    metric counts exactly the union of what every rank validated — padding
    duplicates would shift the mean and fail the parity assertion.
    """

    def __init__(self) -> None:
        super().__init__()
        self.seen_val_ids: list[int] = []
        self.merged_val_ids: list[int] | None = None

    def validation_step(self, batch: dict, step: int) -> TaskStepResult:
        self.seen_val_ids.extend(int(value) for value in batch["idx"])
        return TaskStepResult(loss=torch.tensor(0.5), log_dict={})

    def export_metric_state(self, split: str) -> dict:
        return {"ids": list(self.seen_val_ids)}

    def merge_metric_state(self, split: str, state: dict | None) -> None:
        if state is not None:
            base = self.merged_val_ids if self.merged_val_ids is not None else []
            self.merged_val_ids = base + list(state.get("ids", []))

    def reset_metrics(self, split: str) -> None:
        self.seen_val_ids = []

    def compute_metrics(self, split: str) -> dict[str, float]:
        ids = (
            self.merged_val_ids
            if self.merged_val_ids is not None
            else self.seen_val_ids
        )
        if not ids:
            return {"val_metric/mean_id": 0.0}
        return {"val_metric/mean_id": float(sum(ids)) / len(ids)}


def _mock_two_rank_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub a 2-rank process group for single-process engine checks.

    Rank/world queries answer 0/2, reduction collectives are no-ops (the
    flag semantics keep write barriers green) and the DDP wrapper is the
    identity so the real model runs in-process.
    """
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda tensor, *a, **k: None
    )
    monkeypatch.setattr(
        torch.distributed, "destroy_process_group", lambda *a, **k: None
    )
    monkeypatch.setattr(
        torch.nn.parallel, "DistributedDataParallel", lambda model, **kw: model
    )


# ── H3-3c: un-padded cross-rank validation sharding ─────────────────────


class TestStridedValSamplerPartition:
    """H3-3c: the stride sampler partitions the val set exactly."""

    def test_shards_partition_the_val_set_exactly(self):
        """The union of per-rank shards is every index exactly once —
        including shapes where ``DistributedSampler`` padding duplicates
        head samples (N=3/W=4 would pad rank 0 to ``[0, 0]``)."""
        for num_samples, world in [(7, 2), (5, 2), (6, 3), (3, 4), (5, 8)]:
            shards = [
                list(_StridedValSampler(num_samples, rank=rank, world_size=world))
                for rank in range(world)
            ]
            flat = [index for shard in shards for index in shard]
            assert sorted(flat) == list(range(num_samples))
            assert len(flat) == num_samples  # no padding duplicates anywhere

    def test_shard_lengths_track_the_stride(self):
        """N=7/W=2 gives shards of 4 and 3; ranks past the end of the
        stride are legitimately empty (an empty shard contributes nothing
        to the ranked val-loss merge)."""
        assert list(_StridedValSampler(7, rank=0, world_size=2)) == [0, 2, 4, 6]
        assert list(_StridedValSampler(7, rank=1, world_size=2)) == [1, 3, 5]
        assert list(_StridedValSampler(5, rank=7, world_size=8)) == []
        assert len(_StridedValSampler(5, rank=7, world_size=8)) == 0

    def test_set_epoch_never_reorders_the_shard(self):
        """Validation sharding is deterministic; ``set_epoch`` (called per
        val round for engine compatibility) does not reshuffle it."""
        sampler = _StridedValSampler(7, rank=1, world_size=2)
        before = list(sampler)
        sampler.set_epoch(9)
        assert list(sampler) == before


class TestDdpValMetricParity:
    """H3-3c: mocked 2-rank validation counts every val sample exactly once
    and reproduces the single-process metric over the same val set."""

    @staticmethod
    def _run_fit_task(model, tmp_path, train_loader, val_loader, **kwargs):
        return TaskEngine().fit_task(
            model, train_loader, val_loader,
            out_dir=tmp_path, steps=2, val_every_num_steps=2,
            save_every_num_steps=100, save_last=False, save_best=False,
            devices="cpu", visualize_examples=False, **kwargs,
        )

    def test_two_rank_validation_counts_each_sample_once(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ):
        """val set 7 / world_size 2: the val loader carries the strided
        sampler (not the padding ``DistributedSampler``), rank 0 validates
        its own shard without repeats, the merged metric state holds each
        of the 7 samples exactly once, and the DDP metric equals the
        single-process metric (parity anchor)."""
        import autovision.engine.engine as engine_mod

        # Correctness anchor, single process first: the metric over all 7
        # samples is mean(0..6) = 3.0. Runs BEFORE the distributed mock so
        # WORLD_SIZE stays unset.
        single = _CountingValModel()
        self._run_fit_task(
            single, tmp_path,
            DataLoader(_FlatDataset(4), batch_size=2),
            DataLoader(_ValIdDataset(7), batch_size=2),
        )
        assert single.compute_metrics("val") == {"val_metric/mean_id": 3.0}
        assert sorted(single.seen_val_ids) == list(range(7))

        _mock_two_rank_group(monkeypatch)
        exported_states: list[Any] = []

        def fake_all_gather_object(gathered, value, *args, **kwargs):
            # Rank 0's real shard plus the peer's shard of the SAME sampler
            # class the engine installs (the partition's exactness itself is
            # anchored by TestStridedValSamplerPartition).
            exported_states.append(dict(value))
            gathered[0] = value
            gathered[1] = {"ids": list(_StridedValSampler(7, rank=1, world_size=2))}

        monkeypatch.setattr(
            torch.distributed, "all_gather_object", fake_all_gather_object
        )
        monkeypatch.setattr(
            torch.distributed, "broadcast_object_list",
            lambda payload, *a, **k: None,
        )

        recorded_samplers: list[Any] = []
        real_data_loader = engine_mod.DataLoader

        def _recording_data_loader(*args, **kwargs):
            if "sampler" in kwargs:
                recorded_samplers.append(kwargs["sampler"])
            return real_data_loader(*args, **kwargs)

        monkeypatch.setattr(engine_mod, "DataLoader", _recording_data_loader)

        model = _CountingValModel()
        self._run_fit_task(
            model, tmp_path,
            DataLoader(_FlatDataset(4), batch_size=2),
            DataLoader(_ValIdDataset(7), batch_size=2),
        )
        # Two rebuilds happened (train, then val); the val one carries the
        # un-padded strided sampler instead of ``DistributedSampler``.
        assert len(recorded_samplers) == 2
        assert isinstance(recorded_samplers[1], _StridedValSampler)
        # Rank 0's own exported shard: its stride samples, no repeats.
        assert exported_states == [{"ids": [0, 2, 4, 6]}]
        # The merged state holds each of the 7 samples exactly once.
        assert model.merged_val_ids is not None
        assert len(model.merged_val_ids) == 7
        assert sorted(model.merged_val_ids) == list(range(7))
        # Parity anchor: identical metric to the single-process run.
        assert model.compute_metrics("val") == single.compute_metrics("val")


# ── H3-3e: world_size > len(dataset) must fail loudly ───────────────────


class TestEmptyShardGuard:
    """H3-3e: ``world_size > len(dataset)`` leaves every rank an empty
    training shard (zero steps per epoch); both engines must refuse with an
    error naming the minimum data requirement."""

    def test_fit_task_refuses_world_size_above_dataset_size(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ):
        """fit_task with WORLD_SIZE=4 over a 2-sample dataset raises
        ``AutoVisionError`` asking for at least 4 samples."""
        monkeypatch.setenv("WORLD_SIZE", "4")
        with pytest.raises(AutoVisionError, match="at least 4"):
            TaskEngine().fit_task(
                _SimpleTaskModel(), DataLoader(_FlatDataset(2), batch_size=2),
                None, out_dir=tmp_path, steps=2, save_last=False,
                save_best=False, devices="cpu", visualize_examples=False,
            )

    def test_engine_fit_refuses_world_size_above_dataset_size(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ):
        """``Engine.fit`` raises the same guard before any collective."""
        monkeypatch.setenv("WORLD_SIZE", "4")
        with pytest.raises(AutoVisionError, match="at least 4"):
            Engine().fit(
                nn.Linear(4, 2), DataLoader(_FlatDataset(2), batch_size=2),
                out_dir=tmp_path, max_epochs=1, devices="cpu",
            )

    def test_single_process_is_unaffected(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        """The same 2-sample dataset under world_size=1 still trains."""
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        result = TaskEngine().fit_task(
            _SimpleTaskModel(), DataLoader(_FlatDataset(2), batch_size=2),
            None, out_dir=tmp_path, steps=1, save_last=False,
            save_best=False, devices="cpu", visualize_examples=False,
        )
        assert result.steps_run == 1


# ── H3-3h: batch=1 + BatchNorm under DDP advisory ───────────────────────


class _BnNetwork(nn.Module):
    """Conv-shaped network whose BatchNorm2d tolerates a batch of one."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.fc = nn.Linear(12, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.bn(x).flatten(1))


class TestSyncBatchNormAdvisory:
    """H3-3h: DDP + BatchNorm + per-device batch_size=1 warns once that
    per-rank BN statistics come from a single sample; conversion stays the
    user's explicit choice."""

    @staticmethod
    def _run(model, tmp_path, batch_size: int):
        return TaskEngine().fit_task(
            model, DataLoader(_ImageLikeDataset(), batch_size=batch_size),
            None, out_dir=tmp_path, steps=2, save_last=False,
            save_best=False, devices="cpu", visualize_examples=False,
        )

    def test_batch_one_with_batchnorm_warns_once(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        """A BN model at batch_size=1 under the mocked 2-rank group emits
        exactly one advisory naming the SyncBN conversion."""
        _mock_two_rank_group(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="autovision"):
            result = self._run(_SimpleTaskModel(_BnNetwork()), tmp_path, batch_size=1)
        assert result.steps_run == 2  # advisory only, training proceeds
        bn_warnings = [
            record for record in caplog.records
            if "convert_sync_batchnorm" in record.message
        ]
        assert len(bn_warnings) == 1

    def test_batch_above_one_with_batchnorm_stays_silent(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        """batch_size=2 gives BN real per-rank statistics: no advisory."""
        _mock_two_rank_group(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="autovision"):
            self._run(_SimpleTaskModel(_BnNetwork()), tmp_path, batch_size=2)
        assert not [
            record for record in caplog.records
            if "convert_sync_batchnorm" in record.message
        ]

    def test_batch_one_without_batchnorm_stays_silent(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        """A BN-free model at batch_size=1 does not trigger the advisory."""
        _mock_two_rank_group(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="autovision"):
            self._run(
                _SimpleTaskModel(nn.Sequential(nn.Flatten(), nn.Linear(12, 2))),
                tmp_path, batch_size=1,
            )
        assert not [
            record for record in caplog.records
            if "convert_sync_batchnorm" in record.message
        ]
