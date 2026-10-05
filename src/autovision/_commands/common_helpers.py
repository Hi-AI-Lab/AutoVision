# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Shared command helpers: out dir, batch size, num workers."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import torch

from autovision._distributed import is_global_rank_zero
from autovision._env import Env
from autovision.types import PathLike

__all__ = [
    "get_global_batch_size",
    "get_num_workers",
    "get_out_dir",
]

logger = logging.getLogger("autovision")


def get_out_dir(
    out: PathLike, resume_interrupted: bool = False, overwrite: bool = False
) -> Path:
    """Resolve and create the output directory.

    A non-empty existing directory requires ``resume_interrupted`` or
    ``overwrite`` — protects previous runs from silent clobbering.
    ``resume_interrupted=True`` keeps the previous contents (resume may pick
    up ``last.ckpt``); ``overwrite=True`` replaces them, per the contract in
    the error message — stale artifacts (e.g. an old ``metrics.jsonl``)
    would otherwise silently mix into the new run.
    """
    out_dir = Path(out)
    if os.environ.get("WORLD_SIZE", "1") != "1" and not is_global_rank_zero():
        # Multi-process run (torchrun): rank 0 owns the directory lifecycle
        # (it performs the conflict check above/below and may rmtree on
        # overwrite). Later ranks may legitimately arrive after rank 0 has
        # already populated the shared output directory, so they must not
        # fail the non-empty check nor delete it.
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    if (
        out_dir.exists()
        and any(out_dir.iterdir())
        and not (resume_interrupted or overwrite)
    ):
        raise ValueError(
            f"Output directory '{out_dir}' exists and is not empty. Use "
            f"overwrite=True to replace it, or resume_interrupted=True to "
            f"continue a previous run."
        )
    if overwrite and not resume_interrupted and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_global_batch_size(
    batch_size: int, dataset_size: int, num_devices: int
) -> int:
    """Resolve the global batch size (per-device = global / devices)."""
    if batch_size > dataset_size:
        logger.warning(
            "Global batch size (%d) is larger than the dataset size (%d); "
            "reducing it to the dataset size.",
            batch_size, dataset_size,
        )
        batch_size = dataset_size
    if batch_size % num_devices != 0:
        raise ValueError(
            f"Global batch size ({batch_size}) must be divisible by the "
            f"number of devices ({num_devices})."
        )
    return batch_size


def _is_slurm() -> bool:
    return "SLURM_JOB_ID" in os.environ


def get_num_workers(num_workers: int | str, num_devices_per_node: int) -> int:
    """Resolve dataloader workers per dataloader (one per device)."""
    if num_workers != "auto":
        return int(num_workers)
    if _is_slurm() and (cpus := os.getenv("SLURM_CPUS_PER_TASK")):
        num_workers = int(cpus) - 1
    else:
        cpus = os.cpu_count() or Env.DEFAULT_NUM_WORKERS_AUTO.value
        num_workers = max(1, cpus // num_devices_per_node) - 1
    num_workers = min(num_workers, Env.MAX_NUM_WORKERS_AUTO.value)
    return max(0, num_workers)


def resolve_data_paths(data: PathLike | Sequence[PathLike]) -> list[Path]:
    """Normalize ``data`` (dir or file list) to a list of paths."""
    if isinstance(data, (str, Path)):
        return [Path(data)]
    return [Path(item) for item in data]


def build_task_dataloader(dataset, batch_size: int, num_workers: int, train: bool):
    """Train/val task dataloaders (global batch already split per device)."""
    from torch.utils.data import DataLoader

    from autovision._env import Env

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        timeout=Env.DATALOADER_TIMEOUT_SEC.value if num_workers > 0 else 0.0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
