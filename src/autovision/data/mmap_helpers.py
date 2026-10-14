# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Dataset mmap file orchestration: naming, rank-0 write, refcount cleanup."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from filelock import FileLock

from autovision.data.serialize.memory_mapped_sequence import (
    MemoryMappedSequence,
    write_items_to_file,
)
from autovision.runtime.distributed import is_local_rank_zero
from autovision.runtime.env import Env

__all__ = [
    "dataset_mmap_context",
    "get_dataset_mmap_file",
    "get_dataset_temp_mmap_path",
]

_REF_COUNT_TIMEOUT = 300.0
_POLL_INTERVAL_SEC = 0.2


def _sha256(identifier: str) -> str:
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()


def _ref_count_path(mmap_filepath: Path) -> Path:
    return mmap_filepath.with_suffix(".ref_count")


def _has_active_ref(mmap_filepath: Path) -> bool:
    ref_path = _ref_count_path(mmap_filepath)
    if not ref_path.exists():
        return False
    try:
        return json.loads(ref_path.read_text()) > 0
    except (json.JSONDecodeError, OSError):
        return False


def get_dataset_temp_mmap_path(
    data: str,
    out_dir: Path,
    resume_interrupted: bool = False,
    overwrite: bool = False,
) -> Path:
    """Compute the mmap file path for the dataset.

    The identifier combines the resolved output directory and the data
    fingerprint (paths + mtimes), so a run re-using an output directory with
    different data gets a fresh dataset file. When
    ``AUTOVISION_MMAP_REUSE_FILE`` is set, only the data fingerprint is used
    (shared across runs). A pre-existing file *without an active refcount*
    means a stale file from a crashed run on the same (out, data) pair —
    an error unless resume/overwrite/reuse.
    """
    reuse = Env.MMAP_REUSE_FILE.value
    data_hash = _data_fingerprint(data)
    if reuse:
        identifier = data_hash
    else:
        identifier = f"{Path(out_dir).resolve()}-{data_hash}"
    if not reuse and not resume_interrupted and not overwrite:
        candidate = Env.DATA_CACHE_DIR.value / f"{_sha256(identifier)}.mmap"
        if candidate.exists() and not _has_active_ref(candidate):
            raise RuntimeError(
                f"Detected multiple runs using output directory "
                f"'{Path(out_dir).resolve()}' with the same data. Use a "
                f"different output directory, or set overwrite=True / "
                f"resume_interrupted=True."
            )
    cache_dir = Env.DATA_CACHE_DIR.value
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{_sha256(identifier)}.mmap"


def _data_fingerprint(data: str) -> str:
    """Fingerprint of the data spec: resolved path + mtime when it exists."""
    path = Path(data)
    try:
        stat = path.stat()
        raw = f"{path.resolve()}|{stat.st_mtime_ns}"
    except OSError:
        raw = str(data)
    return _sha256(raw)


def get_dataset_mmap_file(
    items_fn: Callable[[], Iterable[dict[str, Any]]], mmap_filepath: Path
) -> MemoryMappedSequence:
    """Ensure the mmap file exists and return the sequence for this process.

    Local rank zero writes to a temp file and atomically renames; other ranks
    poll until the file appears (or ``AUTOVISION_MMAP_TIMEOUT_SEC`` elapses; the
    timer resets while rank zero is still writing).
    """
    mmap_filepath.parent.mkdir(parents=True, exist_ok=True)
    if mmap_filepath.exists():
        _increment_ref_count(mmap_filepath)
        return MemoryMappedSequence.from_file(mmap_filepath)

    tmp_path = mmap_filepath.with_suffix(
        f".{_sha256(str(mmap_filepath))}-{os.getpid()}.temp"
    )
    if is_local_rank_zero():
        try:
            write_items_to_file(items_fn(), tmp_path)
            tmp_path.replace(mmap_filepath.resolve())
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        timeout_sec = Env.MMAP_TIMEOUT_SEC.value
        start_time = time.time()
        while not mmap_filepath.exists():
            if tmp_path.exists():
                start_time = time.time()  # rank 0 still writing
            if timeout_sec >= 0 and time.time() - start_time > timeout_sec:
                raise RuntimeError(
                    f"Rank waited {timeout_sec:.0f}s for mmap file "
                    f"'{mmap_filepath}' which never appeared."
                )
            time.sleep(_POLL_INTERVAL_SEC)
    _increment_ref_count(mmap_filepath)
    return MemoryMappedSequence.from_file(mmap_filepath)


def _lock_path(mmap_filepath: Path) -> Path:
    return mmap_filepath.with_suffix(".lock")


def _increment_ref_count(mmap_filepath: Path) -> None:
    ref_path = _ref_count_path(mmap_filepath)
    with FileLock(_lock_path(mmap_filepath), timeout=_REF_COUNT_TIMEOUT):
        current = 0
        if ref_path.exists():
            current = json.loads(ref_path.read_text())
        ref_path.write_text(json.dumps(current + 1))


def _decrement_and_cleanup_if_zero(mmap_filepath: Path, reuse: bool) -> None:
    ref_path = _ref_count_path(mmap_filepath)
    with FileLock(_lock_path(mmap_filepath), timeout=_REF_COUNT_TIMEOUT):
        current = 0
        if ref_path.exists():
            current = json.loads(ref_path.read_text())
        remaining = max(0, current - 1)
        if remaining == 0:
            ref_path.unlink(missing_ok=True)
            if not reuse:
                mmap_filepath.unlink(missing_ok=True)
        else:
            ref_path.write_text(json.dumps(remaining))


@contextmanager
def dataset_mmap_context(
    items_fn: Callable[[], Iterable[dict[str, Any]]],
    data: str,
    out_dir: Path,
    resume_interrupted: bool = False,
    overwrite: bool = False,
) -> Iterator[MemoryMappedSequence]:
    """Context manager: mmap file lifecycle (create, share, refcount, clean)."""
    mmap_filepath = get_dataset_temp_mmap_path(
        data=data, out_dir=out_dir, resume_interrupted=resume_interrupted, overwrite=overwrite
    )
    sequence = get_dataset_mmap_file(items_fn, mmap_filepath)
    try:
        yield sequence
    finally:
        _decrement_and_cleanup_if_zero(mmap_filepath, reuse=Env.MMAP_REUSE_FILE.value)
