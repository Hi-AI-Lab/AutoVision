# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Dataset mmap file orchestration: naming, rank-0 write, refcount cleanup."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from filelock import FileLock

from autovision.data.file_helpers import SUPPORTED_IMAGE_EXTENSIONS
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

logger = logging.getLogger("autovision.data.mmap")


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
    fingerprint_extra: str | None = None,
) -> Path:
    """Compute the mmap file path for the dataset.

    The identifier combines the resolved output directory and the data
    fingerprint (label and image file paths + sizes + mtimes, plus
    ``fingerprint_extra`` such as the ignored-class set). Editing an existing
    YOLO ``.txt`` label invalidates the cache even though the images
    directory mtime does not change, and so does adding or deleting an
    image file — the image enumeration defines the mmap rows for
    image-dir datasets (DATA-03). When ``AUTOVISION_MMAP_REUSE_FILE`` is
    set, only the data fingerprint is used (shared across runs). A
    pre-existing file *without an active refcount* means a stale file from
    a crashed run on the same (out, data) pair — an error unless
    resume/overwrite/reuse.
    """
    reuse = Env.MMAP_REUSE_FILE.value
    data_hash = _data_fingerprint(data, fingerprint_extra)
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


_LABEL_SUFFIXES = frozenset({".txt", ".json", ".xml", ".csv", ".yaml", ".yml"})


def _label_files(path: Path) -> list[Path]:
    """Label files feeding annotation content into the fingerprint.

    For image-dir datasets (YOLO/pretrain) the mmap rows are defined by the
    image enumeration — see :func:`_image_files`; label files only carry
    the per-row annotation bytes, so both feeds must invalidate the cache.
    """
    roots = [path]
    if path.name == "images" and (path.parent / "labels").is_dir():
        roots.append(path.parent / "labels")
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() in _LABEL_SUFFIXES:
            files.append(root)
        elif root.is_dir():
            files.extend(
                child
                for child in root.rglob("*")
                if child.is_file() and child.suffix.lower() in _LABEL_SUFFIXES
            )
    return sorted(files)


def _image_files(path: Path) -> list[Path]:
    """Image files that define the mmap rows for image-dir datasets (DATA-03).

    The canonical ``SUPPORTED_IMAGE_EXTENSIONS`` set is used — a superset of
    the YOLO loader's accepted formats. Hashing a superset can only cause a
    spurious rebuild when an unused format appears; hashing a subset would
    risk a stale cache training on a silently wrong row set.
    """
    if path.is_file():
        return []
    return sorted(
        child
        for child in path.rglob("*")
        if child.is_file() and child.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _data_fingerprint(data: str, extra_key: str | None = None) -> str:
    """Fingerprint of the data spec: label + image paths, sizes, mtimes + extra."""
    path = Path(data)
    try:
        if path.is_file():
            stat = path.stat()
            raw = (
                f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|"
                f"{extra_key or ''}"
            )
        elif path.is_dir():
            label_digest = hashlib.sha256()
            labels = _label_files(path)
            for child in labels:
                stat = child.stat()
                root = path if child.is_relative_to(path) else path.parent
                prefix = "" if root == path else f"{root.name}/"
                label_digest.update(
                    f"{prefix}{child.relative_to(root).as_posix()}|"
                    f"{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8")
                )
            image_digest = hashlib.sha256()
            images = _image_files(path)
            for child in images:
                stat = child.stat()
                image_digest.update(
                    f"{child.relative_to(path).as_posix()}|"
                    f"{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8")
                )
            # No label files at all: still distinguish different roots.
            raw = (
                f"{path.resolve()}|{label_digest.hexdigest()}|{len(labels)}|"
                f"{image_digest.hexdigest()}|{len(images)}|{extra_key or ''}"
            )
        else:
            raw = f"{data}|{extra_key or ''}"
    except OSError as error:
        logger.warning(
            "Data fingerprint for '%s' degraded to a content-blind string "
            "(stat failed: %s); a concurrent edit may not invalidate the "
            "mmap row cache (DATA-23).",
            data,
            error,
        )
        raw = f"{data}|{extra_key or ''}"
    return _sha256(raw)


def _open_and_reference(mmap_filepath: Path) -> "MemoryMappedSequence | None":
    """Open an existing mmap file and take one reference, atomically (H3-2g).

    The old ``exists → open → increment`` sequence raced the concurrent
    ``_decrement_and_cleanup_if_zero``: the file could vanish between the
    existence check and the open (a raw ``FileNotFoundError``), or a
    reference could be incremented onto a file that was already deleted —
    a ghost refcount that blocks cleanup forever. The whole sequence now
    runs inside the same FileLock critical section the writer and the
    cleanup path use.

    Known limitation (recorded, not fixed): on Windows ``os.replace``
    cannot move a file that is currently memory-mapped/open, so a
    concurrent publisher and an open reader can still collide there;
    POSIX rename-over-open semantics make the race benign on Linux/macOS.

    Returns:
        The opened sequence, or ``None`` when the file vanished before the
        lock could see it (the caller falls through to build/wait).
    """
    ref_path = _ref_count_path(mmap_filepath)
    with FileLock(_lock_path(mmap_filepath), timeout=_REF_COUNT_TIMEOUT):
        if not mmap_filepath.exists():
            return None
        # Open before incrementing: a corrupt file must not leak a refcount.
        sequence = MemoryMappedSequence.from_file(mmap_filepath)
        current = _read_ref_count(ref_path)
        ref_path.write_text(json.dumps(current + 1))
        return sequence


def get_dataset_mmap_file(
    items_fn: Callable[[], Iterable[dict[str, Any]]],
    mmap_filepath: Path,
    *,
    overwrite: bool = False,
) -> MemoryMappedSequence:
    """Ensure the mmap file exists and return the sequence for this process.

    Local rank zero writes to a temp file and atomically renames; other ranks
    poll until the file appears or ``AUTOVISION_MMAP_TIMEOUT_SEC`` elapses (a
    fixed deadline per waiting rank). ``overwrite=True`` deletes a stale file
    (and its refcount) before rebuilding.

    Opening an EXISTING file (fast path and post-wait path) goes through
    :func:`_open_and_reference`, which holds the publish/cleanup FileLock
    across the whole check-open-increment sequence (H3-2g). The Windows
    ``os.replace``-on-open limitation noted there is a known boundary
    (recorded, not handled).
    """
    mmap_filepath.parent.mkdir(parents=True, exist_ok=True)
    rank_zero = is_local_rank_zero()
    if overwrite and rank_zero:
        # Only rank zero deletes: under DDP with skewed startup a slow rank
        # used to unlink the file rank0 had JUST rebuilt and published, and
        # every rank then waited out AUTOVISION_MMAP_TIMEOUT_SEC on a file
        # that would never come back (DATA-12). Non-zero ranks are readers.
        mmap_filepath.unlink(missing_ok=True)
        _ref_count_path(mmap_filepath).unlink(missing_ok=True)
    if mmap_filepath.exists():
        sequence = _open_and_reference(mmap_filepath)
        if sequence is not None:
            return sequence
        # The file vanished between the existence check and the locked open
        # (concurrent cleanup) — fall through to the build/wait paths. A
        # non-zero rank with overwrite=True racing rank0's rebuild may open
        # the pre-overwrite row set here (split-brain window); the publish
        # itself is atomic and the deletion race above is closed.

    tmp_path = mmap_filepath.with_suffix(
        f".{_sha256(str(mmap_filepath))}-{os.getpid()}.temp"
    )
    if is_local_rank_zero():
        try:
            write_items_to_file(items_fn(), tmp_path)
            # Publish the refcount BEFORE the mmap file becomes visible,
            # under the same lock: a concurrent rank's stale-file check
            # must never observe an mmap file without an active reference.
            with FileLock(_lock_path(mmap_filepath), timeout=_REF_COUNT_TIMEOUT):
                _ref_count_path(mmap_filepath).write_text(json.dumps(1))
                tmp_path.replace(mmap_filepath.resolve())
        finally:
            tmp_path.unlink(missing_ok=True)
        # The initialized refcount above already holds this rank's
        # reference; incrementing again would leak one.
        return MemoryMappedSequence.from_file(mmap_filepath)
    timeout_sec = Env.MMAP_TIMEOUT_SEC.value
    start_time = time.time()
    while True:
        while not mmap_filepath.exists():
            if timeout_sec >= 0 and time.time() - start_time > timeout_sec:
                raise RuntimeError(
                    f"Rank waited {timeout_sec:.0f}s for mmap file "
                    f"'{mmap_filepath}' which never appeared."
                )
            time.sleep(_POLL_INTERVAL_SEC)
        sequence = _open_and_reference(mmap_filepath)
        if sequence is not None:
            return sequence
        # The file vanished between the existence poll and the locked open
        # (concurrent cleanup) — resume polling instead of crashing (H3-2g).
        if timeout_sec >= 0 and time.time() - start_time > timeout_sec:
            raise RuntimeError(
                f"Rank waited {timeout_sec:.0f}s for mmap file "
                f"'{mmap_filepath}' which never appeared."
            )


def _lock_path(mmap_filepath: Path) -> Path:
    return mmap_filepath.with_suffix(".lock")


def _read_ref_count(ref_path: Path) -> int:
    """Read the refcount; a corrupt file is treated as zero after a warning."""
    if not ref_path.exists():
        return 0
    try:
        return max(0, int(json.loads(ref_path.read_text())))
    except (OSError, ValueError, TypeError) as error:
        logger.warning(
            "Corrupt mmap refcount '%s' (%s); resetting to zero.",
            ref_path,
            error,
        )
        return 0


def _increment_ref_count(mmap_filepath: Path) -> None:
    ref_path = _ref_count_path(mmap_filepath)
    with FileLock(_lock_path(mmap_filepath), timeout=_REF_COUNT_TIMEOUT):
        current = _read_ref_count(ref_path)
        ref_path.write_text(json.dumps(current + 1))


def _decrement_and_cleanup_if_zero(mmap_filepath: Path, reuse: bool) -> None:
    ref_path = _ref_count_path(mmap_filepath)
    with FileLock(_lock_path(mmap_filepath), timeout=_REF_COUNT_TIMEOUT):
        current = _read_ref_count(ref_path)
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
    fingerprint_extra: str | None = None,
) -> Iterator[MemoryMappedSequence]:
    """Context manager: mmap file lifecycle (create, share, refcount, clean)."""
    mmap_filepath = get_dataset_temp_mmap_path(
        data=data,
        out_dir=out_dir,
        resume_interrupted=resume_interrupted,
        overwrite=overwrite,
        fingerprint_extra=fingerprint_extra,
    )
    sequence = get_dataset_mmap_file(
        items_fn, mmap_filepath, overwrite=overwrite
    )
    try:
        yield sequence
    finally:
        _decrement_and_cleanup_if_zero(mmap_filepath, reuse=Env.MMAP_REUSE_FILE.value)
