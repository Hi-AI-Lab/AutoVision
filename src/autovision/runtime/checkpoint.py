# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Checkpoint format: self-describing, atomic writes, metadata embedded.

Layout::

    {
      "state_dict": <method state dict>,
      "autovision": {
        "version", "date", "license_info", "normalize_args",
        "epoch", "global_step", "engine_state": {"optimizer", "scheduler"},
      },
    }

Atomic write: serialize to a unique temp file, fsync it, then ``os.replace``
and fsync the directory — a crash never corrupts the previous checkpoint, and
concurrent writers cannot truncate each other's temp file.
"""

from __future__ import annotations

import glob
import logging
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from autovision._license import LICENSE_INFO
from autovision._version import __version__
from autovision.errors import CheckpointError
from autovision.runtime.fs_utils import sweep_stale_temp_files

__all__ = ["CHECKPOINT_AUTOVISION_KEY", "Checkpoint", "CheckpointMetadata"]

logger = logging.getLogger("autovision.runtime.checkpoint")

CHECKPOINT_AUTOVISION_KEY = "autovision"

#: D34: ``torch.save`` (and plain file IO underneath) can fail transiently
#: (antivirus scans, device flushes on network mounts); the atomic write is
#: retried with exponential backoff before the failure is declared fatal.
#: Module-level so tests can collapse the delays.
_SAVE_RETRY_ATTEMPTS = 3
_SAVE_RETRY_BASE_DELAY_S = 0.5


def _write_atomic(path: Path, payload_writer: Callable[[Path], None]) -> None:
    """Atomic write with bounded transient-failure retry (design D34).

    ``payload_writer(tmp_path)`` writes the payload into the given temp
    file. Every attempt gets a fresh temp file, the rename publishes only a
    fully fsynced file, and a failed attempt leaves the previous checkpoint
    intact — so a retry never turns a transient fault into corruption.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Hygiene first: temp files orphaned by a crashed writer (SIGKILL,
    # power loss) are swept before this run starts writing. The age
    # guard keeps concurrent in-flight writers safe; the name is
    # glob-escaped so metacharacters in the filename cannot break it.
    sweep_stale_temp_files(path.parent, f".{glob.escape(path.name)}.*.tmp")
    last_error: OSError | None = None
    for attempt in range(1, _SAVE_RETRY_ATTEMPTS + 1):
        # Unique temp name: two resumed jobs / HPO trials sharing a path
        # must not interleave writes into one ``.tmp`` file.
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        tmp_path = Path(temp_name)
        try:
            payload_writer(tmp_path)
            with tmp_path.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
            _fsync_directory(path.parent)
            return
        except OSError as error:
            last_error = error
            logger.warning(
                "Checkpoint write to '%s' failed (attempt %d/%d): %s",
                path, attempt, _SAVE_RETRY_ATTEMPTS, error,
            )
            if attempt < _SAVE_RETRY_ATTEMPTS:
                time.sleep(_SAVE_RETRY_BASE_DELAY_S * 2 ** (attempt - 1))
        finally:
            tmp_path.unlink(missing_ok=True)
    # Disk full / quota / I/O failure after every retry: fail loud but keep
    # the previous checkpoint intact.
    raise CheckpointError(
        f"Failed to save checkpoint '{path}': {last_error}"
    ) from last_error


@dataclass
class CheckpointMetadata:
    """Framework metadata embedded in every checkpoint.

    ``epoch`` semantic note (TRAIN-09): the pretraining engine stores a
    0-based epoch here, while fit_task reuses the field for its 0-based
    STEP (the loop counter it resumes from). Consumers that need the
    unambiguous value should prefer ``step`` for task checkpoints; the
    field is ``None`` for checkpoints written before it existed.
    """

    version: str = __version__
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    license_info: str = LICENSE_INFO
    normalize_args: dict | None = None
    epoch: int = -1
    global_step: int = -1
    engine_state: dict[str, Any] = field(default_factory=dict)
    step: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "date": self.date,
            "license_info": self.license_info,
            "normalize_args": self.normalize_args,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "engine_state": self.engine_state,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Checkpoint:
    state_dict: dict[str, Any]
    autovision: CheckpointMetadata

    def save(self, path: Path | str) -> None:
        path = Path(path)
        _write_atomic(
            path,
            lambda tmp: torch.save(
                {
                    "state_dict": self.state_dict,
                    CHECKPOINT_AUTOVISION_KEY: self.autovision.to_dict(),
                },
                tmp,
            ),
        )

    def serialize(self) -> bytes:
        """Full checkpoint payload as bytes (single ``torch.save``, D20).

        The archive path (``save_period`` epochs) serializes ONCE and
        writes several destinations from the same payload instead of
        running ``torch.save`` per path. Note this holds the whole
        checkpoint in memory; the regular single-path :meth:`save` keeps
        streaming to disk precisely to avoid that spike.
        """
        import io

        buffer = io.BytesIO()
        torch.save(
            {
                "state_dict": self.state_dict,
                CHECKPOINT_AUTOVISION_KEY: self.autovision.to_dict(),
            },
            buffer,
        )
        return buffer.getvalue()

    def save_serialized(self, path: Path | str, payload: bytes) -> None:
        """Atomically write a payload produced by :meth:`serialize`.

        Same durability discipline as :meth:`save` (stale-tmp sweep, temp
        file, fsync, rename, directory fsync, transient-write retry) so
        archives written from a shared payload are exactly as crash-safe as
        streamed saves.
        """
        path = Path(path)

        def _write(tmp: Path) -> None:
            with tmp.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

        _write_atomic(path, _write)

    @classmethod
    def from_path(cls, path: Path | str, map_location: Any = "cpu") -> "Checkpoint":
        path = Path(path)
        if not path.exists():
            raise CheckpointError(f"Checkpoint does not exist: '{path}'")
        # weights_only=True: the checkpoint format contains only tensors and
        # plain containers — loading untrusted files must not execute code.
        try:
            raw = torch.load(path, map_location=map_location, weights_only=True)
        except Exception as error:
            raise CheckpointError(
                f"Failed to load checkpoint '{path}': {error}"
            ) from error
        if not isinstance(raw, dict) or CHECKPOINT_AUTOVISION_KEY not in raw:
            raise CheckpointError(
                f"File '{path}' is not an AutoVision checkpoint (missing "
                f"'{CHECKPOINT_AUTOVISION_KEY}' metadata)."
            )
        metadata = CheckpointMetadata.from_dict(raw[CHECKPOINT_AUTOVISION_KEY])
        if metadata.version != __version__:
            # Cross-version loading usually works, but is never guaranteed:
            # warn so a silent format drift cannot enter a production run.
            logger.warning(
                "Checkpoint '%s' was written by AutoVision %s but this "
                "process runs %s; cross-version loading is not guaranteed.",
                path, metadata.version, __version__,
            )
        return cls(state_dict=raw.get("state_dict", {}), autovision=metadata)


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory fsync so the rename itself is durable."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
