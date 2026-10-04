# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Checkpoint format: self-describing, atomic writes, metadata embedded.

Layout::

    {
      "state_dict": <method state dict>,
      "autovision": {
        "version", "date", "license_info", "normalize_args",
        "epoch", "global_step", "engine_state": {"optimizer", "scheduler"},
      },
    }

Atomic write: serialize to ``<path>.tmp`` then ``os.replace`` — a crash never
corrupts the previous checkpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from autovision._license import LICENSE_INFO
from autovision._version import __version__
from autovision.errors import CheckpointError

__all__ = ["CHECKPOINT_AUTOVISION_KEY", "Checkpoint", "CheckpointMetadata"]

CHECKPOINT_AUTOVISION_KEY = "autovision"


@dataclass
class CheckpointMetadata:
    """Framework metadata embedded in every checkpoint."""

    version: str = __version__
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    license_info: str = LICENSE_INFO
    normalize_args: dict | None = None
    epoch: int = -1
    global_step: int = -1
    engine_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "date": self.date,
            "license_info": self.license_info,
            "normalize_args": self.normalize_args,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "engine_state": self.engine_state,
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
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "state_dict": self.state_dict,
                CHECKPOINT_AUTOVISION_KEY: self.autovision.to_dict(),
            },
            tmp_path,
        )
        os.replace(tmp_path, path)

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
                f"File '{path}' is not an Autovision checkpoint (missing "
                f"'{CHECKPOINT_AUTOVISION_KEY}' metadata)."
            )
        metadata = CheckpointMetadata.from_dict(raw[CHECKPOINT_AUTOVISION_KEY])
        return cls(state_dict=raw.get("state_dict", {}), autovision=metadata)
