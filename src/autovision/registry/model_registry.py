# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Model asset registry: filesystem backend with dev/staging/prod promotion.

The registry stores immutable model versions in a plain directory tree::

    {root}/dev/{name}/{version}/<model file>
    {root}/dev/{name}/{version}/manifest.json
    {root}/staging/...
    {root}/prod/...

Design (asset & deployment layer, §12.6):
- Backend abstraction: the filesystem backend is the reference; a future
  MLflow backend implements the same :class:`ModelRegistry` contract.
- Immutable versions: registering over an existing version raises.
- Promotion gate: assets move one stage at a time (dev -> staging -> prod);
  skipping a stage or moving backwards is rejected. Promoted artifacts are
  byte-identical (sha256 re-verified after copy).

The download-and-verify asset counterpart is
:mod:`autovision.registry.downloadable`: name -> (url, sha256) entries
with verified download + cache.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from autovision.encoders.weight_verification import compute_file_sha256
from autovision.errors import AutoVisionError

__all__ = [
    "MODEL_STAGES",
    "RegistryEntry",
    "RegistryError",
    "ModelRegistry",
]

MODEL_STAGES = ("dev", "staging", "prod")
RegistryStage = Literal["dev", "staging", "prod"]

_MANIFEST_NAME = "manifest.json"


class RegistryError(AutoVisionError):
    """Raised for registry layout, immutability or promotion-gate violations."""


@dataclass
class RegistryEntry:
    """One immutable model asset in the registry."""

    name: str
    version: str
    stage: str
    path: Path
    sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: str = ""

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path.name)
        return payload

    @classmethod
    def from_manifest(
        cls, payload: dict[str, Any], entry_dir: Path
    ) -> "RegistryEntry":
        data = dict(payload)
        data["path"] = entry_dir / str(data.pop("path"))
        return cls(**data)


def _version_sort_key(version: str) -> tuple[int, ...] | tuple[str, ...]:
    """Semantic-ish sort: numeric versions sort naturally, others fallback."""
    parts = version.split(".")
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return (version,)


class ModelRegistry:
    """Filesystem-backed model asset registry with a promotion gate."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── registration ────────────────────────────────────────────────────
    def register(
        self,
        model_file: Path | str,
        *,
        name: str,
        version: str,
        stage: RegistryStage = "dev",
        metadata: dict[str, Any] | None = None,
    ) -> RegistryEntry:
        """Register a model file as an immutable (name, version, stage)."""
        self._validate_stage(stage)
        source = Path(model_file)
        if not source.is_file():
            raise RegistryError(f"Model file '{source}' does not exist.")
        entry_dir = self._entry_dir(name, stage, version)
        if entry_dir.exists():
            raise RegistryError(
                f"Registry entry '{name}' v{version} already exists in "
                f"'{stage}'; versions are immutable."
            )
        target = entry_dir / source.name
        entry_dir.mkdir(parents=True)
        shutil.copy2(source, target)
        entry = RegistryEntry(
            name=name,
            version=version,
            stage=stage,
            path=target,
            sha256=compute_file_sha256(target),
            metadata=dict(metadata or {}),
            registered_at=_utc_now(),
        )
        (entry_dir / _MANIFEST_NAME).write_text(
            json.dumps(entry.to_manifest(), indent=2, sort_keys=True)
        )
        return entry

    # ── promotion gate ──────────────────────────────────────────────────
    def promote(
        self,
        *,
        name: str,
        version: str,
        to_stage: RegistryStage = "staging",
    ) -> RegistryEntry:
        """Promote an entry one stage forward (dev -> staging -> prod)."""
        self._validate_stage(to_stage)
        if to_stage == "dev":
            raise RegistryError("Cannot promote to 'dev'.")
        # Promotion gate: exactly one stage forward (dev -> staging -> prod).
        source_stage = MODEL_STAGES[MODEL_STAGES.index(to_stage) - 1]
        if not self.exists(name=name, version=version, stage=source_stage):
            raise RegistryError(
                f"Registry entry '{name}' v{version} not found in "
                f"'{source_stage}'; promotion must move one stage at a time."
            )
        source_entry = self.get(name=name, version=version, stage=source_stage)
        target_dir = self._entry_dir(name, to_stage, version)
        if target_dir.exists():
            raise RegistryError(
                f"Registry entry '{name}' v{version} already exists in "
                f"'{to_stage}'; versions are immutable."
            )
        target_dir.mkdir(parents=True)
        target = target_dir / Path(source_entry.path).name
        shutil.copy2(source_entry.path, target)
        if compute_file_sha256(target) != source_entry.sha256:
            shutil.rmtree(target_dir)
            raise RegistryError(
                f"Promoted artifact '{target}' failed sha256 verification; "
                f"promotion aborted."
            )
        entry = RegistryEntry(
            name=name,
            version=version,
            stage=to_stage,
            path=target,
            sha256=source_entry.sha256,
            metadata=dict(source_entry.metadata),
            registered_at=_utc_now(),
        )
        (target_dir / _MANIFEST_NAME).write_text(
            json.dumps(entry.to_manifest(), indent=2, sort_keys=True)
        )
        return entry

    # ── lookup ──────────────────────────────────────────────────────────
    def get(
        self,
        *,
        name: str,
        stage: RegistryStage = "prod",
        version: str | None = None,
    ) -> RegistryEntry:
        """Load an entry; ``version=None`` resolves the latest in the stage."""
        self._validate_stage(stage)
        stage_dir = self.root / stage / name
        if version is None:
            versions = self._list_versions(stage_dir)
            if not versions:
                raise RegistryError(f"No versions of '{name}' in '{stage}'.")
            version = max(versions, key=_version_sort_key)
        entry_dir = stage_dir / version
        if not entry_dir.is_dir():
            raise RegistryError(
                f"Registry entry '{name}' v{version} not found in '{stage}'."
            )
        return self._read_entry(entry_dir)

    def exists(
        self, *, name: str, version: str, stage: RegistryStage
    ) -> bool:
        return (self.root / stage / name / version).is_dir()

    def list_entries(self, stage: RegistryStage | None = None) -> list[RegistryEntry]:
        """List entries, optionally filtered by stage."""
        stages = [stage] if stage is not None else list(MODEL_STAGES)
        entries: list[RegistryEntry] = []
        for current_stage in stages:
            self._validate_stage(current_stage)
            stage_root = self.root / current_stage
            if not stage_root.is_dir():
                continue
            for name_dir in sorted(stage_root.iterdir()):
                for version_dir in sorted(name_dir.iterdir()):
                    if (version_dir / _MANIFEST_NAME).is_file():
                        entries.append(self._read_entry(version_dir))
        return entries

    # ── internals ───────────────────────────────────────────────────────
    def _entry_dir(self, name: str, stage: str, version: str) -> Path:
        if not name or not version or "/" in name or "/" in version:
            raise RegistryError(
                f"Invalid registry key: name='{name}' version='{version}'."
            )
        return self.root / stage / name / version

    def _read_entry(self, entry_dir: Path) -> RegistryEntry:
        manifest = entry_dir / _MANIFEST_NAME
        if not manifest.is_file():
            raise RegistryError(f"Corrupt registry entry: '{entry_dir}'.")
        entry = RegistryEntry.from_manifest(
            json.loads(manifest.read_text()), entry_dir
        )
        if not entry.path.is_file():
            raise RegistryError(
                f"Registered model file missing: '{entry.path}'."
            )
        return entry

    @staticmethod
    def _validate_stage(stage: str) -> None:
        if stage not in MODEL_STAGES:
            raise RegistryError(
                f"Unknown stage '{stage}'; expected one of {MODEL_STAGES}."
            )

    @staticmethod
    def _list_versions(stage_dir: Path) -> list[str]:
        if not stage_dir.is_dir():
            return []
        return [
            path.name
            for path in stage_dir.iterdir()
            if path.is_dir() and (path / _MANIFEST_NAME).is_file()
        ]


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
