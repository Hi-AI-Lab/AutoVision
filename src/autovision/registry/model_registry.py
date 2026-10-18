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

import functools
import json
import os
import shutil
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock, Timeout

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


_LOCK_TIMEOUT_SEC = 30.0


def _locked(method):
    """Serialize registry operations across processes/threads.

    The filesystem registry is shared state: writes must not race, and a
    reader must not observe a half-copied entry (the manifest is written
    last). ``FileLock`` is reentrant inside one thread, so nested
    ``_write_current``/``_append_audit`` calls do not deadlock.
    """

    @functools.wraps(method)
    def wrapper(self: "ModelRegistry", *args: Any, **kwargs: Any) -> Any:
        try:
            # FileLock serializes processes; the RLock serializes threads
            # sharing this instance (FileLock alone is reentrant per thread
            # and would let two threads through on the same fd). Both are
            # reentrant, so nested _write_current/_append_audit calls work.
            with self._thread_lock, self._lock:
                return method(self, *args, **kwargs)
        except Timeout as error:
            raise RegistryError(
                f"Timed out after {_LOCK_TIMEOUT_SEC:.0f}s waiting for the "
                f"registry lock '{self._lock_path}'."
            ) from error

    return wrapper


def _validate_key_part(label: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(
            f"Invalid registry key: {label}={value!r} "
            "(must be a non-empty string)."
        )
    if value != value.strip():
        raise RegistryError(
            f"Invalid registry key: {label}={value!r} "
            "(leading/trailing whitespace)."
        )
    if value in (".", "..") or "/" in value or "\\" in value or "\x00" in value:
        raise RegistryError(
            f"Invalid registry key: {label}={value!r} (path separators, NUL, "
            "'/' and '..' are not allowed)."
        )


def _validate_key(name: str, version: str) -> None:
    _validate_key_part("name", name)
    _validate_key_part("version", version)


class ModelRegistry:
    """Filesystem-backed model asset registry with a promotion gate."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.root / ".registry.lock"
        self._lock = FileLock(str(self._lock_path), timeout=_LOCK_TIMEOUT_SEC)
        self._thread_lock = threading.RLock()

    # ── registration ────────────────────────────────────────────────────
    @_locked
    def register(
        self,
        model_file: Path | str,
        *,
        name: str,
        version: str,
        stage: RegistryStage = "dev",
        metadata: dict[str, Any] | None = None,
        actor: str | None = None,
    ) -> RegistryEntry:
        """Register a model file as an immutable (name, version, stage).

        Model-card and postprocess sidecars next to the file are copied
        alongside the artifact so deployment gates can read them.
        """
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
        try:
            shutil.copy2(source, target)
            self._copy_sidecars(source, target)
        except Exception:
            shutil.rmtree(entry_dir, ignore_errors=True)
            raise
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
        if self._read_current(name, stage) is None:
            self._write_current(name, stage, version)
        self._append_audit(
            action="register", name=name, version=version, stage=stage,
            actor=actor,
        )
        return entry

    # ── promotion gate ──────────────────────────────────────────────────
    @_locked
    def promote(
        self,
        *,
        name: str,
        version: str,
        to_stage: RegistryStage = "staging",
        gates: Sequence[Any] | None = None,
        actor: str | None = None,
        reason: str | None = None,
    ) -> RegistryEntry:
        """Promote an entry one stage forward (dev -> staging -> prod).

        ``gates`` are evaluated against the promoted copy; any failure
        removes the target and raises. The stage's ``current`` pointer is
        updated only after the gates pass. ``gates=None`` applies the
        default deployment gates (fail-closed); pass ``gates=[]`` only for
        an explicit, audited bypass.
        """
        self._validate_stage(to_stage)
        if to_stage == "dev":
            raise RegistryError("Cannot promote to 'dev'.")
        if gates is None:
            from autovision.registry.deployment_gates import default_gates

            gates = default_gates()
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
        try:
            shutil.copy2(source_entry.path, target)
            self._copy_sidecars(source_entry.path, target)
            if compute_file_sha256(target) != source_entry.sha256:
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
            if gates:
                failures: list[str] = []
                for gate in gates:
                    gate_name = getattr(gate, "name", type(gate).__name__)
                    try:
                        problems = gate.check(entry)
                    except Exception as error:  # noqa: BLE001 - gate boundary
                        problems = [f"{type(error).__name__}: {error}"]
                    failures.extend(f"{gate_name}: {problem}" for problem in problems)
                if failures:
                    raise RegistryError(
                        "Deployment gates failed: " + "; ".join(failures)
                    )
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise
        (target_dir / _MANIFEST_NAME).write_text(
            json.dumps(entry.to_manifest(), indent=2, sort_keys=True)
        )
        self._write_current(name, to_stage, version)
        self._append_audit(
            action="promote", name=name, version=version, stage=to_stage,
            actor=actor, reason=reason,
            details={"from_stage": source_stage},
        )
        return entry

    # ── lookup ──────────────────────────────────────────────────────────
    @_locked
    def get(
        self,
        *,
        name: str,
        stage: RegistryStage = "prod",
        version: str | None = None,
    ) -> RegistryEntry:
        """Load an entry; ``version=None`` resolves the latest in the stage."""
        self._validate_stage(stage)
        _validate_key_part("name", name)
        if version is not None:
            _validate_key_part("version", version)
        stage_dir = self.root / stage / name
        if version is None:
            versions = self._list_versions(stage_dir)
            if not versions:
                raise RegistryError(f"No versions of '{name}' in '{stage}'.")
            version = max(versions, key=_version_sort_key)
        entry_dir = self._entry_dir(name, stage, version)
        if not entry_dir.is_dir():
            raise RegistryError(
                f"Registry entry '{name}' v{version} not found in '{stage}'."
            )
        return self._read_entry(entry_dir)

    @_locked
    def exists(
        self, *, name: str, version: str, stage: RegistryStage
    ) -> bool:
        self._validate_stage(stage)
        return self._entry_dir(name, stage, version).is_dir()

    @_locked
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

    # ── deployment pointer / rollback / audit ───────────────────────────
    @_locked
    def current(
        self, *, name: str, stage: RegistryStage = "prod"
    ) -> RegistryEntry:
        """Resolve the stage's ``current`` pointer (latest as a fallback)."""
        self._validate_stage(stage)
        version = self._read_current(name, stage)
        if version is not None and self.exists(name=name, version=version, stage=stage):
            return self.get(name=name, stage=stage, version=version)
        return self.get(name=name, stage=stage)

    @_locked
    def rollback(
        self,
        *,
        name: str,
        stage: RegistryStage = "prod",
        to_version: str,
        reason: str | None = None,
        actor: str | None = None,
    ) -> RegistryEntry:
        """Point ``current`` at an existing older version (no file copy)."""
        self._validate_stage(stage)
        if not self.exists(name=name, version=to_version, stage=stage):
            raise RegistryError(
                f"Rollback target '{name}' v{to_version} not found in '{stage}'."
            )
        entry = self.get(name=name, stage=stage, version=to_version)
        self._write_current(name, stage, to_version)
        self._append_audit(
            action="rollback", name=name, version=to_version, stage=stage,
            actor=actor, reason=reason,
        )
        return entry

    @_locked
    def audit_log(self) -> list[dict[str, Any]]:
        """Read the append-only audit log (empty when absent)."""
        path = self.root / "audit.jsonl"
        if not path.is_file():
            return []
        entries: list[dict[str, Any]] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    # ── internals ───────────────────────────────────────────────────────
    def _entry_dir(self, name: str, stage: str, version: str) -> Path:
        _validate_key(name, version)
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

    @staticmethod
    def _copy_sidecars(source: Path, target: Path) -> None:
        for suffix in (".modelcard.json", ".postprocess.json"):
            sidecar = Path(str(source) + suffix)
            if sidecar.is_file():
                shutil.copy2(sidecar, Path(str(target) + suffix))

    def _current_path(self, name: str, stage: str) -> Path:
        _validate_key_part("name", name)
        return self.root / stage / name / "current.json"

    def _read_current(self, name: str, stage: str) -> str | None:
        path = self._current_path(name, stage)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        version = payload.get("version")
        return str(version) if version else None

    @_locked
    def _write_current(self, name: str, stage: str, version: str) -> None:
        path = self._current_path(name, stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": version, "updated_at": _utc_now()}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(path)

    @_locked
    def _append_audit(
        self,
        *,
        action: str,
        name: str,
        version: str,
        stage: str,
        actor: str | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "timestamp": _utc_now(),
            "action": action,
            "name": name,
            "version": version,
            "stage": stage,
            "actor": actor or os.environ.get("AUTOVISION_ACTOR", "local"),
            "reason": reason,
            "details": dict(details or {}),
        }
        with (self.root / "audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
