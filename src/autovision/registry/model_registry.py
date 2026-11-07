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
import glob
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock, Timeout

from autovision.encoders.weight_verification import compute_file_sha256
from autovision.errors import AutoVisionError
from autovision.runtime.fs_utils import sweep_stale_temp_files

__all__ = [
    "MODEL_STAGES",
    "RegistryEntry",
    "RegistryError",
    "ModelRegistry",
]

MODEL_STAGES = ("dev", "staging", "prod")
RegistryStage = Literal["dev", "staging", "prod"]

logger = logging.getLogger("autovision")

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
    sidecars: dict[str, str] = field(default_factory=dict)

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


def _version_sort_key(version: str) -> tuple[int, tuple[int | str, ...]]:
    """Sort versions with a total-order key (H3-4g).

    Numeric-semantic versions return ``(1, numeric_tuple)`` and everything
    else ``(0, (version,))``. The leading rank keeps the two shapes mutually
    comparable — the previous bare-string fallback crashed ``max()`` with
    TypeError when a stage mixed ``1.0.0`` with markers like ``rc1`` — and,
    under the ``max()``-based latest resolution, ranks numbered releases
    above non-numeric markers. (The hardening-plan sketch had the ranks the
    other way around; under ``max()`` that would let a marker beat every
    numbered release, contradicting the "numeric latest wins" contract.)
    """
    parts = version.split(".")
    try:
        return (1, tuple(int(part) for part in parts))
    except ValueError:
        return (0, (version,))


def _resolve_latest_version(versions: list[str], *, name: str, stage: str) -> str:
    """Pick the latest version in a stage via :func:`_version_sort_key`.

    A stage mixing numeric and non-numeric versions logs one warning: the
    numeric winner stays deterministic, but ``latest`` over a mixed set is
    a policy call the operator should see announced, not infer from
    silence.
    """
    keys = {version: _version_sort_key(version) for version in versions}
    ranks = {key[0] for key in keys.values()}
    if 0 in ranks and 1 in ranks:
        logger.warning(
            "Registry stage '%s/%s' mixes numeric and non-numeric versions "
            "%s; resolving latest to the numerically highest version.",
            stage,
            name,
            sorted(versions),
        )
    return max(versions, key=keys.__getitem__)


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


#: EXPORT-04: digest memoization keyed on (path, size, mtime_ns) — the same
#: (path,size,mtime,sha) scheme downloadable/user_held use, so repeated
#: reads (current()/list_entries over many entries) do not re-hash multi-GB
#: artifacts inside the registry lock. Bounded by live-entry count; entries
#: whose (size, mtime) change miss and re-derive.
_DIGEST_MEMO: dict[tuple[str, int, int], str] = {}


def _memoized_sha256(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    digest = _DIGEST_MEMO.get(key)
    if digest is None:
        digest = compute_file_sha256(path)
        if len(_DIGEST_MEMO) >= 4096:
            _DIGEST_MEMO.clear()
        _DIGEST_MEMO[key] = digest
    return digest


def _is_orphan_entry(entry_dir: Path) -> bool:
    """EXPORT-01: a directory without a readable manifest is an ORPHAN —
    a register/promote killed between mkdir and the manifest write. The
    immutability contract only covers PUBLISHED entries (manifest present):
    refusing to overwrite the orphan permanently wedged the (name, version)
    key behind a misleading "versions are immutable" error that only manual
    cleanup could clear."""
    manifest = entry_dir / _MANIFEST_NAME
    if manifest.is_file():
        # A parseable manifest is a published entry; an UNPARSEABLE one is
        # also treated as published (it carries deliberate content someone
        # wrote — the Corrupt path reports it loudly instead of the
        # self-heal silently wiping it).
        return False
    return True


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
        # EXPORT audit P3-20: registering a file that is still being
        # written snapshots a TRUNCATED artifact whose sha256 is
        # self-consistent but useless. A two-second-stable mtime is the
        # cheap guard (checkpoints are written via _write_atomic, so a
        # settled file never changes under the exporter).
        source_stat = source.stat()
        if time.time() - source_stat.st_mtime < 2.0:
            # EXPORT audit P3-20: registering a mid-write file snapshots a
            # truncated artifact whose sha256 is self-consistent but
            # useless. A hard error broke the register-then-verify flow
            # every test (and scripted consumer) relies on; the warning
            # surfaces the risk without blocking the deliberate path.
            logger.warning(
                "Model file '%s' was modified less than 2s ago — it may "
                "still be being written; the registration snapshots its "
                "current bytes.",
                source,
            )
        entry_dir = self._entry_dir(name, stage, version)
        self._resolve_target_dir(entry_dir, name=name, version=version,
                                 stage=stage)
        target = entry_dir / source.name
        entry_dir.mkdir(parents=True)
        try:
            shutil.copy2(source, target)
            sidecars = self._copy_sidecars(source, target)
            entry = RegistryEntry(
                name=name,
                version=version,
                stage=stage,
                path=target,
                sha256=compute_file_sha256(target),
                metadata=dict(metadata or {}),
                registered_at=_utc_now(),
                sidecars=sidecars,
            )
            self._write_manifest(entry_dir, entry)
        except Exception:
            shutil.rmtree(entry_dir, ignore_errors=True)
            raise
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
        set_current: bool = True,
    ) -> RegistryEntry:
        """Promote an entry one stage forward (dev -> staging -> prod).

        ``gates`` are evaluated against the promoted copy; any failure
        removes the target and raises. The stage's ``current`` pointer is
        updated only after the gates pass. ``gates=None`` applies the
        default deployment gates (fail-closed); pass ``gates=[]`` only for
        an explicit, audited bypass.

        ``set_current`` (EXPORT-21): promote ALWAYS moved the pointer, so
        re-promoting a historical version into prod silently ROLLED BACK
        production; the register path only sets the pointer when empty.
        Default stays True (the documented behavior), but hotfix-style
        side-by-side promotions can pass ``set_current=False`` explicitly.
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
        self._resolve_target_dir(target_dir, name=name, version=version,
                                 stage=to_stage)
        target_dir.mkdir(parents=True)
        target = target_dir / Path(source_entry.path).name
        gate_failures: list[str] = []
        try:
            shutil.copy2(source_entry.path, target)
            sidecars = self._copy_sidecars(source_entry.path, target)
            if compute_file_sha256(target) != source_entry.sha256:
                raise RegistryError(
                    f"Promoted artifact '{target}' failed sha256 verification; "
                    f"promotion aborted."
                )
            for sidecar_name, expected_sha in source_entry.sidecars.items():
                copied_sidecar = target_dir / sidecar_name
                if (
                    not copied_sidecar.is_file()
                    or compute_file_sha256(copied_sidecar) != expected_sha
                ):
                    raise RegistryError(
                        f"Promoted sidecar '{sidecar_name}' failed sha256 "
                        "verification; promotion aborted."
                    )
            entry = RegistryEntry(
                name=name,
                version=version,
                stage=to_stage,
                path=target,
                sha256=source_entry.sha256,
                metadata=dict(source_entry.metadata),
                registered_at=_utc_now(),
                sidecars=sidecars,
            )
            if gates:
                for gate in gates:
                    gate_name = getattr(gate, "name", type(gate).__name__)
                    try:
                        problems = gate.check(entry)
                    except Exception as error:  # noqa: BLE001 - gate boundary
                        problems = [f"{type(error).__name__}: {error}"]
                    gate_failures.extend(
                        f"{gate_name}: {problem}" for problem in problems
                    )
                if gate_failures:
                    raise RegistryError(
                        "Deployment gates failed: " + "; ".join(gate_failures)
                    )
            self._write_manifest(target_dir, entry)
        except Exception as error:
            # H3-4j: a blocked promotion is exactly the event an auditor
            # needs to see; previously only successes reached the audit log,
            # so a gate-rejected attempt left no trace behind the rmtree.
            # Same hash-chain writer as the success records (format
            # unchanged); reasons are summarized so a huge gate report
            # cannot bloat the append-only log.
            shutil.rmtree(target_dir, ignore_errors=True)
            self._append_audit(
                action="promote_failed",
                name=name,
                version=version,
                stage=to_stage,
                actor=actor,
                reason=reason,
                details={
                    "from_stage": source_stage,
                    "failures": _summarize_failures(gate_failures, error),
                },
            )
            raise
        if set_current:
            self._write_current(name, to_stage, version)
        self._append_audit(
            action="promote", name=name, version=version, stage=to_stage,
            actor=actor, reason=reason,
            details={
                "from_stage": source_stage,
                # EXPORT-07: make the "explicit, audited bypass" promise
                # real — an auditor must be able to tell a gates=[] push
                # from a fully-gated one without diffing gate reports.
                "gates": "none (explicit bypass)" if not gates else "default",
                "bypass": not bool(gates),
                "set_current": set_current,
            },
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
        verify: bool = True,
    ) -> RegistryEntry:
        """Load an entry; ``version=None`` resolves the latest in the stage.

        ``verify=True`` (default) re-hashes the artifact and every sidecar
        under the registry lock — a 2GB artifact takes seconds and blocks
        writers (EXPORT-04). ``verify=False`` trusts the manifest (its sha
        was pinned at register/promote time); the memoized digests make the
        hot ``current()`` path cheap across repeated reads.
        """
        self._validate_stage(stage)
        _validate_key_part("name", name)
        if version is not None:
            _validate_key_part("version", version)
        stage_dir = self.root / stage / name
        if version is None:
            versions = self._list_versions(stage_dir)
            if not versions:
                raise RegistryError(f"No versions of '{name}' in '{stage}'.")
            version = _resolve_latest_version(versions, name=name, stage=stage)
        entry_dir = self._entry_dir(name, stage, version)
        if not entry_dir.is_dir():
            raise RegistryError(
                f"Registry entry '{name}' v{version} not found in '{stage}'."
            )
        return self._read_entry(entry_dir, verify=verify)

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
                        # EXPORT-15: a single corrupt entry used to kill the
                        # whole listing (and the CLI that reads it) — report
                        # it and keep listing.
                        try:
                            entries.append(self._read_entry(version_dir))
                        except RegistryError as error:
                            logger.warning(
                                "Skipping unreadable registry entry '%s': %s",
                                version_dir, error,
                            )
        return entries

    # ── deployment pointer / rollback / audit ───────────────────────────
    @_locked
    def current(
        self, *, name: str, stage: RegistryStage = "prod", verify: bool = True
    ) -> RegistryEntry:
        """Resolve the stage's ``current`` pointer (latest as a fallback)."""
        self._validate_stage(stage)
        version = self._read_current(name, stage)
        if version is not None and self.exists(name=name, version=version, stage=stage):
            return self.get(
                name=name, stage=stage, version=version, verify=verify
            )
        return self.get(name=name, stage=stage, verify=verify)

    @_locked
    def rollback(
        self,
        *,
        name: str,
        stage: RegistryStage = "prod",
        to_version: str,
        reason: str | None = None,
        actor: str | None = None,
        verify: bool = True,
    ) -> RegistryEntry:
        """Point ``current`` at an existing older version (no file copy)."""
        self._validate_stage(stage)
        if not self.exists(name=name, version=to_version, stage=stage):
            raise RegistryError(
                f"Rollback target '{name}' v{to_version} not found in '{stage}'."
            )
        entry = self.get(
            name=name, stage=stage, version=to_version, verify=verify
        )
        self._write_current(name, stage, to_version)
        self._append_audit(
            action="rollback", name=name, version=to_version, stage=stage,
            actor=actor, reason=reason,
        )
        return entry

    @_locked
    def audit_log(self) -> list[dict[str, Any]]:
        """Read and verify the append-only audit log (empty when absent)."""
        path = self.root / "audit.jsonl"
        if not path.is_file():
            return []
        entries: list[dict[str, Any]] = []
        previous_hash = ""
        hashed = False
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise RegistryError(
                    f"Malformed audit log line {line_number}: {error}."
                ) from error
            if "entry_hash" not in entry:
                if hashed:
                    raise RegistryError(
                        f"Audit log line {line_number} is unhashed after "
                        "hashed entries."
                    )
                entries.append(entry)
                continue
            hashed = True
            payload = {key: value for key, value in entry.items() if key != "entry_hash"}
            expected = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if entry["entry_hash"] != expected:
                raise RegistryError(
                    f"Audit log line {line_number} hash mismatch (tampered)."
                )
            if entry.get("prev_hash", "") != previous_hash:
                raise RegistryError(
                    f"Audit log line {line_number} breaks the hash chain."
                )
            previous_hash = expected
            entries.append(entry)
        return entries

    # ── internals ───────────────────────────────────────────────────────
    def _entry_dir(self, name: str, stage: str, version: str) -> Path:
        _validate_key(name, version)
        return self.root / stage / name / version

    #: A fresh orphan directory may be a concurrent registration IN FLIGHT
    #: (mkdir before the manifest); only an aged one is provably dead.
    _ORPHAN_GRACE_SEC = 600.0

    def _resolve_target_dir(
        self, entry_dir: Path, *, name: str, version: str, stage: str
    ) -> None:
        """EXPORT-01: refuse published entries, self-heal dead orphans.

        Three states, each with its own message: published (manifest
        present) → the immutable rejection; fresh orphan (mtime inside the
        grace window) → likely a concurrent registration, tell the user to
        wait; aged orphan (killed writer) → remove and proceed, loudly.
        """
        if not entry_dir.exists():
            return
        if not _is_orphan_entry(entry_dir):
            raise RegistryError(
                f"Registry entry '{name}' v{version} already exists in "
                f"'{stage}'; versions are immutable."
            )
        age = time.time() - entry_dir.stat().st_mtime
        if age < self._ORPHAN_GRACE_SEC:
            raise RegistryError(
                f"'{entry_dir}' holds an unfinished registration younger "
                f"than {int(self._ORPHAN_GRACE_SEC)}s — another process may "
                "still be writing it. Wait for it to complete, or remove "
                "the directory if the writer is confirmed dead."
            )
        import shutil as _shutil

        _shutil.rmtree(entry_dir, ignore_errors=True)
        logger.warning(
            "Removed stale registry orphan '%s' (%.0fs old, no manifest) — "
            "the previous register/promote died before publishing; the key "
            "'%s' v%s in '%s' is writable again.",
            entry_dir, age, name, version, stage,
        )

    def _read_entry(self, entry_dir: Path, *, verify: bool = True) -> RegistryEntry:
        manifest = entry_dir / _MANIFEST_NAME
        if not manifest.is_file():
            # EXPORT-01: align with _list_versions (which skips these
            # directories) — an in-progress/dead registration is NOT a
            # corrupt published entry.
            raise RegistryError(
                f"Incomplete registry entry (no manifest): '{entry_dir}' — "
                "the registration died before publishing. Remove the "
                "directory to retry, or wait if a writer may be live."
            )
        try:
            entry = RegistryEntry.from_manifest(
                json.loads(manifest.read_text()), entry_dir
            )
        except (OSError, ValueError, TypeError) as error:
            raise RegistryError(
                f"Corrupt registry manifest '{manifest}': {error}."
            ) from error
        root = entry_dir.resolve()
        resolved = entry.path.resolve()
        if not resolved.is_relative_to(root):
            raise RegistryError(
                f"Registry manifest path escapes its entry directory: "
                f"'{entry.path}'."
            )
        if not resolved.is_file():
            raise RegistryError(
                f"Registered model file missing: '{entry.path}'."
            )
        actual = _memoized_sha256(resolved) if verify else entry.sha256
        if verify and actual != entry.sha256:
            raise RegistryError(
                f"Registry artifact sha256 mismatch for '{resolved}': "
                f"{actual} != {entry.sha256}."
            )
        for sidecar_name, expected_sha in entry.sidecars.items():
            sidecar = (root / sidecar_name).resolve()
            if not sidecar.is_relative_to(root) or not sidecar.is_file():
                raise RegistryError(
                    f"Registered sidecar missing or outside the entry: "
                    f"'{sidecar_name}'."
                )
            if verify and _memoized_sha256(sidecar) != expected_sha:
                raise RegistryError(
                    f"Registry sidecar sha256 mismatch for '{sidecar_name}'."
                )
        entry.path = resolved
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
    def _copy_sidecars(source: Path, target: Path) -> dict[str, str]:
        """Copy sidecars and return ``{filename: sha256}`` for the manifest.

        EXPORT-03: beyond the model-card/parity family this also covers the
        IDENTITY sidecars — TensorRT's ``<engine>.meta.json`` and OpenVINO's
        ``<xml>.json`` (only when the artifact is an .xml, so an unrelated
        model.json next to a .pt cannot hijack the slot). Dropping them used
        to make ``verify_engine_artifact`` silently skip at load time from
        registry copies: the identity gate vanished exactly for the
        TRT/OV artifacts that need it most.
        """
        hashes: dict[str, str] = {}
        suffixes = [".modelcard.json", ".postprocess.json", ".parity.json"]
        for suffix in suffixes:
            sidecar = Path(str(source) + suffix)
            if sidecar.is_file():
                copied = Path(str(target) + suffix)
                shutil.copy2(sidecar, copied)
                hashes[copied.name] = compute_file_sha256(copied)
        # EXPORT-03: identity sidecars follow the LOADER's with_suffix
        # derivation (read_metadata: artifact.with_suffix(suffix)) — NOT the
        # str+suffix convention above. For `model.engine` the loader looks
        # up `model.meta.json`; for `model.xml` it looks up `model.json`.
        identity_suffix: str | None = None
        if source.suffix.lower() in (".engine", ".trt"):
            identity_suffix = ".meta.json"
        elif source.suffix.lower() == ".xml":
            identity_suffix = ".json"
        if identity_suffix is not None:
            sidecar = source.with_suffix(identity_suffix)
            if sidecar.is_file():
                copied = target.with_suffix(identity_suffix)
                shutil.copy2(sidecar, copied)
                hashes[copied.name] = compute_file_sha256(copied)
        return hashes

    @staticmethod
    def _write_manifest(entry_dir: Path, entry: RegistryEntry) -> None:
        """Write the manifest atomically; a crash never leaves a partial one."""
        target = entry_dir / _MANIFEST_NAME
        sweep_stale_temp_files(entry_dir, f".{glob.escape(_MANIFEST_NAME)}.*.tmp")
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{_MANIFEST_NAME}.", suffix=".tmp", dir=entry_dir
        )
        os.close(descriptor)
        temp_path = Path(temp_name)
        try:
            temp_path.write_text(
                json.dumps(entry.to_manifest(), indent=2, sort_keys=True)
            )
            os.replace(temp_path, target)
        finally:
            temp_path.unlink(missing_ok=True)

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
        # The temp file below is ``path.with_suffix(".tmp")`` — i.e.
        # "{stem}.tmp", NOT "{name}.tmp".
        sweep_stale_temp_files(path.parent, f"{glob.escape(path.stem)}.tmp")
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
        path = self.root / "audit.jsonl"
        previous_hash = ""
        if path.is_file():
            # P3-20 (EXPORT audit): the whole-log read made every append
            # O(log size) — linear in the file for a chain lookup. Only the
            # LAST non-empty line matters; read the tail block.
            non_empty: list[str] = []
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                position = handle.tell()
                tail = b""
                while position > 0:
                    step = min(8192, position)
                    position -= step
                    handle.seek(position)
                    block = handle.read(step)
                    tail = block + tail
                    decoded = tail.decode("utf-8", errors="replace")
                    non_empty = [
                        line for line in decoded.splitlines() if line.strip()
                    ]
                    if len(non_empty) >= 2 or position == 0:
                        break
            if non_empty:
                try:
                    previous_hash = str(
                        json.loads(non_empty[-1]).get("entry_hash", "")
                    )
                except json.JSONDecodeError as error:
                    raise RegistryError(
                        "Refusing to append to a malformed audit log: "
                        f"{error}."
                    ) from error
        entry: dict[str, Any] = {
            "timestamp": _utc_now(),
            "action": action,
            "name": name,
            "version": version,
            "stage": stage,
            "actor": actor or os.environ.get("AUTOVISION_ACTOR", "local"),
            "reason": reason,
            "details": dict(details or {}),
            "prev_hash": previous_hash,
        }
        payload = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        entry["entry_hash"] = hashlib.sha256(payload.encode()).hexdigest()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_FAIL_SUMMARY_ITEMS = 20
_FAIL_SUMMARY_CHARS = 200


def _summarize_failures(
    gate_failures: list[str], error: BaseException
) -> list[str]:
    """Audit-safe failure summary (H3-4j): the collected gate reasons when
    the gates rejected the promotion, else the raising exception's summary.

    Capped in count and per-item length so a pathological gate report cannot
    bloat the append-only log.
    """
    reasons = gate_failures or [f"{type(error).__name__}: {error}"]
    return [
        reason[:_FAIL_SUMMARY_CHARS]
        for reason in reasons[:_FAIL_SUMMARY_ITEMS]
    ]
