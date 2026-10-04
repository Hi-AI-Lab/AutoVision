# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Framework configuration surface via environment variables.

Every environment variable is declared here as a frozen :class:`EnvVar`
descriptor so that defaults, types, and parsing behavior live in one place.
All variables use the ``VTRAIN_`` prefix.

Parsing is deliberately lenient for booleans: any of ``1/true/yes/on``
(case-insensitive) enables a flag. This avoids the baseline pitfall where a
quoted ``"1"`` value silently failed the ``x == "1"`` check.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

__all__ = ["Env", "EnvVar"]

T = TypeVar("T")


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _parse_optional_str(value: str) -> str | None:
    return value or None


@dataclass(frozen=True)
class EnvVar(Generic[T]):
    """A typed environment variable descriptor."""

    name: str
    _default: Any = field(kw_only=True)
    _type: Callable[[Any], T] = field(kw_only=True)
    convert_empty_str_to_default: bool = True

    def default(self) -> T:
        return self._default() if callable(self._default) else self._default

    @property
    def value(self) -> T:
        raw = os.getenv(self.name)
        if raw is None or (self.convert_empty_str_to_default and raw == ""):
            return self.default()
        try:
            return self._type(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Environment variable {self.name}={raw!r} is invalid: {error}"
            ) from error

    def set(self, value: T | None) -> None:
        """Set (or unset with ``None``) the variable in the process env."""
        if value is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = str(value)


def _cache_dir() -> Path:
    return Path.home() / ".cache" / "autovision"


class Env:
    """All framework environment variables."""

    LOG_LEVEL = EnvVar[str]("VTRAIN_LOG_LEVEL", _default="INFO", _type=str)
    CACHE_DIR = EnvVar[Path]("VTRAIN_CACHE_DIR", _default=_cache_dir, _type=Path)
    MODEL_CACHE_DIR = EnvVar[Path](
        "VTRAIN_MODEL_CACHE_DIR", _default=lambda: Env.CACHE_DIR.value / "models", _type=Path
    )
    TMP_DIR = EnvVar[Path](
        "VTRAIN_TMP_DIR", _default=lambda: Env.CACHE_DIR.value / "data", _type=Path
    )
    DATA_CACHE_DIR = EnvVar[Path](
        "VTRAIN_DATA_CACHE_DIR", _default=lambda: Env.TMP_DIR.value, _type=Path
    )
    DATALOADER_TIMEOUT_SEC = EnvVar[float](
        "VTRAIN_DATALOADER_TIMEOUT_SEC", _default=180.0, _type=float
    )
    MAX_NUM_WORKERS_AUTO = EnvVar[int](
        "VTRAIN_MAX_NUM_WORKERS_AUTO", _default=8, _type=int
    )
    DEFAULT_NUM_WORKERS_AUTO = EnvVar[int](
        "VTRAIN_DEFAULT_NUM_WORKERS_AUTO", _default=8, _type=int
    )
    MMAP_TIMEOUT_SEC = EnvVar[float](
        "VTRAIN_MMAP_TIMEOUT_SEC", _default=300.0, _type=float
    )
    MMAP_REUSE_FILE = EnvVar[bool](
        "VTRAIN_MMAP_REUSE_FILE", _default=False, _type=_parse_bool
    )
    VERIFY_OUT_DIR_TIMEOUT_SEC = EnvVar[float](
        "VTRAIN_VERIFY_OUT_DIR_TIMEOUT_SEC", _default=30.0, _type=float
    )
    DOWNLOAD_CHUNK_TIMEOUT_SEC = EnvVar[float](
        "VTRAIN_DOWNLOAD_CHUNK_TIMEOUT_SEC", _default=180.0, _type=float
    )
    OFFLINE = EnvVar[bool]("VTRAIN_OFFLINE", _default=False, _type=_parse_bool)
    MIRROR_URL = EnvVar[str | None](
        "VTRAIN_MIRROR_URL", _default=None, _type=_parse_optional_str
    )
    # Compliance: telemetry is opt-in (default off), the inverse of the baseline.
    TELEMETRY = EnvVar[bool]("VTRAIN_TELEMETRY", _default=False, _type=_parse_bool)
    STEP_TIMEOUT_SEC = EnvVar[float](
        "VTRAIN_STEP_TIMEOUT_SEC", _default=0.0, _type=float  # 0 disables the watchdog
    )

    @staticmethod
    def values() -> dict[str, Any]:
        """Snapshot of all variable values (for run metadata / diagnostics)."""
        return {
            attr.name: attr.value
            for attr in Env.__dict__.values()
            if isinstance(attr, EnvVar)
        }

# Late additions (peripheral subsystems)
Env.TELEMETRY_ENDPOINT = EnvVar[str](
    "VTRAIN_TELEMETRY_ENDPOINT", _default="https://telemetry.example.invalid/v0/e",
    _type=str,
)
