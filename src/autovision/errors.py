# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Custom exception hierarchy.

Every error raised intentionally by the framework derives from
:class:`AutoVisionError` so callers can distinguish framework errors from
accidental bugs (which raise plain exceptions).
"""

__all__ = [
    "AutoVisionError",
    "ConfigError",
    "ConfigMissingKeysError",
    "ConfigUnknownKeyError",
    "ConfigValidationError",
    "CheckpointError",
    "ResultValidationError",
    "UnresolvedAutoError",
    "UnknownModelError",
]


class AutoVisionError(Exception):
    """Base class for all framework errors."""


class UnknownModelError(AutoVisionError):
    """A model cannot be attributed to any known package."""


class ConfigError(AutoVisionError):
    """Base class for user-facing configuration errors."""


class ConfigUnknownKeyError(ConfigError):
    """A configuration key is not recognized."""


class ConfigMissingKeysError(ConfigError):
    """Required configuration keys are missing."""


class ConfigValidationError(ConfigError):
    """A configuration value failed validation."""


class UnresolvedAutoError(AutoVisionError):
    """An ``"auto"`` value was read before being resolved.

    This is a framework bug, not a user error: the resolution chain in the
    command layer must resolve every ``"auto"`` before model instantiation.
    """


class CheckpointError(AutoVisionError):
    """A checkpoint is missing, unreadable, or incompatible."""


class ResultValidationError(AutoVisionError, ValueError):
    """A result payload failed its shape/cardinality contract (H3-1h).

    Dual inheritance is deliberate: :class:`AutoVisionError` keeps the
    framework error family catchable as one type, while :class:`ValueError`
    preserves compatibility with callers written before the hierarchy
    existed — these checks raised bare ``ValueError`` previously. The MRO
    (ResultValidationError -> AutoVisionError -> ValueError) has both bases
    as siblings under ``Exception``, so no linearization conflict arises.
    """
