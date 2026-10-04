# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Custom exception hierarchy.

Every error raised intentionally by the framework derives from
:class:`AutovisionError` so callers can distinguish framework errors from
accidental bugs (which raise plain exceptions).
"""

__all__ = [
    "AutovisionError",
    "ConfigError",
    "ConfigMissingKeysError",
    "ConfigUnknownKeyError",
    "ConfigValidationError",
    "CheckpointError",
    "UnresolvedAutoError",
    "UnknownModelError",
]


class AutovisionError(Exception):
    """Base class for all framework errors."""


class UnknownModelError(AutovisionError):
    """A model cannot be attributed to any known package."""


class ConfigError(AutovisionError):
    """Base class for user-facing configuration errors."""


class ConfigUnknownKeyError(ConfigError):
    """A configuration key is not recognized."""


class ConfigMissingKeysError(ConfigError):
    """Required configuration keys are missing."""


class ConfigValidationError(ConfigError):
    """A configuration value failed validation."""


class UnresolvedAutoError(AutovisionError):
    """An ``"auto"`` value was read before being resolved.

    This is a framework bug, not a user error: the resolution chain in the
    command layer must resolve every ``"auto"`` before model instantiation.
    """


class CheckpointError(AutovisionError):
    """A checkpoint is missing, unreadable, or incompatible."""
