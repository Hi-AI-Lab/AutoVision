# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Deprecated alias of :mod:`autovision.data` (compat shim).

Removed in the next minor release; import from the new location.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "autovision._data is a deprecated alias; import from "
    "autovision.data instead.",
    DeprecationWarning,
    stacklevel=2,
)

from autovision.data import *  # noqa: E402,F401,F403
from autovision.data import __all__ as _exported  # noqa: E402,F401
