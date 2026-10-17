# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""License normalization + installed-dependency inventory (compliance)."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any, Iterable

__all__ = [
    "DEFAULT_ALLOWLIST",
    "LicenseEntry",
    "dependency_inventory",
    "normalize_license",
]

DEFAULT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "Python-2.0",
        "PSF-2.0",
        "MPL-2.0",
    }
)

_ALIASES = {
    "apache 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "mit": "MIT",
    "mit license": "MIT",
    "bsd": "BSD-3-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd 3-clause": "BSD-3-Clause",
    "bsd-2-clause": "BSD-2-Clause",
    "isc": "ISC",
    "python software foundation license": "PSF-2.0",
    "psf-2.0": "PSF-2.0",
    "mozilla public license 2.0": "MPL-2.0",
    "mpl-2.0": "MPL-2.0",
}


@dataclass(frozen=True)
class LicenseEntry:
    name: str
    version: str
    license: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version, "license": self.license}


def normalize_license(value: str | None) -> str:
    """Map common metadata strings to an SPDX-ish id; unknown is explicit."""
    if not value:
        return "UNKNOWN"
    text = " ".join(str(value).split())
    lowered = text.lower()
    for alias, spdx in _ALIASES.items():
        if alias in lowered:
            return spdx
    if len(text) <= 64 and " " not in text:
        return text
    return "UNKNOWN"


def _license_from_metadata(metadata: Any) -> str:
    expression = metadata.get("License-Expression")
    if expression:
        return normalize_license(expression)
    declared = metadata.get("License")
    if declared:
        return normalize_license(declared)
    classifiers = metadata.get_all("Classifier") or []
    for classifier in classifiers:
        if classifier.startswith("License ::"):
            return normalize_license(classifier.split("::")[-1])
    return "UNKNOWN"


def dependency_inventory(
    *, skip: Iterable[str] = ()
) -> list[LicenseEntry]:
    """Installed distributions with their declared license (sorted)."""
    skipped = {name.lower().replace("_", "-") for name in skip}
    entries: dict[tuple[str, str], LicenseEntry] = {}
    for distribution in importlib.metadata.distributions():
        metadata = distribution.metadata
        name = metadata.get("Name")
        if not name:
            continue
        normalized_name = name.lower().replace("_", "-")
        if normalized_name in skipped:
            continue
        version = metadata.get("Version") or ""
        entries[(name, version)] = LicenseEntry(
            name=name,
            version=version,
            license=_license_from_metadata(metadata),
        )
    return [entries[key] for key in sorted(entries)]
