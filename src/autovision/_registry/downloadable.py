# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Downloadable model registry (baseline counterpart of
``DOWNLOADABLE_MODEL_URL_AND_HASH``): name -> (url, sha256) with verified
download + local cache.

Entries are usually populated by task-model packages at import time; tests
and plugins can add their own via :func:`register_downloadable`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from autovision._models.weight_verification import compute_file_sha256

__all__ = [
    "DOWNLOADABLE_MODEL_URL_AND_HASH",
    "DownloadableModelError",
    "DownloadableModel",
    "register_downloadable",
    "resolve_downloadable",
]

logger = logging.getLogger("autovision")


class DownloadableModelError(RuntimeError):
    """Raised when a downloadable model is unknown or fails verification."""


@dataclass(frozen=True)
class DownloadableModel:
    """A hosted model checkpoint pinned by its sha256."""

    url: str
    sha256: str


DOWNLOADABLE_MODEL_URL_AND_HASH: dict[str, DownloadableModel] = {}


def register_downloadable(
    name: str, url: str, sha256: str
) -> DownloadableModel:
    """Register (or replace) a downloadable model entry."""
    if not name or not url or not sha256:
        raise DownloadableModelError(
            "register_downloadable requires name, url and sha256."
        )
    entry = DownloadableModel(url=url, sha256=sha256)
    DOWNLOADABLE_MODEL_URL_AND_HASH[name] = entry
    return entry


def resolve_downloadable(
    name: str,
    *,
    cache_dir: Path | str | None = None,
) -> Path:
    """Resolve a downloadable model name to a verified local file.

    Downloads the pinned URL into the cache (env ``AUTOVISION_CACHE_DIR`` or
    ~/.cache/autovision by default), verifies sha256, and returns the path.
    An existing cache hit is re-verified (corruption is healed by
    re-download).
    """
    if name not in DOWNLOADABLE_MODEL_URL_AND_HASH:
        known = ", ".join(sorted(DOWNLOADABLE_MODEL_URL_AND_HASH)) or "<none>"
        raise DownloadableModelError(
            f"Unknown downloadable model '{name}'; registered: {known}."
        )
    entry = DOWNLOADABLE_MODEL_URL_AND_HASH[name]
    cache_root = (
        Path(cache_dir)
        if cache_dir is not None
        else _default_cache_dir()
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / f"{name}.pt"
    if target.is_file() and compute_file_sha256(target) == entry.sha256:
        return target
    if target.is_file():
        logger.warning(
            "Cached model '%s' failed sha256 verification; re-downloading.", name
        )
        target.unlink()
    _download(entry.url, target)
    actual = compute_file_sha256(target)
    if actual != entry.sha256:
        target.unlink()
        raise DownloadableModelError(
            f"Downloaded model '{name}' sha256 mismatch: expected "
            f"{entry.sha256}, got {actual}."
        )
    return target


def _default_cache_dir() -> Path:
    import os

    env = os.environ.get("AUTOVISION_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "autovision" / "downloadable_models"


def _download(url: str, target: Path) -> None:
    import urllib.request

    logger.info("Downloading model from %s", url)
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310 (pinned urls)
            target.write_bytes(response.read())
    except Exception as error:  # noqa: BLE001 (clear network failure surface)
        if target.exists():
            target.unlink()
        raise DownloadableModelError(
            f"Failed to download model from '{url}': {error}"
        ) from error
