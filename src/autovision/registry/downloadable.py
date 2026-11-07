# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Downloadable model registry: name -> (url, sha256[, license]) with
verified download + local cache.

Entries are usually populated by task-model packages at import time; tests
and plugins can add their own via :func:`register_downloadable`. License
metadata comes from :mod:`autovision.registry.assets`.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock, Timeout

from autovision.encoders.weight_verification import compute_file_sha256
from autovision.errors import AutoVisionError
from autovision.registry.assets import LicenseRef
from autovision.runtime.env import Env

__all__ = [
    "ASSET_MANIFEST",
    "DownloadableModelError",
    "DownloadableModel",
    "LicenseRef",
    "register_downloadable",
    "resolve_downloadable",
]

logger = logging.getLogger("autovision")

#: How long a process waits for another process's in-flight download of the
#: same asset before giving up. Downloads dwarf the registry lock budget
#: (model containers are 100MB-class), so this is its own generous ceiling.
# EXPORT-19: bounded per-source retries with exponential backoff.
_DOWNLOAD_ATTEMPTS_PER_SOURCE = 3
_DOWNLOAD_RETRY_BASE_S = 1.0
_LOCK_TIMEOUT_SEC = 1800.0

#: Streaming download chunk size (constant memory for multi-GB containers).
_DOWNLOAD_CHUNK_SIZE = 1 << 22

#: Verified-digest cache (H3-4i): (resolved path, size, mtime_ns, expected
#: sha) -> True. Mirrors the user_held_assets pattern — any file change
#: invalidates the entry, so a corrupted/healed cache always re-hashes.
_VERIFIED_FILES: dict[tuple[str, int, int, str], bool] = {}


class DownloadableModelError(AutoVisionError):
    """Raised when a downloadable model is unknown or fails verification."""


@dataclass(frozen=True)
class DownloadableModel:
    """A hosted model checkpoint pinned by its sha256.

    ``license`` is optional until the Phase 4 asset switch (design doc 28,
    ledger P3-1) and mandatory afterwards: every downloadable binary must
    state its license (see :mod:`autovision.registry.assets`).

    ``mirrors`` lists fallback URLs tried in order when the primary ``url``
    fails (download error or sha256 mismatch) — e.g. a regional hub mirror
    next to an upstream release URL.
    """

    url: str
    sha256: str
    license: LicenseRef | None = None
    mirrors: tuple[str, ...] = ()


ASSET_MANIFEST: dict[str, DownloadableModel] = {}


def register_downloadable(
    name: str,
    url: str,
    sha256: str,
    license: LicenseRef | None = None,
    mirrors: tuple[str, ...] = (),
) -> DownloadableModel:
    """Register (or replace) a downloadable model entry."""
    if not name or not url or not sha256:
        raise DownloadableModelError(
            "register_downloadable requires name, url and sha256."
        )
    # EXPORT-08: the name becomes `cache_dir/<name>.pt` — validate against
    # cache-directory escape. HF-style single-level namespace slashes
    # ("d-fine/dfine_s_coco") are the established convention and stay
    # legal; what must never pass: '..' components (traversal), absolute
    # names, backslashes (Windows separator), NUL, whitespace edges, or
    # empty segments. Plugins registering at import time make this a real
    # write-primitive, not just hygiene.
    if name != name.strip() or not name:
        raise DownloadableModelError(
            f"Invalid downloadable name {name!r} (empty or whitespace edges)."
        )
    if "\\x00" in name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise DownloadableModelError(
            f"Invalid downloadable name {name!r} (NUL, backslash or an "
            "absolute path are not allowed — the name becomes a cache "
            "filename)."
        )
    segments = [segment for segment in name.split("/") if segment]
    if (
        not segments
        or any(segment == ".." for segment in segments)
        or any(not segment.strip() for segment in segments)
    ):
        raise DownloadableModelError(
            f"Invalid downloadable name {name!r} ('..' components and "
            "empty segments are not allowed — the name becomes a cache "
            "filename)."
        )
    entry = DownloadableModel(url=url, sha256=sha256, license=license, mirrors=mirrors)
    ASSET_MANIFEST[name] = entry
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
    re-download); the digest result is memoized per (path, size, mtime), so
    repeated resolves in one process do not re-hash the container.

    Concurrent resolves of the same asset serialize on a per-target
    ``FileLock``: without it, two processes download into the same path and
    a reader can observe the loser's mismatched bytes.
    """
    if name not in ASSET_MANIFEST:
        known = ", ".join(sorted(ASSET_MANIFEST)) or "<none>"
        raise DownloadableModelError(
            f"Unknown downloadable model '{name}'; registered: {known}."
        )
    entry = ASSET_MANIFEST[name]
    if entry.license is None:
        # Legacy entries registered before the R7 license field existed
        # (design doc 28 §6.3); all of them are scheduled for replacement
        # in the Phase 4 asset switch.
        logger.warning(
            "Downloadable asset '%s' carries no license metadata (legacy "
            "entry, pending Phase 4 replacement); resolving anyway.",
            name,
        )
    cache_root = (
        Path(cache_dir)
        if cache_dir is not None
        else _default_cache_dir()
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / f"{name}.pt"
    # Registered names may contain a namespace ("d-fine/dfine_s_coco"),
    # so the cache target can be nested; the lock sits NEXT TO the target
    # so each asset gets its own mutex regardless of nesting.
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(
            str(target.parent / f".{target.name}.lock"), timeout=_LOCK_TIMEOUT_SEC
        ):
            return _resolve_locked(name, entry, target)
    except Timeout as error:
        raise DownloadableModelError(
            f"Timed out after {_LOCK_TIMEOUT_SEC:.0f}s waiting for another "
            f"process downloading '{name}' into '{target}'."
        ) from error


def _resolve_locked(
    name: str, entry: DownloadableModel, target: Path
) -> Path:
    """Cache-check / download / verify for one asset, under the target lock."""
    if target.is_file() and _file_matches(target, entry.sha256):
        return target
    if target.is_file():
        logger.warning(
            "Cached model '%s' failed sha256 verification; re-downloading.", name
        )
        _evict_digest_cache(target)
        # EXPORT-18: keep the bytes on disk when OFFLINE — deleting the only
        # local copy of a model the user cannot re-download is destructive;
        # the corrupt file stays for manual inspection/repair.
        if not Env.OFFLINE.value:
            target.unlink()
    if Env.OFFLINE.value:
        raise DownloadableModelError(
            f"AUTOVISION_OFFLINE=1 and no verified cache for '{name}' at "
            f"'{target}'. Run once without AUTOVISION_OFFLINE to populate "
            "the cache, or pre-seed it with a sha256-matching file."
        )
    last_error: DownloadableModelError | None = None
    for source in (entry.url, *entry.mirrors):
        # EXPORT-19: transient network failures used to fall straight to the
        # next mirror (or abort with no mirror) — a small bounded backoff on
        # each source covers the common blip without masking persistent
        # sha256 mismatches (those fail after a successful download).
        attempts = _DOWNLOAD_ATTEMPTS_PER_SOURCE
        downloaded = False
        for attempt in range(1, attempts + 1):
            try:
                _download(source, target)
                downloaded = True
                break
            except DownloadableModelError as error:
                last_error = error
                if attempt < attempts:
                    delay = _DOWNLOAD_RETRY_BASE_S * 2 ** (attempt - 1)
                    logger.warning(
                        "Download attempt %d/%d from %s failed (%s); "
                        "retrying in %.1fs.",
                        attempt, attempts, source, error, delay,
                    )
                    time.sleep(delay)
        if not downloaded:
            # No bytes on disk from this source — hash-checking would crash
            # on the missing file; move to the next mirror.
            logger.warning(
                "Download source failed after %d attempts, trying next: %s",
                attempts, source,
            )
            continue
        actual = compute_file_sha256(target)
        if actual != entry.sha256:
            _evict_digest_cache(target)
            target.unlink()
            last_error = DownloadableModelError(
                f"Downloaded model '{name}' from '{source}' sha256 mismatch: "
                f"expected {entry.sha256}, got {actual}."
            )
            logger.warning("sha256 mismatch, trying next source: %s", source)
            continue
        # The post-download hash doubles as the first memo entry: without
        # this, every later resolve would pay one full-file hash before the
        # cache catches up (H3-4i).
        _mark_verified(target, entry.sha256)
        return target
    raise last_error or DownloadableModelError(
        f"No download source succeeded for '{name}'."
    )


def _digest_cache_key(target: Path, expected_sha: str) -> tuple[str, int, int, str]:
    """Memo key: (resolved path, size, mtime_ns, expected digest)."""
    stat = target.stat()
    return (str(target.resolve()), stat.st_size, stat.st_mtime_ns, expected_sha)


def _file_matches(target: Path, expected_sha: str) -> bool:
    """sha256 match, memoized per (path, size, mtime_ns, expected digest).

    The memo exists because a resolve re-runs on every engine/registry load
    and the containers are large; a cache hit skips the full-file hash. The
    stat triple keeps it sound: any rewrite of the file changes size or
    mtime and forces a real hash, so corruption is still healed.
    """
    cache_key = _digest_cache_key(target, expected_sha)
    if _VERIFIED_FILES.get(cache_key):
        return True
    if compute_file_sha256(target) != expected_sha:
        return False
    _VERIFIED_FILES[cache_key] = True
    return True


def _mark_verified(target: Path, expected_sha: str) -> None:
    """Record a freshly computed matching digest (post-download verify)."""
    _VERIFIED_FILES[_digest_cache_key(target, expected_sha)] = True


def _evict_digest_cache(target: Path) -> None:
    """Drop memoized digests for ``target`` before unlinking/replacing it."""
    resolved = str(target.resolve())
    for key in [key for key in _VERIFIED_FILES if key[0] == resolved]:
        del _VERIFIED_FILES[key]


def _default_cache_dir() -> Path:
    import os

    env = os.environ.get("AUTOVISION_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "autovision" / "downloadable_models"


def _download(url: str, target: Path) -> None:
    """Stream ``url`` into ``target`` via a ``.part`` file + ``os.replace``.

    The whole container used to be read into memory and written directly to
    the final path: a crash mid-download left a truncated file at the real
    name, and concurrent readers could bind it. The staged ``.part`` write
    keeps the target name reserved for complete bytes only.
    """
    import urllib.request

    part = target.with_name(f"{target.name}.part")
    logger.info("Downloading model from %s", url)
    try:
        with urllib.request.urlopen(  # noqa: S310  # nosec B310
            url, timeout=Env.DOWNLOAD_CHUNK_TIMEOUT_SEC.value
        ) as response:
            with part.open("wb") as handle:
                while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(part, target)
    except Exception as error:  # noqa: BLE001 (clear network failure surface)
        part.unlink(missing_ok=True)
        raise DownloadableModelError(
            f"Failed to download model from '{url}': {error}"
        ) from error
