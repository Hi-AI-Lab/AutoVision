# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Dataset scan cache (design doc §4.8 D8): per-file manifest verdicts.

Upstream ultralytics caches scan results next to the dataset so repeated
startups skip the full per-file verification pass; the cache key mixes the
scanned-file stats (``relpath, size, mtime_ns``) with the task parameters
that affect the verdicts, so any change triggers an automatic rescan.

AutoVision applies the same idea to the file-manifest validation layer:
"does each referenced image exist / decode, and is its annotation payload
complete?". The cache stores **verdicts** (``"ok" | "background" |
"corrupt"``) plus a census summary — never parsed annotations, which stay
owned by the dataset mmap rows (a second source of annotation truth would
be a correctness hazard).

The cache file ``<root>/.autovision_scan_cache.json`` holds one report per
fingerprint (train/val splits and param variants share one root without
thrashing each other), written atomically (tmp + ``os.replace``). Every
operation is best-effort: a missing, corrupt, or unwritable cache must
never change dataset semantics, only re-do the work.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from autovision.data.file_helpers import open_image

__all__ = [
    "CACHE_FILENAME",
    "SCAN_CACHE_SCHEMA",
    "load_scan_cache",
    "manifest_records",
    "scan_fingerprint",
    "scan_files",
    "scan_manifest",
    "store_scan_cache",
    "summarize",
]

SCAN_CACHE_SCHEMA = "1"
CACHE_FILENAME = ".autovision_scan_cache.json"
#: (size, mtime_ns) sentinel for a manifest path missing from disk: both are
#: impossible values for a real file, so the fingerprint still changes when
#: the file appears.
_MISSING_STAT = (-1, -1)
#: Bound the per-root report map so param-variant experimentation cannot
#: grow the cache file without limit.
_MAX_ENTRIES = 16

logger = logging.getLogger("autovision.data.scan_cache")


def _relative_key(path: Path, root: Path) -> str:
    """Stable cache key for an image path: root-relative when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def manifest_records(
    root: Path, rows: Iterable[Mapping[str, Any]]
) -> list[tuple[str, int, int]]:
    """``(relpath, size, mtime_ns)`` per row's image, sorted for stability.

    Rows whose image is missing from disk use the ``(-1, -1)`` sentinel so
    the fingerprint distinguishes "missing" from any real stat.
    """
    records: list[tuple[str, int, int]] = []
    for row in rows:
        raw = row.get("image_path")
        if raw is None:
            records.append(("<missing-image-path>", *_MISSING_STAT))
            continue
        key = _relative_key(Path(str(raw)), root)
        try:
            stat = Path(str(raw)).stat()
            records.append((key, stat.st_size, stat.st_mtime_ns))
        except OSError:
            records.append((key, *_MISSING_STAT))
    records.sort()
    return records


def scan_fingerprint(
    records: Sequence[tuple[str, int, int]], task_params: Mapping[str, Any]
) -> str:
    """sha256 over the schema, sorted file records, and sorted task params.

    Any change to the file set, a file's ``size``/``mtime_ns``, the schema,
    or any task parameter invalidates the fingerprint and forces a rescan.
    """
    digest = hashlib.sha256()
    digest.update(f"schema={SCAN_CACHE_SCHEMA}\n".encode("utf-8"))
    digest.update(
        json.dumps(sorted(records), separators=(",", ":")).encode("utf-8")
    )
    digest.update(
        json.dumps(dict(sorted(task_params.items())), sort_keys=True).encode("utf-8")
    )
    return digest.hexdigest()


def load_scan_cache(root: Path, fingerprint: str) -> dict[str, Any] | None:
    """Return the cached report for ``fingerprint``, or ``None``.

    A missing file, a schema mismatch, or an unknown fingerprint returns
    ``None`` silently (normal first-run / invalidated-cache cases); a
    JSON-corrupt or unreadable file logs a warning and returns ``None``.
    """
    cache_path = Path(root) / CACHE_FILENAME
    try:
        raw = cache_path.read_text()
    except FileNotFoundError:
        return None
    except OSError as error:
        logger.warning("Scan cache '%s' is unreadable (%s); rescanning.", cache_path, error)
        return None
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        logger.warning("Scan cache '%s' is corrupt (%s); rescanning.", cache_path, error)
        return None
    if not isinstance(document, dict) or document.get("schema") != SCAN_CACHE_SCHEMA:
        return None
    reports = document.get("reports")
    if not isinstance(reports, dict):
        return None
    report = reports.get(fingerprint)
    return dict(report) if isinstance(report, dict) else None


def store_scan_cache(root: Path, fingerprint: str, payload: Mapping[str, Any]) -> None:
    """Merge ``payload`` under ``fingerprint`` into the cache file atomically.

    Existing reports for other fingerprints are preserved (train/val share
    one root). The write is tmp + ``os.replace``; any failure is logged and
    swallowed — caching is an optimization, never a correctness step.
    """
    cache_path = Path(root) / CACHE_FILENAME
    reports: dict[str, Any] = {}
    try:
        existing = json.loads(cache_path.read_text())
        if (
            isinstance(existing, dict)
            and existing.get("schema") == SCAN_CACHE_SCHEMA
            and isinstance(existing.get("reports"), dict)
        ):
            reports = {str(key): value for key, value in existing["reports"].items()}
    except (OSError, json.JSONDecodeError):
        reports = {}  # a corrupt cache file is rewritten, not trusted
    reports.pop(fingerprint, None)
    reports[fingerprint] = dict(payload)  # re-insert: freshest entry sorts last
    while len(reports) > _MAX_ENTRIES:
        del reports[next(iter(reports))]
    document = {"schema": SCAN_CACHE_SCHEMA, "reports": reports}
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=cache_path.parent, prefix=f".{CACHE_FILENAME}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as file:
                json.dump(document, file, indent=2, sort_keys=True)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp_name, cache_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
    except OSError as error:
        logger.warning("Could not write scan cache '%s' (%s); continuing.", cache_path, error)


def summarize(per_file: Mapping[str, str]) -> tuple[dict[str, int], str]:
    """Census counters and the ``"X images, Y backgrounds, Z corrupt"`` line."""
    counters = {
        "images": sum(1 for verdict in per_file.values() if verdict == "ok"),
        "backgrounds": sum(1 for verdict in per_file.values() if verdict == "background"),
        "corrupt": sum(1 for verdict in per_file.values() if verdict == "corrupt"),
    }
    summary = (
        f"{counters['images']} images, {counters['backgrounds']} backgrounds, "
        f"{counters['corrupt']} corrupt"
    )
    return counters, summary


def _is_readable(path: Path, num_channels: int) -> bool:
    """Decode probe using the same decoder ``__getitem__`` will use."""
    try:
        open_image(path, num_channels=num_channels)
    except Exception:  # noqa: BLE001 - any decode failure is the finding
        return False
    return True


def scan_files(
    root: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    num_channels: int = 3,
) -> dict[str, Any]:
    """Verify every manifest row's image and annotation payload.

    Verdicts: ``"corrupt"`` (missing file, undecodable image, or
    unparseable / inconsistent annotation payload), ``"background"`` (valid
    image without boxes), ``"ok"`` (valid image with boxes). The returned
    payload carries per-file verdicts, census counters, and a summary line
    — never parsed annotations.
    """
    per_file: dict[str, str] = {}
    for row in rows:
        raw = row.get("image_path")
        if raw is None:
            per_file["<missing-image-path>"] = "corrupt"
            continue
        path = Path(str(raw))
        key = _relative_key(path, root)
        per_file[key] = _classify(path, row, num_channels)
    counters, summary = summarize(per_file)
    return {"per_file": per_file, "counters": counters, "summary": summary}


def _classify(path: Path, row: Mapping[str, Any], num_channels: int) -> str:
    """One file's verdict: file readable and annotation payload complete."""
    if not path.is_file() or not _is_readable(path, num_channels):
        return "corrupt"
    try:
        boxes = json.loads(str(row.get("bboxes", "[]")))
        labels = json.loads(str(row.get("class_labels", "[]")))
    except json.JSONDecodeError:
        return "corrupt"
    if not isinstance(boxes, list) or not isinstance(labels, list):
        return "corrupt"
    if len(boxes) != len(labels):
        return "corrupt"
    if not boxes:
        return "background"
    return "ok"


def scan_manifest(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    task_params: Mapping[str, Any],
    num_channels: int = 3,
) -> tuple[dict[str, Any], bool]:
    """Load-or-scan the manifest: return ``(report, from_cache)``.

    On a fingerprint hit the cached verdicts are reused verbatim; on a miss
    the full :func:`scan_files` pass runs and its payload is stored. The
    census line is logged either way so repeated startups stay observable.
    """
    fingerprint = scan_fingerprint(manifest_records(root, rows), task_params)
    cached = load_scan_cache(root, fingerprint)
    if cached is not None:
        logger.info("Scan cache hit for '%s': %s.", root, cached.get("summary"))
        return cached, True
    report = scan_files(root, rows, num_channels=num_channels)
    store_scan_cache(root, fingerprint, report)
    logger.info(
        "Scanned '%s' (%s); verdicts cached for reuse.", root, report.get("summary")
    )
    return report, False
