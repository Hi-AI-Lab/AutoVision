# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Dataset download registry for the public detection datasets.

Model weights have their own pinned-manifest pipeline
(``encoders.weight_verification``); datasets are plain public artifacts.
The registry records official URLs and, when the source publishes them,
sizes and sha256 digests — entries without a digest WARN at download time
instead of silently skipping verification. Nothing here runs at import
time and no test requires network access.
"""

from __future__ import annotations

import hashlib
import http.client
import logging
import os
import posixpath
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlparse

from autovision.errors import AutoVisionError
from autovision.runtime.env import Env

__all__ = ["DATASET_REGISTRY", "DatasetFile", "DatasetSpec", "download_dataset"]

logger = logging.getLogger("autovision")


@dataclass(frozen=True)
class DatasetFile:
    url: str
    size_bytes: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    description: str
    homepage: str
    files: tuple[DatasetFile, ...] = field(default=())
    #: Set when the source requires manual acquisition (registration form,
    #: license click-through). ``download_dataset`` refuses with instructions.
    manual_instructions: str | None = None
    license_note: str = ""


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "coco2017-val": DatasetSpec(
        description="COCO 2017 validation images + annotations (detection).",
        homepage="https://cocodataset.org",
        files=(
            DatasetFile(url="https://images.cocodataset.org/zips/val2017.zip"),
            DatasetFile(
                url="https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            ),
        ),
        license_note="Images: Flickr terms of use; annotations: CC BY 4.0.",
    ),
    "coco2017-train": DatasetSpec(
        description="COCO 2017 train images (118k, ~18GB) + annotations.",
        homepage="https://cocodataset.org",
        files=(
            DatasetFile(url="https://images.cocodataset.org/zips/train2017.zip"),
            DatasetFile(
                url="https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            ),
        ),
        license_note="Images: Flickr terms of use; annotations: CC BY 4.0.",
    ),
    "voc2007-trainval": DatasetSpec(
        description="PASCAL VOC 2007 trainval (~450MB tar).",
        homepage="http://host.robots.ox.ac.uk/pascal/VOC/",
        files=(
            DatasetFile(
                url=(
                    "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/"
                    "VOCtrainval_06-Nov-2007.tar"
                )
            ),
        ),
        license_note="CLUF at the VOC homepage; non-commercial research.",
    ),
    "visdrone2017-det": DatasetSpec(
        description="VisDrone-DET2017 (manual acquisition required).",
        homepage="https://github.com/VisDrone/VisDrone-Dataset",
        manual_instructions=(
            "VisDrone is distributed by the AISKYEYE set, Lab of MEDIS, "
            "Tianjin University via a request form / OneDrive+GoogleDrive "
            "mirrors listed at the homepage. Download DET-train/val and "
            "unpack under your dataset root, then point the train config's "
            "data dict at the annotation directories."
        ),
        license_note="Free for research; see homepage for terms.",
    ),
}


class UnknownDatasetError(AutoVisionError):
    pass


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discard(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - cleanup is best effort
        logger.warning("Could not remove partial download %s", path)


#: Statuses that indicate a transient server-side condition worth retrying.
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
#: Total transfer attempts (initial + retries) before giving up.
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SEC = 5.0
_BACKOFF_CAP_SEC = 60.0
#: Headroom absorbs extraction side artifacts, not just the archive itself.
_DISK_HEADROOM_FACTOR = 1.5
#: Only pre-flight when the pinned size is large enough to matter (> 1 MiB).
_DISK_CHECK_MIN_SIZE_BYTES = 1 << 20


class _IncompleteTransfer(Exception):
    """The response body ended before its Content-Length; retryable.

    Raised internally so the retry loop can treat a dropped transfer like
    any other network failure: keep the ``*.part`` prefix and resume.
    """


def _sleep(seconds: float) -> None:
    """Backoff seam, isolated so tests can inject a no-op sleeper."""
    time.sleep(seconds)


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff for the failed ``attempt`` (0-based): 5s, 10s..."""
    return min(_BACKOFF_BASE_SEC * 2**attempt, _BACKOFF_CAP_SEC)


def _part_path(dest: Path) -> Path:
    """Deterministic staging path for ``dest`` (hidden ``*.part`` file).

    The name carries no pid/uuid suffix so a later run of the same download
    resumes the same transfer. Two concurrent downloads of one destination
    would share the file, but the pinned sha256 fails closed on corruption.
    """
    return dest.parent / f".{dest.name}.part"


def _content_length(response: http.client.HTTPResponse) -> int | None:
    """Content-Length of the current response, or ``None`` when unusable."""
    raw = response.getheader("Content-Length")
    try:
        return int(raw) if raw else None
    except (TypeError, ValueError):
        return None


def _content_range_start(header: str | None) -> int | None:
    """Start offset carried by a ``Content-Range`` header value, if any."""
    if not header:
        return None
    value = header.strip().lower()
    if value.startswith("bytes"):
        value = value[len("bytes") :].strip()
    span = value.split("/", 1)[0].strip()
    start_text = span.split("-", 1)[0].strip()
    try:
        return int(start_text)
    except ValueError:
        return None


def _ensure_disk_space(spec: DatasetFile, dest_dir: Path, name: str) -> None:
    """Pre-flight free space at ``dest_dir`` with a 1.5x safety factor.

    Skipped when the registry pins no size or the size is at most 1 MiB:
    the headroom check only pays off for the large archives it protects.
    """
    if spec.size_bytes is None or spec.size_bytes <= _DISK_CHECK_MIN_SIZE_BYTES:
        return
    required = int(_DISK_HEADROOM_FACTOR * spec.size_bytes)
    free = shutil.disk_usage(dest_dir).free
    if free < required:
        raise AutoVisionError(
            f"Insufficient disk space for '{name}': {free} bytes free under "
            f"'{dest_dir}', need {required} (1.5x the pinned "
            f"{spec.size_bytes} bytes). Free up space or pick another "
            "output directory."
        )


def _download_file(spec: DatasetFile, dest: Path) -> Path:
    """Download ``spec.url`` to ``dest`` atomically with retry and resume.

    The bytes stream to a deterministic ``*.part`` file in the destination
    directory, are verified against the pinned size/sha256, and only then
    move onto the final path with :func:`os.replace`; the final path never
    holds a partial download. Transient failures (status 408/429/5xx and
    dropped connections) retry with exponential backoff and resume from the
    ``*.part`` prefix via an HTTP ``Range`` request (a 206 answer is only
    accepted when its ``Content-Range`` start matches the local prefix; a
    416 answer or a Range-ignoring 200 answer resets the transfer). Other
    4xx codes fail fast. Content-invalidating failures (sha256/size
    mismatch) delete the partial file; a resumable prefix is kept for a
    later run.
    """
    if Env.OFFLINE.value:
        raise AutoVisionError(
            f"AUTOVISION_OFFLINE=1; refusing to download '{dest.name}'. "
            "Pre-populate the archive or dataset root, or unset "
            "AUTOVISION_OFFLINE."
        )
    if Env.REQUIRE_DATASET_HASHES.value and spec.sha256 is None:
        raise AutoVisionError(
            f"Dataset artifact '{dest.name}' has no pinned sha256 and "
            "AUTOVISION_REQUIRE_DATASET_HASHES=1; refusing the download. "
            "Register a mirrored artifact with a digest, or unset the flag "
            "to accept unverified downloads."
        )
    if spec.sha256 is None:
        logger.warning(
            "Dataset artifact %s has no pinned sha256 in the registry; "
            "integrity is not verified for this download.",
            dest.name,
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    scheme = urlparse(spec.url).scheme.lower()
    if scheme not in ("https", "http"):
        raise AutoVisionError(
            f"Dataset artifact '{dest.name}' uses unsupported URL scheme "
            f"'{scheme}'; only https/http are allowed."
        )
    _ensure_disk_space(spec, dest.parent, dest.name)
    part = _part_path(dest)
    timeout = Env.DOWNLOAD_CHUNK_TIMEOUT_SEC.value
    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        offset = part.stat().st_size if part.exists() else 0
        resume_requested = offset > 0
        progress = None
        try:
            if resume_requested:
                request = urllib.request.Request(
                    spec.url, headers={"Range": f"bytes={offset}-"}
                )
            else:
                request = urllib.request.Request(spec.url)
            # scheme is restricted to https/http above.
            with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
                status = response.status
                if resume_requested and status == 206:
                    start = _content_range_start(
                        response.getheader("Content-Range")
                    )
                    if start != offset:
                        # The server's resume point disagrees with the local
                        # prefix; the combined bytes could never verify.
                        _discard(part)
                        continue
                    mode = "ab"
                elif resume_requested and status == 200:
                    # Server ignored the Range header: appending its full
                    # body would mis-assemble the file, so restart from zero.
                    _discard(part)
                    continue
                elif resume_requested:
                    raise AutoVisionError(
                        f"Server returned unexpected status {status} for a "
                        f"Range resume of '{dest.name}'."
                    )
                elif status == 206:
                    # A 206 without a sent Range is a server bug unless the
                    # Content-Range carries the full body from byte zero.
                    start = _content_range_start(
                        response.getheader("Content-Range")
                    )
                    if start != 0:
                        raise AutoVisionError(
                            f"Server returned 206 with Content-Range start "
                            f"{start} without a Range request for "
                            f"'{dest.name}'."
                        )
                    mode = "wb"
                else:
                    mode = "wb"
                content_length = _content_length(response)
                try:
                    from tqdm import tqdm

                    progress = tqdm(
                        total=(
                            None
                            if content_length is None
                            else offset + content_length
                        ),
                        initial=offset,
                        unit="B",
                        unit_scale=True,
                        desc=dest.name,
                        leave=False,
                    )
                except ImportError:  # pragma: no cover - tqdm is a core dependency
                    progress = None
                received = 0
                with part.open(mode) as handle:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
                        received += len(chunk)
                        if progress is not None:
                            progress.update(len(chunk))
                    handle.flush()
                    os.fsync(handle.fileno())
            if content_length is not None and received != content_length:
                raise _IncompleteTransfer(received, content_length)
            size = part.stat().st_size
            if spec.size_bytes is not None and size != spec.size_bytes:
                _discard(part)
                raise AutoVisionError(
                    f"Downloaded '{dest.name}' has {size} bytes, expected "
                    f"{spec.size_bytes}."
                )
            if spec.sha256 is not None:
                actual = _digest(part)
                if actual != spec.sha256:
                    _discard(part)
                    raise AutoVisionError(
                        f"sha256 mismatch for '{dest.name}': expected "
                        f"{spec.sha256}, got {actual}; partial download deleted."
                    )
            os.replace(part, dest)
            return dest
        except urllib.error.HTTPError as error:
            last_error = error
            # Release the error response's socket before the next attempt.
            error.close()
            if resume_requested and error.code == 416:
                # The Range is unsatisfiable: the prefix is stale, restart.
                # Not slept on: unlike 5xx, this is not server overload and
                # the very next attempt carries no Range header at all.
                _discard(part)
                continue
            if error.code in _RETRYABLE_STATUS_CODES:
                if attempt + 1 >= _MAX_ATTEMPTS:
                    break
                _sleep(_backoff_delay(attempt))
                continue
            # Non-retryable 4xx: fail fast. The prefix on disk is still a
            # valid resume point for a fixed/authorized later attempt.
            break
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            OSError,
            _IncompleteTransfer,
        ) as error:
            # Dropped transfer: the on-disk prefix stays valid for resume.
            last_error = error
            if attempt + 1 >= _MAX_ATTEMPTS:
                break
            _sleep(_backoff_delay(attempt))
            continue
        except AutoVisionError:
            raise
        except Exception as error:
            # Unclassified failure: do not keep state around it.
            _discard(part)
            raise AutoVisionError(
                f"Failed to download '{dest.name}' from '{spec.url}': {error}"
            ) from error
        finally:
            if progress is not None:
                progress.close()
    raise AutoVisionError(
        f"Failed to download '{dest.name}' from '{spec.url}': "
        f"{last_error or 'transfer did not complete'}"
    ) from last_error


def _reject_unsafe_member(name: str, archive: Path) -> None:
    """Reject zip/tar members that are absolute or contain ``..`` traversal."""
    if not name or "\x00" in name:
        raise AutoVisionError(
            f"Archive '{archive}' contains an invalid member name {name!r}; "
            "refusing extraction."
        )
    normalized = name.replace("\\", "/")
    windows = PureWindowsPath(name)
    if (
        normalized.startswith("/")
        or PurePosixPath(normalized).is_absolute()
        or windows.drive
        or windows.is_absolute()
    ):
        raise AutoVisionError(
            f"Archive '{archive}' contains an absolute-path member {name!r}; "
            "refusing extraction."
        )
    if ".." in PurePosixPath(normalized).parts:
        raise AutoVisionError(
            f"Archive '{archive}' contains a path-traversal member {name!r}; "
            "refusing extraction."
        )


def _reject_unsafe_link(member: tarfile.TarInfo, archive: Path) -> None:
    linkname = member.linkname
    normalized = linkname.replace("\\", "/")
    windows = PureWindowsPath(linkname)
    if (
        not normalized
        or "\x00" in linkname
        or normalized.startswith("/")
        or windows.drive
        or windows.is_absolute()
    ):
        raise AutoVisionError(
            f"Archive '{archive}' member {member.name!r} has an absolute or "
            f"invalid link target {linkname!r}; refusing extraction."
        )
    # Symlink targets are relative to the member's directory, hard-link
    # targets to the archive root (tar semantics).
    if member.issym():
        base = PurePosixPath(member.name.replace("\\", "/")).parent
    else:
        base = PurePosixPath(".")
    resolved = posixpath.normpath(str(base / normalized))
    if resolved == ".." or resolved.startswith("../") or resolved.startswith("/"):
        raise AutoVisionError(
            f"Archive '{archive}' member {member.name!r} links outside the "
            f"extraction root ({linkname!r}); refusing extraction."
        )


def _validate_tar_members(
    bundle: tarfile.TarFile, archive: Path
) -> list[tarfile.TarInfo]:
    """Equivalent of ``tarfile.data_filter`` for builds lacking ``filter``.

    Validates every member before extracting anything and strips setuid/
    setgid/sticky bits. Only regular files, directories, and links whose
    targets stay inside the extraction root are allowed.
    """
    members = bundle.getmembers()
    for member in members:
        _reject_unsafe_member(member.name, archive)
        if member.isreg() or member.isdir():
            pass
        elif member.issym() or member.islnk():
            _reject_unsafe_link(member, archive)
        else:
            raise AutoVisionError(
                f"Archive '{archive}' contains unsupported member "
                f"{member.name!r} (not a regular file, directory, or link); "
                "refusing extraction."
            )
        member.mode &= 0o777
    return members


def _extract(archive: Path, out_dir: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                _reject_unsafe_member(member.filename, archive)
            bundle.extractall(out_dir)  # nosec B202
        return
    with tarfile.open(archive) as bundle:
        try:
            # tarfile data_filter validates members (links/
            # device files/traversal) before extraction.
            bundle.extractall(out_dir, filter="data")
        except TypeError:
            # Python builds without the tarfile filter backport (3.10 builds
            # older than 3.10.12/any build lacking the keyword). Applying the
            # equivalent checks by hand keeps extraction safe rather than
            # silently extracting an unvalidated member. Reopen the archive:
            # a mid-extraction failure may have consumed the stream.
            with tarfile.open(archive) as fallback:
                # Every member is validated by
                # _validate_tar_members before extraction.
                fallback.extractall(  # nosec B202
                    out_dir, members=_validate_tar_members(fallback, archive)
                )


def _verify_cached_archive(spec: DatasetFile, archive: Path) -> None:
    """Fail closed when a cached archive no longer matches its pins."""
    if spec.size_bytes is not None:
        actual_size = archive.stat().st_size
        if actual_size != spec.size_bytes:
            raise AutoVisionError(
                f"cached archive failed size verification for '{archive}': "
                f"{actual_size} bytes, expected {spec.size_bytes}; delete the "
                "archive and retry."
            )
    if spec.sha256 is not None:
        actual = _digest(archive)
        if actual != spec.sha256:
            raise AutoVisionError(
                f"cached archive failed sha256 verification for '{archive}': "
                f"expected {spec.sha256}, got {actual}; delete the archive "
                "and retry."
            )
    if spec.size_bytes is None and spec.sha256 is None:
        logger.warning(
            "Cached dataset archive %s has no pinned size_bytes or sha256; "
            "integrity is not verified before extraction.",
            archive.name,
        )


def download_dataset(name: str, out_dir: Path | str) -> Path:
    """Download + extract a registered dataset under ``out_dir/<name>``.

    Downloads are written to a deterministic ``*.part`` file, verified, and
    moved atomically onto the final path. Interrupted transfers retry with
    backoff and resume via HTTP Range requests, across retries and later
    runs. Cached archives are re-verified against the registry's pinned
    size/sha256 before extraction. Already-extracted archives are skipped,
    so the call is idempotent at file granularity.
    """
    spec = DATASET_REGISTRY.get(name)
    if spec is None:
        raise UnknownDatasetError(
            f"Unknown dataset '{name}'. Registered: {sorted(DATASET_REGISTRY)}"
        )
    if spec.manual_instructions is not None:
        raise AutoVisionError(
            f"Dataset '{name}' requires manual acquisition. {spec.manual_instructions}"
        )
    if Env.REQUIRE_DATASET_HASHES.value:
        unpinned = [Path(entry.url).name for entry in spec.files if entry.sha256 is None]
        if unpinned:
            raise AutoVisionError(
                f"AUTOVISION_REQUIRE_DATASET_HASHES=1 but dataset '{name}' "
                f"has no pinned sha256 for {unpinned}."
            )
    root = Path(out_dir) / name
    root.mkdir(parents=True, exist_ok=True)
    for entry in spec.files:
        archive_name = Path(entry.url).name
        archive = root / archive_name
        if archive.exists():
            _verify_cached_archive(entry, archive)
        else:
            _download_file(entry, archive)
        _extract(archive, root)
    return root
