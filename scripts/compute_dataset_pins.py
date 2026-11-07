#!/usr/bin/env python
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Compute paste-ready sha256/size pins for dataset registry entries (H1-9).

``DATASET_REGISTRY`` in ``src/autovision/data/downloads.py`` verifies
downloads against pinned ``sha256``/``size_bytes`` values, but the official
archives only carry pins once someone computes them from a trusted copy.
This script does exactly that computation and prints a ``DatasetFile(...)``
snippet that can be pasted into the registry.

Usage::

    # From an already-downloaded official archive:
    python scripts/compute_dataset_pins.py val2017.zip \\
        --url https://images.cocodataset.org/zips/val2017.zip

    # Straight from the source (needs network access):
    python scripts/compute_dataset_pins.py \\
        https://images.cocodataset.org/zips/val2017.zip

After filling the pin into the registry, record the mirror, date, and
upstream page the bytes came from in ``docs/provenance/LEDGER.md``, and
remove the URL from ``UNPINNED_ALLOWLIST`` in ``tests/test_data_hardening.py``
(an empty allowlist is the target state).

Standard library only: ``hashlib`` chunks files and archives; ``urllib`` is
exercised only when a target is an http(s) URL.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn
from urllib.parse import urlparse

_CHUNK_BYTES = 1 << 20

_PLACEHOLDER_URL = "<paste the registry URL here>"


def _sha256_and_size(chunks: Iterator[bytes]) -> tuple[str, int]:
    """Consume ``chunks`` and return ``(sha256 hex digest, total bytes)``."""
    digest = hashlib.sha256()
    size = 0
    for chunk in chunks:
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _file_chunks(path: Path) -> Iterator[bytes]:
    """Stream ``path`` in fixed-size blocks (constant memory for 18GB tars)."""
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            yield chunk


def _url_chunks(url: str) -> Iterator[bytes]:
    """Stream an http(s) ``url`` in fixed-size blocks."""
    # The scheme is restricted to https/http by the caller before this runs.
    with urllib.request.urlopen(url) as response:  # nosec B310
        while chunk := response.read(_CHUNK_BYTES):
            yield chunk


def _snippet(url: str, sha256: str, size_bytes: int) -> str:
    """Format the paste-ready ``DatasetFile`` registry entry."""
    return (
        "DatasetFile(\n"
        f'    url="{url}",\n'
        f"    size_bytes={size_bytes},\n"
        f'    sha256="{sha256}",\n'
        "),"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print sha256, size, and a paste-ready DatasetFile snippet for "
            "dataset artifacts (H1-9)."
        )
    )
    parser.add_argument(
        "targets",
        nargs="+",
        metavar="TARGET",
        help="Local path or https URL of an official dataset artifact.",
    )
    parser.add_argument(
        "--url",
        help=(
            "Registry URL to print in the snippet when the target is a "
            "local file (required for a paste-ready snippet)."
        ),
    )
    return parser


def _fail(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.exit(2, f"error: {message}\n")


def main(argv: list[str] | None = None) -> int:
    """Hash every target and print one snippet per target; 0 on success."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.url and len(args.targets) != 1:
        _fail(parser, "--url applies to exactly one target.")

    failures = 0
    for target in args.targets:
        scheme = urlparse(target).scheme.lower()
        try:
            if scheme in ("http", "https"):
                registry_url = target
                sha256, size_bytes = _sha256_and_size(_url_chunks(target))
            elif scheme:
                _fail(parser, f"unsupported URL scheme '{scheme}': '{target}'")
            else:
                path = Path(target)
                if not path.is_file():
                    raise FileNotFoundError(f"no such file: '{path}'")
                registry_url = args.url or _PLACEHOLDER_URL
                sha256, size_bytes = _sha256_and_size(_file_chunks(path))
        except (OSError, urllib.error.URLError) as error:
            print(f"{target}: FAILED: {error}", file=sys.stderr)
            failures += 1
            continue
        if registry_url == _PLACEHOLDER_URL:
            print(
                f"# {target}: pass --url REGISTRY_URL for a paste-ready snippet.",
                file=sys.stderr,
            )
        print(f"# {target}")
        print(f"# sha256: {sha256}")
        print(f"# size_bytes: {size_bytes}")
        print(_snippet(registry_url, sha256, size_bytes))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
