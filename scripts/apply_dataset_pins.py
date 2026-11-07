#!/usr/bin/env python
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Apply computed sha256/size pins into the dataset registry (H1-9).

Pairs with ``compute_dataset_pins.py``: that script computes pins from a
trusted copy of an official archive, this one rewrites the matching
``DatasetFile(...)`` entries in ``src/autovision/data/downloads.py`` so the
pins land in the registry without hand-editing. The same URL may appear in
several specs (the COCO annotations archive is referenced by both the val
and train specs); every occurrence gets the pin.

Input is a JSON mapping of registry URL to its pin::

    {
        "https://images.cocodataset.org/zips/val2017.zip": {
            "sha256": "<64 hex chars>",
            "size_bytes": 815585330
        }
    }

Usage::

    # Dry run: print the patch, write nothing (exit 1 when pins are missing).
    python scripts/apply_dataset_pins.py pins.json --check

    # Apply, then re-import the registry and verify every pin landed.
    python scripts/apply_dataset_pins.py pins.json

    # Replace pins that already exist (e.g. re-pinning after a mirror move).
    python scripts/apply_dataset_pins.py pins.json --force

After applying, remove the URL from ``UNPINNED_ALLOWLIST`` in
``tests/test_data_hardening.py`` and record the source mirror and date in
``docs/provenance/LEDGER.md`` (the pin is only as trustworthy as the bytes
it was computed from).

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TARGET = _REPO_ROOT / "src" / "autovision" / "data" / "downloads.py"

_DATASET_FILE_RE = re.compile(r"DatasetFile\(")
_URL_RE = re.compile(r"""url\s*=\s*["']([^"']+)["']""")

_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class Pin:
    """A verified artifact pin keyed by registry URL."""

    sha256: str
    size_bytes: int


def _load_pins(path: Path) -> dict[str, Pin]:
    """Load and validate the URL-to-pin JSON mapping."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("error: pins file must be a JSON object keyed by URL")
    pins: dict[str, Pin] = {}
    for url, entry in raw.items():
        try:
            sha256 = str(entry["sha256"]).lower()
            size_bytes = int(entry["size_bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(
                f"error: pin for '{url}' needs 'sha256' (str) and "
                f"'size_bytes' (int): {error}"
            ) from error
        if not _SHA256_RE.match(sha256):
            raise SystemExit(f"error: pin for '{url}' is not a sha256 hex digest")
        if size_bytes <= 0:
            raise SystemExit(f"error: pin for '{url}' has a non-positive size")
        pins[url] = Pin(sha256=sha256, size_bytes=size_bytes)
    return pins


def _iter_dataset_file_blocks(source: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, text)`` spans of every ``DatasetFile(...)`` call."""
    blocks: list[tuple[int, int, str]] = []
    for match in _DATASET_FILE_RE.finditer(source):
        open_index = match.end() - 1
        depth = 0
        for index in range(open_index, len(source)):
            char = source[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    blocks.append(
                        (match.start(), index + 1, source[match.start() : index + 1])
                    )
                    break
        else:
            raise ValueError("unbalanced parentheses in a DatasetFile entry")
    return blocks


def _format_entry(url: str, pin: Pin, indent: str) -> str:
    """Render the canonical multi-line ``DatasetFile`` entry with the pin."""
    inner = indent + "    "
    return (
        f"{indent}DatasetFile(\n"
        f'{inner}url="{url}",\n'
        f"{inner}size_bytes={pin.size_bytes},\n"
        f'{inner}sha256="{pin.sha256}",\n'
        f"{indent})"
    )


def _patch_source(
    source: str, pins: dict[str, Pin], *, force: bool, path: Path
) -> tuple[str, list[str], int]:
    """Rewrite matching ``DatasetFile`` entries; return (source, notes, changed)."""
    notes: list[str] = []
    changed = 0
    matched: set[str] = set()
    result: list[str] = []
    cursor = 0
    for start, end, block in _iter_dataset_file_blocks(source):
        url_match = _URL_RE.search(block)
        if url_match is None:
            continue
        url = url_match.group(1)
        if url not in pins:
            continue
        pin = pins[url]
        matched.add(url)
        # The block span starts at "DatasetFile(" — the entry's indentation
        # lives on the same line *before* it; swallow it so the rebuilt
        # entry re-indents cleanly.
        line_start = source.rfind("\n", 0, start) + 1
        prefix = source[line_start:start]
        indent = prefix if prefix.strip() == "" else "    "
        existing_sha = re.search(r"""sha256\s*=\s*["']([^"']*)["']""", block)
        existing_size = re.search(r"size_bytes\s*=\s*(\d+)", block)
        if existing_sha and existing_sha.group(1) and not force:
            same = existing_sha.group(1).lower() == pin.sha256 and (
                existing_size is not None
                and int(existing_size.group(1)) == pin.size_bytes
            )
            if not same:
                raise SystemExit(
                    f"error: '{url}' is already pinned with different values "
                    f"in {path}; re-pin intentionally with --force."
                )
            notes.append(f"  = {url} (already pinned, unchanged)")
            continue
        result.append(source[cursor:line_start])
        result.append(_format_entry(url, pin, indent))
        cursor = end
        changed += 1
        notes.append(f"  + {url}")
    result.append(source[cursor:])
    unmatched = sorted(set(pins) - matched)
    if unmatched:
        raise SystemExit(
            "error: no DatasetFile entry found in "
            f"{path} for: {', '.join(unmatched)}"
        )
    return "".join(result), notes, changed


def _verify_registry(pins_path: Path, pins: dict[str, Pin], target: Path) -> None:
    """Load the patched registry file in a fresh interpreter; check pins."""
    code = (
        "import importlib.util, json, sys\n"
        "spec = importlib.util.spec_from_file_location(\n"
        "    'registry_under_test', sys.argv[2])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['registry_under_test'] = module\n"
        "spec.loader.exec_module(module)\n"
        "pins = json.load(open(sys.argv[1], encoding='utf-8'))\n"
        "found = {}\n"
        "for spec_entry in module.DATASET_REGISTRY.values():\n"
        "    for item in spec_entry.files:\n"
        "        found[item.url] = (item.sha256, item.size_bytes)\n"
        "bad = []\n"
        "for url, pin in pins.items():\n"
        "    actual = found.get(url)\n"
        "    expected = (pin['sha256'].lower(), int(pin['size_bytes']))\n"
        "    if actual != expected:\n"
        "        bad.append((url, actual, expected))\n"
        "for url, actual, expected in bad:\n"
        "    print(f'  ! {url}: registry={actual} expected={expected}',\n"
        "          file=sys.stderr)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(pins_path), str(target)],
        cwd=_REPO_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "error: the patched registry file does not carry the pins "
            "(import or value mismatch); investigate before committing."
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite DatasetFile entries in the dataset registry with "
            "computed sha256/size pins (H1-9)."
        )
    )
    parser.add_argument(
        "pins",
        type=Path,
        metavar="PINS_JSON",
        help="JSON file mapping registry URL to {sha256, size_bytes}.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=_DEFAULT_TARGET,
        help="Registry source file (default: src/autovision/data/downloads.py).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Dry run: print what would change and exit 1 if anything would.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite entries that already carry a different pin.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Patch the registry with the given pins; 0 when fully applied."""
    parser = _parser()
    args = parser.parse_args(argv)
    pins = _load_pins(args.pins)
    source = args.file.read_text(encoding="utf-8")
    try:
        patched, notes, changed = _patch_source(
            source, pins, force=args.force, path=args.file
        )
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    for note in notes:
        print(note)
    if changed == 0:
        print("nothing to do: every URL is already pinned with matching values.")
        return 0
    if args.check:
        print(f"check mode: {changed} entry(ies) would change; no files written.")
        return 1
    args.file.write_text(patched, encoding="utf-8")
    print(f"patched {changed} entry(ies) in {args.file}; verifying…")
    _verify_registry(args.pins, pins, args.file)
    print("verified: the patched registry file imports and carries every pin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
