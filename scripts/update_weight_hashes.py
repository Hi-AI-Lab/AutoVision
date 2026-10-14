# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Pin registry-published sha256 hashes into the weight manifest.

Fetches the LFS sha256 of every weight file for each manifest entry and
rewrites ``src/autovision/encoders/weight_hashes.py`` with the pinned tier
enabled. Requires network access to the model registry. Only registry
metadata is fetched (LFS pointers), never the weights themselves.

Each entry is pinned to the commit the default branch pointed at when
this script ran (``revision=<sha>``), so verification stays reproducible
even if the upstream repository moves.

Usage::

    python scripts/update_weight_hashes.py            # pin all entries
    python scripts/update_weight_hashes.py --check    # verify round-trip only
    python scripts/update_weight_hashes.py --repo facebook/dinov2-small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from autovision.encoders import weight_hashes  # noqa: E402
from autovision.encoders.weight_verification import (  # noqa: E402
    _lfs_sha256,
    _render_manifest,
)


def pin_all(repo_filter: str | None) -> dict:
    from dataclasses import replace

    from huggingface_hub import HfApi

    manifest = dict(weight_hashes.WEIGHT_MANIFEST)
    for repo_id, artifact in sorted(manifest.items()):
        if repo_filter and repo_id != repo_filter:
            continue
        print(f"Fetching LFS hashes for {repo_id} ...")
        try:
            revision = HfApi().model_info(repo_id).sha
            hashes = _lfs_sha256(repo_id, revision)
        except Exception as error:  # gated/private repos without a token, offline, ...
            print(f"  WARNING: could not pin {repo_id}: {error}")
            continue
        if not hashes:
            print("  WARNING: no LFS weight files found; leaving unpinned.")
            continue
        manifest[repo_id] = replace(
            artifact,
            revision=revision,
            sha256=dict(hashes),
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", help="pin a single repository instead of the whole manifest"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the manifest file matches its own renderer, then exit",
    )
    args = parser.parse_args()

    if args.check:
        expected = _render_manifest(dict(weight_hashes.WEIGHT_MANIFEST))
        actual = weight_hashes.__file__ and Path(weight_hashes.__file__).read_text()
        if expected != actual:
            print("Manifest file does not match its renderer; regenerate it.")
            return 1
        print("Manifest round-trip OK.")
        return 0

    pinned = pin_all(args.repo)
    manifest_path = Path(weight_hashes.__file__)
    manifest_path.write_text(_render_manifest(pinned))
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
