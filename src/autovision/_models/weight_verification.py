# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Verified pretrained-weight loading (supply-chain integrity, §3.2/§6.3).

Every pretrained download goes through :func:`load_pretrained_verified`:

1. ``VTRAIN_MIRROR_URL`` is bridged to the HF endpoint mechanism when the
   process has not already configured ``HF_ENDPOINT`` (the mirror consumer).
2. ``VTRAIN_OFFLINE=1`` resolves from the local HF cache only; a missing
   snapshot raises an actionable error instead of attempting network I/O.
3. The snapshot's weight files are sha256-verified against the manifest in
   :mod:`autovision._models.weight_hashes`. Pinned hashes verify fully
   offline; unpinned entries fall back to the LFS sha256 published in the
   registry metadata (transfer-integrity tier).
4. On a hash mismatch the repo cache is deleted and the download retried
   once; a second mismatch raises :class:`WeightVerificationError`.

The manifest is machine-pinned via ``scripts/update_weight_hashes.py`` so
the pinned tier grows over time without trusting first use.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autovision._env import Env

__all__ = [
    "WeightArtifact",
    "WeightVerificationError",
    "get_weight_artifact",
    "load_pretrained_verified",
]

logger = logging.getLogger("autovision")

_WEIGHT_PATTERNS = ("*.safetensors", "*.bin")
_SNAPSHOT_PATTERNS = ("config.json", "*.safetensors", "*.bin")
_RETRY_MESSAGE = (
    "If the failure persists the download source may be compromised or "
    "truncated: set VTRAIN_MIRROR_URL to a trusted mirror, or download the "
    "model manually and pass a local directory via model_args."
)


class WeightVerificationError(RuntimeError):
    """A pretrained weight file failed sha256 verification after one retry."""


@dataclass(frozen=True)
class WeightArtifact:
    """Supply-chain metadata for one pretrained checkpoint repository.

    ``sha256`` maps weight filenames to pinned hashes; an empty dict means
    the entry is unpinned and verification falls back to the LFS sha256
    from the registry metadata (see module docstring).
    """

    repo_id: str
    license: str
    source: str = "huggingface"
    revision: str | None = None
    sha256: dict[str, str] = field(default_factory=dict)


def get_weight_artifact(repo_id: str) -> WeightArtifact | None:
    """Manifest entry for ``repo_id`` (``None`` for unlisted repositories)."""
    # Late import: the manifest data module re-exports WeightArtifact from
    # here, so the dependency only points one way at call time.
    from autovision._models.weight_hashes import WEIGHT_MANIFEST

    return WEIGHT_MANIFEST.get(repo_id)


def compute_file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of a file (constant memory for multi-GB weights)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _lfs_sha256(repo_id: str, revision: str | None) -> dict[str, str]:
    """Registry-published LFS sha256 per weight filename (network tier)."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    hashes: dict[str, str] = {}
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None)
        if lfs and sibling.rfilename.endswith((".safetensors", ".bin")):
            hashes[sibling.rfilename] = lfs.sha256
    return hashes


def _delete_repo_cache(snapshot_dir: Path) -> None:
    """Delete the HF cache entry (``models--org--name``) for a snapshot."""
    for parent in snapshot_dir.parents:
        if parent.name.startswith("models--"):
            shutil.rmtree(parent, ignore_errors=True)
            logger.info("Deleted corrupted weight cache '%s'.", parent)
            return
    # Snapshot outside the HF cache layout (test/custom dir): remove the
    # snapshot itself so the retry re-downloads.
    shutil.rmtree(snapshot_dir, ignore_errors=True)


def _verify_snapshot(snapshot_dir: Path, artifact: WeightArtifact) -> str:
    """Verify every weight file in the snapshot.

    Returns the verification tier ("pinned" or "lfs-metadata"); raises
    :class:`WeightVerificationError` on the first mismatch.
    """
    weight_files = sorted(
        path
        for pattern in _WEIGHT_PATTERNS
        for path in snapshot_dir.glob(pattern)
        if path.is_file()
    )
    if not weight_files:
        raise WeightVerificationError(
            f"Snapshot of '{artifact.repo_id}' contains no weight files "
            f"matching {_WEIGHT_PATTERNS}; the download is incomplete."
        )
    unpinned = [
        path.name for path in weight_files if path.name not in artifact.sha256
    ]
    registry_hashes: dict[str, str] = {}
    if unpinned:
        registry_hashes = _lfs_sha256(artifact.repo_id, artifact.revision)
    tier = "pinned" if not unpinned else "lfs-metadata"
    for path in weight_files:
        expected = artifact.sha256.get(path.name) or registry_hashes.get(path.name)
        if not expected:
            raise WeightVerificationError(
                f"No sha256 available for '{path.name}' of '{artifact.repo_id}' "
                "(not pinned in the manifest and not published in the "
                "registry metadata). Run scripts/update_weight_hashes.py to "
                "pin it."
            )
        actual = compute_file_sha256(path)
        if actual != expected:
            raise WeightVerificationError(
                f"sha256 mismatch for '{artifact.repo_id}/{path.name}': "
                f"expected {expected}, got {actual}. {_RETRY_MESSAGE}"
            )
        logger.info("Verified %s (%s tier).", path.name, tier)
    return tier


def _download_snapshot(
    repo_id: str, revision: str | None, local_files_only: bool = False
) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(_SNAPSHOT_PATTERNS),
            local_files_only=local_files_only,
        )
    )


def load_pretrained_verified(
    model_cls: Any,
    repo_id: str,
    config_kwargs: dict[str, Any] | None = None,
) -> Any:
    """``model_cls.from_pretrained`` wrapped with the verification flow."""
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:  # hub < 1.0 re-exports at the top level
        from huggingface_hub import LocalEntryNotFoundError  # noqa: F401

    artifact = get_weight_artifact(repo_id)
    if artifact is None:
        logger.warning(
            "Model '%s' is not in the weight manifest; loading WITHOUT "
            "sha256 verification.", repo_id,
        )
        return model_cls.from_pretrained(repo_id, **(config_kwargs or {}))
    revision = artifact.revision

    # Mirror consumer (§3.2): VTRAIN_MIRROR_URL -> HF endpoint, unless the
    # process already pinned one explicitly.
    mirror_url = Env.MIRROR_URL.value
    if mirror_url and not os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = mirror_url
        logger.info("Using weight mirror: %s", mirror_url)

    if Env.OFFLINE.value:
        try:
            snapshot_dir = _download_snapshot(
                repo_id, revision, local_files_only=True
            )
        except LocalEntryNotFoundError as error:
            raise RuntimeError(
                f"VTRAIN_OFFLINE=1 and no cached snapshot for '{repo_id}'. "
                "Run once without VTRAIN_OFFLINE to populate the cache, or "
                "download the model manually and point model_args at it."
            ) from error
        if artifact.sha256:
            _verify_snapshot(snapshot_dir, artifact)  # offline-capable tier
        else:
            logger.warning(
                "Offline load of '%s' skipped sha256 verification (entry is "
                "unpinned; run scripts/update_weight_hashes.py).", repo_id,
            )
        return model_cls.from_pretrained(snapshot_dir, **(config_kwargs or {}))

    snapshot_dir = _download_snapshot(repo_id, revision)
    try:
        _verify_snapshot(snapshot_dir, artifact)
    except WeightVerificationError:
        # One delete-cache + re-download retry before failing hard.
        logger.warning("Hash mismatch for '%s'; retrying once.", repo_id)
        _delete_repo_cache(snapshot_dir)
        snapshot_dir = _download_snapshot(repo_id, revision)
        _verify_snapshot(snapshot_dir, artifact)  # raises on second failure
    return model_cls.from_pretrained(snapshot_dir, **(config_kwargs or {}))


def _render_manifest(manifest: dict[str, WeightArtifact]) -> str:
    """Deterministic manifest source (single formatting source, used by
    ``scripts/update_weight_hashes.py`` to re-pin hashes in place)."""
    lines = [
        "# Copyright (c) 2026 Autovision contributors. Apache-2.0.",
        '"""Pinned pretrained-weight manifest (machine-managed).',
        "",
        "Regenerate with scripts/update_weight_hashes.py; hand-edits are",
        "overwritten on the next pinning run.",
        '"""',
        "",
        "from autovision._models.weight_verification import WeightArtifact",
        "",
        "__all__ = ['WEIGHT_MANIFEST']",
        "",
        "WEIGHT_MANIFEST = {",
    ]
    for repo_id in sorted(manifest):
        artifact = manifest[repo_id]
        lines.append(f"    {repo_id!r}: WeightArtifact(")
        lines.append(f"        repo_id={repo_id!r},")
        lines.append(f"        license={artifact.license!r},")
        lines.append(f"        source={artifact.source!r},")
        lines.append(f"        revision={artifact.revision!r},")
        lines.append(f"        sha256={artifact.sha256!r},")
        lines.append("    ),")
    lines.append("}")
    return "\n".join(lines) + "\n"
