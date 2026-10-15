# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""GTR teacher assets: local MIT checkpoints (roadmap design doc 31, G0).

The Intellindust GTR detection checkpoints ({model, ema, last_epoch}
payloads) are registered here with pinned sha256 digests so teacher
consumption is verifiable end to end. Checkpoints are MIT-licensed
(HuggingFace `Phoenix8125/GTR` front-matter); the files stay at their
research-repo location — this registry pins identity, not distribution.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from autovision.errors import AutoVisionError

__all__ = ["GTRAsset", "GTR_ASSET_ROOT", "GTR_ASSETS", "resolve_gtr_asset"]

GTR_ASSET_ROOT = Path.home() / "Research/Project/Dino-Dome-Research/weight/gtr/det"
_GTR_HF = "https://huggingface.co/Phoenix8125/GTR"


@dataclass(frozen=True)
class GTRAsset:
    name: str
    path: Path
    sha256: str
    license: str = "MIT"
    format: str = "gtr_ckpt_v1"  # {model, ema, last_epoch}; EMA is the
    # benchmark口径 (GTR README: all numbers measured with EMA weights)
    num_classes: int = 80
    source: str = _GTR_HF
    coco_ap: float | None = None


GTR_ASSETS: dict[str, GTRAsset] = {
    "gtr_s_coco": GTRAsset(
        name="gtr_s_coco",
        path=GTR_ASSET_ROOT / "gtr_s_coco.pth",
        sha256="69b8bed4d2bdddd8c965170195a6e40e6b848d13c9bb72f73776dfa0f5dd1979",
        coco_ap=53.6,
    ),
    "gtr_m_coco": GTRAsset(
        name="gtr_m_coco",
        path=GTR_ASSET_ROOT / "gtr_m_coco.pth",
        sha256="a1a90111bb75286bbda7f30ff87a99cc156ca9b75a779ec8f995156cd926a6dd",
        coco_ap=57.3,
    ),
    "gtr_l_coco": GTRAsset(
        name="gtr_l_coco",
        path=GTR_ASSET_ROOT / "gtr_l_coco.pth",
        sha256="d7a3e07b2159a6e4953c038aa2a1f6ccfe67a2e3d7564862a230acba7276d298",
        coco_ap=58.9,
    ),
    "gtr_x_coco": GTRAsset(
        name="gtr_x_coco",
        path=GTR_ASSET_ROOT / "gtr_x_coco.pth",
        sha256="547cc2c2d29b32315fa9d533431e35ed0b17807e82f0803779a91f268ff878fb",
        coco_ap=59.4,
    ),
}


class GTRAssetError(AutoVisionError):
    pass


def verify_gtr_asset(asset: GTRAsset, chunk_size: int = 1 << 22) -> None:
    """Fail unless the checkpoint exists and matches its pinned sha256."""
    if not asset.path.exists():
        raise GTRAssetError(
            f"GTR asset '{asset.name}' not found at '{asset.path}'. "
            f"Source: {asset.source} (license {asset.license})."
        )
    digest = hashlib.sha256()
    with asset.path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    if digest.hexdigest() != asset.sha256:
        raise GTRAssetError(
            f"GTR asset '{asset.name}' sha256 mismatch — refusing to use "
            f"unverified teacher weights (pinned {asset.sha256[:12]}…, "
            f"got {digest.hexdigest()[:12]}…)."
        )


def resolve_gtr_asset(name: str, verify: bool = True) -> GTRAsset:
    """Return the registered asset, optionally verifying its sha256."""
    asset = GTR_ASSETS.get(name)
    if asset is None:
        supported = ", ".join(sorted(GTR_ASSETS))
        raise GTRAssetError(f"Unknown GTR asset '{name}'. Registered: {supported}.")
    if verify:
        verify_gtr_asset(asset)
    return asset
