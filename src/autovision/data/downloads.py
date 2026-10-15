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
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from autovision.errors import AutoVisionError

__all__ = ["DATASET_REGISTRY", "DatasetFile", "DatasetSpec", "download_dataset"]


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
            DatasetFile(url="http://images.cocodataset.org/zips/val2017.zip"),
            DatasetFile(
                url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            ),
        ),
        license_note="Images: Flickr terms of use; annotations: CC BY 4.0.",
    ),
    "coco2017-train": DatasetSpec(
        description="COCO 2017 train images (118k, ~18GB) + annotations.",
        homepage="https://cocodataset.org",
        files=(
            DatasetFile(url="http://images.cocodataset.org/zips/train2017.zip"),
            DatasetFile(
                url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
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


def _download_file(spec: DatasetFile, dest: Path) -> Path:
    import logging

    logger = logging.getLogger("autovision")
    if spec.sha256 is None:
        logger.warning(
            "Dataset artifact %s has no pinned sha256 in the registry; "
            "integrity is not verified for this download.",
            dest.name,
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(spec.url) as response, dest.open("wb") as handle:
        total = response.getheader("Content-Length")
        total = int(total) if total else None
        try:
            from tqdm import tqdm

            progress = tqdm(
                total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
            )
        except ImportError:  # pragma: no cover - tqdm is a core dependency
            progress = None
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            if progress is not None:
                progress.update(len(chunk))
        if progress is not None:
            progress.close()
    if spec.size_bytes is not None and dest.stat().st_size != spec.size_bytes:
        raise AutoVisionError(
            f"Downloaded '{dest.name}' has {dest.stat().st_size} bytes, "
            f"expected {spec.size_bytes}."
        )
    if spec.sha256 is not None and _digest(dest) != spec.sha256:
        dest.unlink(missing_ok=True)
        raise AutoVisionError(
            f"sha256 mismatch for '{dest.name}'; download deleted."
        )
    return dest


def _extract(archive: Path, out_dir: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(out_dir)
        return
    with tarfile.open(archive) as bundle:
        bundle.extractall(out_dir, filter="data")


def download_dataset(name: str, out_dir: Path | str) -> Path:
    """Download + extract a registered dataset under ``out_dir/<name>``.

    Already-extracted archives are skipped, so the call is idempotent and
    resumable at file granularity.
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
    root = Path(out_dir) / name
    root.mkdir(parents=True, exist_ok=True)
    for entry in spec.files:
        archive_name = Path(entry.url).name
        archive = root / archive_name
        if not archive.exists():
            _download_file(entry, archive)
        _extract(archive, root)
    return root
