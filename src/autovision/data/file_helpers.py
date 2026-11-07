# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Data helpers: image filename discovery and decoding."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torchvision.io import ImageReadMode, read_image

from autovision.errors import AutoVisionError

__all__ = [
    "build_relative_lookup",
    "list_image_filenames_from_dir",
    "load_annotation_json",
    "match_with_stem_fallback",
    "open_image",
    "open_mask",
    "required_field",
    "resolve_under",
]

logger = logging.getLogger("autovision")

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}


def list_image_filenames_from_dir(image_dir: Path) -> Iterator[Path]:
    """Recursively list image filenames under ``image_dir`` (sorted, stable)."""
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: '{image_dir}'")
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield path


def open_image(filename: Path | str, num_channels: int = 3) -> torch.Tensor:
    """Decode an image to a uint8 (C, H, W) tensor honoring ``num_channels``.

    Fast path: ``torchvision.io.read_image``. Fallback: PIL for exotic
    formats; EXIF orientation is normalized so boxes and pixels agree, and
    truncated files raise instead of silently training on partial data.

    Channel contract (DATA-09): the returned channel count always matches
    ``num_channels`` regardless of the source file — 3 converts to RGB, 1
    converts to grayscale ("L", PIL ITU-R 601-2 luma). The torchvision fast
    path decodes ``UNCHANGED`` for 1-channel requests, so a source-native
    channel count that slips through it (an RGB JPEG or an RGBA PNG) is
    re-decoded via the PIL path; the conversion rule no longer depends on
    the file format or the decoder that happened to win.
    """
    if num_channels not in (1, 3):
        raise AutoVisionError(
            f"num_channels must be 1 or 3, got {num_channels!r}."
        )
    mode = ImageReadMode.RGB if num_channels == 3 else ImageReadMode.UNCHANGED
    # torchvision's fast decoder does not normalize EXIF rotation. Route
    # oriented JPEGs through PIL so boxes and pixels share one frame.
    if Path(filename).suffix.lower() in {".jpg", ".jpeg"}:
        try:
            from PIL import Image

            with Image.open(filename) as image:
                if image.getexif().get(274, 1) not in (None, 1):
                    return _open_image_pil(filename, num_channels)
        except Exception:  # noqa: BLE001 - the normal decoder reports it
            pass
    try:
        tensor = read_image(str(filename), mode=mode)
    except RuntimeError:
        return _open_image_pil(filename, num_channels)
    if tensor.shape[0] != num_channels:
        # The UNCHANGED fast path returned the source's own channel count
        # (RGB source for a 1-channel request) — re-decode via PIL so the
        # conversion is identical for every file format (DATA-09).
        return _open_image_pil(filename, num_channels)
    return tensor


#: One process-level warning for the third-party-flipped truncation flag;
#: the check runs per decode, so an undeduplicated warning would spam once
#: per image on large datasets (review P3-4).
_TRUNCATED_FLAG_WARNED = False


def _open_image_pil(filename: Path | str, num_channels: int) -> torch.Tensor:
    from PIL import Image, ImageFile, ImageOps
    from torchvision.transforms import functional as TF

    # Truncated images must fail loudly: silently decoding a partial image
    # trains on pixels that no longer match its annotations. The PIL flag
    # is process-global (H1-10), so this path never writes it — mask
    # decoding widens it inside a scoped context manager instead. If some
    # third-party import left it enabled, say so once and keep going.
    global _TRUNCATED_FLAG_WARNED
    if ImageFile.LOAD_TRUNCATED_IMAGES and not _TRUNCATED_FLAG_WARNED:
        _TRUNCATED_FLAG_WARNED = True
        logger.warning(
            "PIL ImageFile.LOAD_TRUNCATED_IMAGES is True while opening image "
            "'%s'; truncated images will decode instead of failing loudly "
            "(the flag was likely flipped by a third-party import).",
            filename,
        )
    try:
        with Image.open(filename) as image:
            image = ImageOps.exif_transpose(image)
            if num_channels == 3:
                image = image.convert("RGB")
            elif num_channels == 1:
                image = image.convert("L")
            return TF.pil_to_tensor(image)
    except Exception as error:
        raise RuntimeError(
            f"Failed to open image '{filename}': {error}"
        ) from error


def open_mask(filename: Path | str) -> torch.Tensor:
    """Decode a segmentation mask to an integer (H, W) tensor.

    Masks store region ids (e.g. panoptic ids), so the pixel values must be
    preserved exactly: read without color conversion and never scale.
    Single-channel masks are squeezed from (1, H, W) to (H, W).
    """
    try:
        mask = read_image(str(filename), mode=ImageReadMode.UNCHANGED)
    except RuntimeError:
        mask = _open_mask_pil(filename)
    if mask.ndim == 2:
        return mask
    if mask.shape[0] == 1:
        return mask[0]
    raise ValueError(
        f"Mask '{filename}' has {mask.shape[0]} channels; single-channel "
        "masks (grayscale/palette PNG) are required."
    )


@contextmanager
def _truncated_image_loading() -> Iterator[None]:
    """Scope ``PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True`` to one decode.

    The flag is process-global, and DataLoader workers decode images and
    masks interchangeably: leaving it enabled after a mask read would make
    every later truncated JPEG decode silently (H1-10). Saving/restoring
    around the mask decode keeps the lenient mask semantics for historical
    datasets without leaking into image decoding.
    """
    from PIL import ImageFile

    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        yield
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def _open_mask_pil(filename: Path | str) -> torch.Tensor:
    from PIL import Image
    from torchvision.transforms import functional as TF

    with _truncated_image_loading():
        try:
            with Image.open(filename) as image:
                return TF.pil_to_tensor(image)
        except Exception as error:
            raise RuntimeError(
                f"Failed to open mask '{filename}': {error}"
            ) from error


def resolve_under(root: Path | str, file_name: str) -> Path:
    """Resolve a dataset-relative ``file_name`` under ``root``, refusing escapes.

    Annotation formats (COCO ``file_name``, ...) name files relative to a
    dataset root, and hand-edited or hostile annotations may carry ``..``
    traversal or absolute paths. Every loader goes through this one guard
    instead of re-implementing containment checks (H3-2d).

    Parameters:
        root: Dataset root directory; may itself be relative.
        file_name: Annotation-declared file name, relative to ``root``.

    Returns:
        The resolved absolute path, guaranteed inside ``root``.

    Raises:
        AutoVisionError: When the resolved path escapes ``root`` (message
            carries both the declared name and the resolved path).
    """
    root_path = Path(root).resolve()
    candidate = (root_path / file_name).resolve()
    if not candidate.is_relative_to(root_path):
        raise AutoVisionError(
            f"Dataset file name {file_name!r} escapes the dataset root "
            f"'{root_path}': resolved to '{candidate}'."
        )
    return candidate


def load_annotation_json(path: Path | str, *, kind: str = "annotations") -> dict[str, Any]:
    """Load an annotation json (COCO, panoptic segments, ...), failing loudly.

    A truncated or 0-byte annotation file used to surface as a bare
    ``json.JSONDecodeError`` (or ``OSError``) with no file name in the
    message; every loader goes through this one wrapper instead so the
    error always names the file (H3-2e, same style as ``versioning.py``).

    Parameters:
        path: Annotation file path.
        kind: Human-readable kind for the error message (e.g.
            ``"COCO annotations"``, ``"panoptic segments"``).

    Returns:
        The parsed JSON object.

    Raises:
        AutoVisionError: When the file is unreadable, not valid JSON, or
            does not hold a JSON object.
    """
    annotation_path = Path(path)
    try:
        text = annotation_path.read_text()
    except OSError as error:
        raise AutoVisionError(
            f"Could not read {kind} '{annotation_path}': {error}."
        ) from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise AutoVisionError(
            f"{kind} '{annotation_path}' is not valid JSON: {error}."
        ) from error
    if not isinstance(payload, dict):
        raise AutoVisionError(f"{kind} '{annotation_path}' must hold a JSON object.")
    return payload


def required_field(
    mapping: Mapping[str, Any],
    key: str,
    *,
    source: Path | str,
    entry: str,
) -> Any:
    """Fetch a required annotation field, naming the key and location.

    A bare ``KeyError`` from a hand-edited or truncated annotation file
    carries no file context; every loader goes through this guard so the
    error names the annotation file, the entry kind and the missing key
    (H3-2e).

    Parameters:
        mapping: One annotation entry (e.g. a COCO ``images`` row).
        key: Required field name (e.g. ``"width"``).
        source: Annotation file path, for the error message.
        entry: Short entry description, e.g. ``"images entry"``.

    Raises:
        AutoVisionError: When the key is missing (or the entry is not a
            mapping at all).
    """
    try:
        return mapping[key]
    except (KeyError, TypeError) as error:
        raise AutoVisionError(
            f"Annotation file '{source}': {entry} is missing the required "
            f"key '{key}'."
        ) from error


def build_relative_lookup(
    paths: Iterable[Path],
    root: Path,
    *,
    prefer: Callable[[Path, Path], Path] | None = None,
) -> tuple[dict[str, Path], dict[str, Path], set[str]]:
    """Index ``paths`` under ``root`` for pair matching (H3-2i).

    The primary key is the suffix-stripped path RELATIVE to ``root`` so a
    nested tree (``sub/a.jpg`` vs ``other/a.jpg``) can never cross-match
    by bare stem. The bare-stem table is kept as a fallback for layouts
    where the two trees are structured differently, together with the set
    of stems that appear in MORE THAN ONE directory — a stem-fallback hit
    on one of those is exactly the old silent cross-match bug.

    Parameters:
        paths: Files to index (pre-filtered by the caller).
        root: Directory the relative keys are taken against.
        prefer: Optional tie-breaker for two files with the same key
            (e.g. depth maps prefer ``.npy`` over ``.png``); it receives
            ``(candidate, current)`` and returns the winner.

    Returns:
        ``(by_relative, by_stem, ambiguous_stems)``.
    """
    by_relative: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    stem_directories: dict[str, set[Path]] = {}
    for path in paths:
        key = path.relative_to(root).with_suffix("").as_posix()
        current = by_relative.get(key)
        if current is None or prefer is None or prefer(path, current) is path:
            by_relative[key] = path
        chosen = by_stem.get(path.stem)
        if chosen is None or (prefer is not None and prefer(path, chosen) is path):
            by_stem[path.stem] = path
        stem_directories.setdefault(path.stem, set()).add(path.parent)
    ambiguous = {
        stem for stem, directories in stem_directories.items() if len(directories) > 1
    }
    return by_relative, by_stem, ambiguous


def match_with_stem_fallback(
    image_path: Path,
    images_root: Path | None,
    by_relative: dict[str, Path],
    by_stem: dict[str, Path],
) -> tuple[Path | None, bool]:
    """Look up the counterpart file for one image (H3-2i).

    The suffix-stripped relative path is the primary key; the bare stem is
    only consulted when the relative key misses (differently-shaped
    trees). Callers warn when that fallback resolves to one of the
    ambiguous stems returned by :func:`build_relative_lookup`.

    Returns:
        ``(match, used_stem_fallback)`` — ``match`` is ``None`` when no
        counterpart exists at all.
    """
    key: str | None = None
    if images_root is not None:
        try:
            key = image_path.relative_to(images_root).with_suffix("").as_posix()
        except ValueError:
            key = None
    if key is not None:
        match = by_relative.get(key)
        if match is not None:
            return match, False
    return by_stem.get(image_path.stem), True
