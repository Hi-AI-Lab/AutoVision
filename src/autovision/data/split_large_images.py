# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Training-side slicing of large-image datasets into YOLO tiles (design D9).

Sliding windows (``crop`` x ``crop``, stride ``crop * (1 - gap)``, final
row/column aligned to the bottom/right edge so windows never leave the
image) cover every source image. Windows whose image-filled fraction falls
below ``min_window_rate`` are dropped; when all of them would be, the
highest-rate batch is kept so small images survive. Targets are assigned to
every window with ``IoF >= iof_thr`` (intersection over the target's own
area), so one target may enter several windows.

Geometry here is numpy/PIL only. ``tile_window_starts`` is the single window
enumerator, shared with the inference-side tiler
(``tasks.shared.tiling_utils.tile_image``); it lives in the data layer
because ``tasks -> data`` is the only layering-legal import direction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageOps

from autovision.errors import AutoVisionError
from autovision.geometry.masks import polygon_area

__all__ = ["split_large_image_dataset", "tile_window_starts"]

#: Smallest usable window: below this a "tile" has no meaningful context.
_MIN_CROP = 64

_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".pgm", ".png", ".ppm", ".tif", ".tiff", ".webp"}
)


def tile_window_starts(
    image_height: int, image_width: int, tile_size: int, overlap: float
) -> list[tuple[int, int]]:
    """Enumerate ``(x_start, y_start)`` offsets of tiles covering an image.

    The stride is ``int((1 - overlap) * tile_size)``; the final row/column is
    aligned to the bottom/right edge so tiles never extend past the image.
    Images not larger than the tile yield a single ``(0, 0)``. Shared with
    the inference-side ``tile_image`` — keep the two contracts in sync.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    step = max(1, int((1.0 - overlap) * tile_size))
    max_y = max(image_height - tile_size, 0)
    max_x = max(image_width - tile_size, 0)
    y_starts = list(range(0, max_y + 1, step)) or [0]
    x_starts = list(range(0, max_x + 1, step)) or [0]
    if y_starts[-1] != max_y:
        y_starts.append(max_y)
    if x_starts[-1] != max_x:
        x_starts.append(max_x)
    # Sorted scan order: the edge-aligned window appended above lands out
    # of sequence otherwise, and consumers (manifest order, tests) expect
    # the raster-scan layout.
    return [
        (x, y)
        for y in sorted(y_starts)
        for x in sorted(x_starts)
    ]


def split_large_image_dataset(
    images_dir: Path, labels_dir: Path | None, out_dir: Path, *,
    crop: int = 1024, gap: float = 0.2, iof_thr: float = 0.7,
    min_window_rate: float = 0.6,
    label_format: Literal["hbb", "obb"] = "hbb",
) -> Path:
    """Slice a large-image detection dataset into YOLO training tiles.

    Args:
        images_dir: Directory of source images (flat, iterated in sorted
            name order).
        labels_dir: YOLO label directory (one ``<stem>.txt`` per image) or
            ``None`` to slice images only, writing no labels.
        out_dir: Output root; receives ``images/``, ``labels/`` (only when
            ``labels_dir`` is given) and a ``tiles.jsonl`` manifest mapping
            every tile back to its source image and window.

    Keyword Args:
        crop: Square window size in pixels (>= 64).
        gap: Fraction of ``crop`` between consecutive window origins, i.e.
            neighbouring windows overlap by ``gap * crop`` ([0, 0.9)).
        iof_thr: Intersection-over-foreground threshold assigning a target
            to a window ([0.5, 1.0]); one target may enter several windows.
        min_window_rate: Minimum fraction of a window covered by real image
            pixels ((0, 1]); windows below it are dropped, and when all
            would be, the highest-rate batch is kept instead.
        label_format: ``"hbb"`` rows ``cls cx cy w h`` or ``"obb"`` rows
            ``cls x1 y1 x2 y2 x3 y3 x4 y4``, YOLO-normalized.

    Returns:
        The output directory ``out_dir``.

    Raises:
        AutoVisionError: On invalid parameters, missing input directories,
            unreadable images, or malformed label lines.

    Assigned targets keep their original geometry (shifted into the window
    frame and normalized by the window size); for OBB the polygon is only
    clipped to the window to *measure* truncation, never to rewrite it.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir is not None else None
    out_dir = Path(out_dir)
    _validate_parameters(crop, gap, iof_thr, min_window_rate, label_format)
    if not images_dir.is_dir():
        raise AutoVisionError(f"images_dir is not a directory: '{images_dir}'.")
    if labels_dir is not None and not labels_dir.is_dir():
        raise AutoVisionError(f"labels_dir is not a directory: '{labels_dir}'.")
    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not image_paths:
        raise AutoVisionError(f"no readable images found in '{images_dir}'.")

    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    if labels_dir is not None:
        labels_out.mkdir(parents=True, exist_ok=True)

    with (out_dir / "tiles.jsonl").open("w", encoding="utf-8") as manifest:
        for image_path in image_paths:
            records = _tile_one_image(
                image_path,
                labels_dir,
                images_out,
                labels_out,
                crop=crop,
                gap=gap,
                iof_thr=iof_thr,
                min_window_rate=min_window_rate,
                label_format=label_format,
            )
            for record in records:
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_dir


def _validate_parameters(
    crop: int,
    gap: float,
    iof_thr: float,
    min_window_rate: float,
    label_format: str,
) -> None:
    """Reject semantically invalid slicer parameters with clear messages."""
    if isinstance(crop, bool) or not isinstance(crop, (int, np.integer)):
        raise AutoVisionError(f"crop must be an integer >= {_MIN_CROP} pixels, got {crop!r}.")
    if int(crop) < _MIN_CROP:
        raise AutoVisionError(f"crop must be >= {_MIN_CROP} pixels, got {int(crop)}.")
    if isinstance(gap, bool) or not isinstance(
        gap, (float, int, np.floating, np.integer)
    ) or not 0.0 <= float(gap) < 0.9:
        raise AutoVisionError(f"gap must be a number in [0, 0.9), got {gap!r}.")
    if isinstance(iof_thr, bool) or not isinstance(
        iof_thr, (float, int, np.floating, np.integer)
    ) or not 0.5 <= float(iof_thr) <= 1.0:
        raise AutoVisionError(
            f"iof_thr must be a number in [0.5, 1.0], got {iof_thr!r}."
        )
    if isinstance(min_window_rate, bool) or not isinstance(
        min_window_rate, (float, int, np.floating, np.integer)
    ) or not 0.0 < float(min_window_rate) <= 1.0:
        raise AutoVisionError(
            f"min_window_rate must be a number in (0, 1], got {min_window_rate!r}."
        )
    if label_format not in ("hbb", "obb"):
        raise AutoVisionError(
            f"label_format must be 'hbb' or 'obb', got {label_format!r}."
        )


def _tile_one_image(
    image_path: Path,
    labels_dir: Path | None,
    images_out: Path,
    labels_out: Path,
    *,
    crop: int,
    gap: float,
    iof_thr: float,
    min_window_rate: float,
    label_format: str,
) -> list[dict[str, object]]:
    """Write tiles (and label files) for one source image; return manifest rows."""
    with Image.open(image_path) as image:
        # EXIF-normalize like open_image, so boxes and pixels share one frame.
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        windows = _kept_windows(height, width, crop, gap, min_window_rate)
        if labels_dir is None:
            classes: np.ndarray = np.zeros(0, dtype=np.int64)
            geometry: np.ndarray = np.zeros((0, 4), dtype=np.float64)
        else:
            classes, geometry = _load_targets(
                labels_dir / f"{image_path.stem}.txt", width, height, label_format
            )
        records: list[dict[str, object]] = []
        for window in windows:
            x, y, w, h = window
            tile_name = f"{image_path.stem}__{x}___{y}{image_path.suffix}"
            image.crop((x, y, x + w, y + h)).save(images_out / tile_name)
            if label_format == "hbb":
                lines = _hbb_label_lines(classes, geometry, window, iof_thr)
            else:
                lines = _obb_label_lines(classes, geometry, window, iof_thr)
            if labels_dir is not None:
                # Empty label files are meaningful: YOLO background tiles.
                (labels_out / f"{Path(tile_name).stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
            records.append(
                {
                    "tile": tile_name,
                    "source": image_path.name,
                    "window": [x, y, w, h],
                    "targets": len(lines),
                }
            )
    return records


def _kept_windows(
    height: int, width: int, crop: int, gap: float, min_window_rate: float
) -> list[tuple[int, int, int, int]]:
    """Return windows ``(x, y, w, h)`` that pass the image-fill-rate gate.

    The rate is the window's real-image area over ``crop ** 2``; windows are
    clipped to the image so the last row/column sits flush with the bottom /
    right edge. When every window falls below ``min_window_rate`` (small
    images), the highest-rate batch is kept so no image is lost.
    """
    area = float(crop * crop)
    windows = [
        (x, y, min(crop, width - x), min(crop, height - y))
        for x, y in tile_window_starts(height, width, crop, gap)
    ]
    rates = [(w * h) / area for _, _, w, h in windows]
    kept = [win for win, rate in zip(windows, rates) if rate >= min_window_rate]
    if kept:
        return kept
    best_rate = max(rates)
    return [win for win, rate in zip(windows, rates) if rate >= best_rate - 1e-9]


def _load_targets(
    label_path: Path, width: int, height: int, label_format: str
) -> tuple[np.ndarray, np.ndarray]:
    """Parse one YOLO label file into (class ids, pixel-space geometry).

    HBB rows become ``(N, 4)`` xyxy boxes; OBB rows become ``(N, 8)``
    flattened corner polygons. A missing file means an unlabeled image
    (zero targets), not an error.
    """
    if not label_path.is_file():
        fields = 4 if label_format == "hbb" else 8
        return np.zeros(0, dtype=np.int64), np.zeros((0, fields), dtype=np.float64)
    expected = 5 if label_format == "hbb" else 9
    classes: list[int] = []
    rows: list[list[float]] = []
    for line_no, raw in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) != expected:
            raise AutoVisionError(
                f"'{label_path}' line {line_no}: {label_format} labels need "
                f"{expected} fields, got {len(parts)}."
            )
        try:
            classes.append(int(parts[0]))
            rows.append([float(value) for value in parts[1:]])
        except ValueError as error:
            raise AutoVisionError(
                f"'{label_path}' line {line_no}: numeric parse failed ({error})."
            ) from error
    geometry = np.asarray(rows, dtype=np.float64).reshape(-1, expected - 1)
    geometry[:, 0::2] *= width
    geometry[:, 1::2] *= height
    if label_format == "hbb":
        cx, cy, bw, bh = geometry.T
        boxes = np.stack(
            (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0), axis=1
        )
        return np.asarray(classes, dtype=np.int64), boxes
    return np.asarray(classes, dtype=np.int64), geometry


def _hbb_label_lines(
    classes: np.ndarray,
    boxes_xyxy: np.ndarray,
    window: tuple[int, int, int, int],
    iof_thr: float,
) -> list[str]:
    """YOLO HBB rows for the targets whose IoF with the window passes."""
    if classes.shape[0] == 0:
        return []
    x, y, w, h = window
    keep = _box_iof(boxes_xyxy, window) >= iof_thr
    return [
        f"{int(cls)} "
        f"{((box[0] + box[2]) / 2.0 - x) / w:.6f} "
        f"{((box[1] + box[3]) / 2.0 - y) / h:.6f} "
        f"{(box[2] - box[0]) / w:.6f} "
        f"{(box[3] - box[1]) / h:.6f}"
        for cls, box in zip(classes[keep], boxes_xyxy[keep])
    ]


def _obb_label_lines(
    classes: np.ndarray,
    polygons: np.ndarray,
    window: tuple[int, int, int, int],
    iof_thr: float,
) -> list[str]:
    """YOLO OBB rows for targets whose window-truncation IoF passes.

    Kept targets output their ORIGINAL polygon (shifted, normalized); the
    Sutherland–Hodgman clip exists only to measure how much of the polygon
    the window actually contains. The per-polygon loop is inherent to the
    sequential clip and runs on tiny (<= 8 vertex) inputs offline.
    """
    if classes.shape[0] == 0:
        return []
    x, y, w, h = window
    origin = np.array([x, y], dtype=np.float64)
    extent = np.array([w, h], dtype=np.float64)
    lines: list[str] = []
    for cls, polygon in zip(classes, polygons):
        corners = polygon.reshape(-1, 2)
        # Clip in the window frame: translation is area-preserving, so the
        # clipped/original ratio is exactly polygon-inter-window / polygon.
        local = corners - origin
        clipped = _clip_polygon_to_canvas(local, (h, w))
        if clipped.shape[0] < 3:
            continue
        area = polygon_area(corners)
        if area <= 0.0 or polygon_area(clipped) / area < iof_thr:
            continue
        normalized = local / extent
        lines.append(
            f"{int(cls)} "
            + " ".join(f"{value:.6f}" for value in normalized.reshape(-1))
        )
    return lines


def _box_iof(
    boxes_xyxy: np.ndarray, window: tuple[int, int, int, int]
) -> np.ndarray:
    """Vectorized intersection-over-foreground of each box with the window."""
    x, y, w, h = window
    ix1 = np.maximum(boxes_xyxy[:, 0], float(x))
    iy1 = np.maximum(boxes_xyxy[:, 1], float(y))
    ix2 = np.minimum(boxes_xyxy[:, 2], float(x + w))
    iy2 = np.minimum(boxes_xyxy[:, 3], float(y + h))
    intersection = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)
    areas = (
        np.clip(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 0.0, None)
        * np.clip(boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 0.0, None)
    )
    return intersection / np.maximum(areas, 1e-12)


def _clip_polygon_to_canvas(
    polygon: np.ndarray, canvas: tuple[int, int]
) -> np.ndarray:
    """Clip an ``(N, 2)`` polygon to ``[0, w] x [0, h]`` (Sutherland–Hodgman).

    Returns the clipped polygon in traversal order; fewer than 3 points
    means the polygon is (effectively) outside the canvas.
    """
    height, width = canvas
    points = np.asarray(polygon, dtype=np.float64)
    for axis, limit, keep_below in (
        (0, float(width), True),  # x <= w
        (1, float(height), True),  # y <= h
        (0, 0.0, False),  # x >= 0
        (1, 0.0, False),  # y >= 0
    ):
        points = _clip_half_plane(points, axis, limit, keep_below)
        if points.shape[0] < 3:
            break
    return points


def _clip_half_plane(
    points: np.ndarray, axis: int, limit: float, keep_below: bool
) -> np.ndarray:
    """One Sutherland–Hodgman pass against ``coord <= limit`` (or ``>=``).

    Vertices are emitted in edge-traversal order — reordering them would
    make the shoelace area of the result wrong.
    """
    if points.shape[0] == 0:
        return points
    values = points[:, axis]
    inside = values <= limit if keep_below else values >= limit
    nxt = np.roll(points, -1, axis=0)
    nxt_inside = np.roll(inside, -1)
    pieces: list[np.ndarray] = []
    for index in range(points.shape[0]):
        current, following = points[index], nxt[index]
        if inside[index]:
            pieces.append(current[None])
        if bool(inside[index]) != bool(nxt_inside[index]):
            delta = following[axis] - current[axis]
            t = 0.0 if delta == 0.0 else (limit - current[axis]) / delta
            pieces.append((current + t * (following - current))[None])
    if not pieces:
        return np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(pieces, axis=0)
