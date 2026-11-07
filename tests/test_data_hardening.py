# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Data-layer hardening regressions (H1-9, H1-10, H3-2a..d).

Each test references its hardening-plan id so the intent stays anchored
(hardening plan §3.5 / H4-3):

- H1-10: ``PIL.ImageFile.LOAD_TRUNCATED_IMAGES`` is process-global; mask
  decoding must scope its lenient flip instead of leaking it into image
  decoding (a leaked flag silently decodes truncated JPEGs into training).
- H3-2a: YOLO/OBB loaders count malformed label lines and warn once.
- H3-2b: COCO detection loader fails loudly on duplicate image ids and
  warns (with a count) on annotations with undeclared category ids.
- H3-2c: COCO instance segmentation uses pre-grouped annotations with
  output identical to the naive images × annotations scan.
- H3-2d: ``resolve_under`` guards file_name traversal in every COCO loader.
- H1-9: every direct-download registry artifact must be pinned (or sit in
  the explicit unpinned allowlist until its pin is computed).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.conftest import create_image

from autovision.data.coco_instance_segmentation_dataset import (
    COCOInstanceSegmentationDataset,
)
from autovision.data.coco_object_detection_dataset import (
    COCOObjectDetectionDataArgs,
    SplitArgs,
)
from autovision.data.downloads import DATASET_REGISTRY
from autovision.data.file_helpers import (
    _open_image_pil,
    _open_mask_pil,
    _truncated_image_loading,
    open_image,
    resolve_under,
)
from autovision.data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from autovision.data.yolo_oriented_object_detection_dataset import (
    YOLOOrientedObjectDetectionDataArgs,
)
from autovision.errors import AutoVisionError

# ── H1-10: truncated-image global flag isolation ───────────────────────────


def _write_truncated_jpeg(path: Path, keep_fraction: float = 0.7) -> None:
    """Write a JPEG cut mid-entropy-scan (header complete, tail missing)."""
    from PIL import Image

    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=95)
    data = buffer.getvalue()
    path.write_bytes(data[: max(1, int(len(data) * keep_fraction))])


def test_mask_read_scopes_the_truncated_flag_and_image_still_fails_loudly(
    tmp_path: Path,
) -> None:
    """H1-10: a lenient mask read must not enable truncated *image* decoding.

    Order is the point: the mask read used to flip the process-global PIL
    flag to True and never restore it, so the truncated JPEG below would
    silently decode partial pixels on the very next image read.
    """
    from PIL import ImageFile

    truncated = tmp_path / "truncated.jpg"
    _write_truncated_jpeg(truncated)

    # The lenient mask semantics stay: a truncated mask still decodes.
    mask = _open_mask_pil(truncated)
    assert mask.ndim >= 2
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False  # restored, not leaked

    with pytest.raises(RuntimeError, match="Failed to open image"):
        _open_image_pil(truncated, num_channels=3)
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False


def test_truncated_image_loading_context_restores_previous_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H1-10: the scoped context manager saves/restores the global flag."""
    from PIL import ImageFile

    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", False)
    with _truncated_image_loading():
        assert ImageFile.LOAD_TRUNCATED_IMAGES is True
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False

    # A third-party True is preserved, not clobbered back to False.
    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)
    with _truncated_image_loading():
        assert ImageFile.LOAD_TRUNCATED_IMAGES is True
    assert ImageFile.LOAD_TRUNCATED_IMAGES is True


def test_image_pil_warns_without_blocking_when_global_was_flipped(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H1-10: image decoding warns once if a third-party enabled the flag,
    still decodes, and never writes or resets the global itself."""
    from PIL import Image, ImageFile

    image_path = tmp_path / "ok.jpg"
    Image.new("RGB", (8, 8)).save(image_path, format="JPEG")
    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)

    with caplog.at_level(logging.WARNING, logger="autovision"):
        tensor = _open_image_pil(image_path, num_channels=3)
    assert tensor.shape[0] == 3
    assert "LOAD_TRUNCATED_IMAGES" in caplog.text
    assert ImageFile.LOAD_TRUNCATED_IMAGES is True  # untouched, not reset


# ── H3-2a: YOLO/OBB malformed label lines are counted, not silent ─────────


def _yolo_args(
    tmp_path: Path, label_text: str, obb: bool = False
) -> tuple[YOLOObjectDetectionDataArgs, Path]:
    """One-image YOLO split with the given label text (image never decoded)."""
    images_dir = tmp_path / "train" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "img.jpg").write_bytes(b"\xff\xd8stub")
    label_path = tmp_path / "train" / "labels" / "img.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(label_text)
    args_cls = YOLOOrientedObjectDetectionDataArgs if obb else YOLOObjectDetectionDataArgs
    args = args_cls(
        path=tmp_path,
        train=tmp_path / "train" / "images",
        val=tmp_path / "val" / "images",
        names={0: "a", 1: "b"},
    )
    return args, label_path


def test_yolo_malformed_lines_are_counted_and_warned_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """H3-2a: bad field counts are skipped but reported with file, count,
    and the first offending line number."""
    args, label_path = _yolo_args(
        tmp_path,
        "0 0.5 0.5 0.2 0.2\nnot enough fields\n1 0.1 0.1 0.3 0.3\ntwo words\n",
    )
    with caplog.at_level(logging.WARNING, logger="autovision"):
        rows = list(args.list_image_info("train"))

    assert len(rows) == 1
    assert json.loads(rows[0]["bboxes"]) == [[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.3, 0.3]]
    warnings = [r for r in caplog.records if "malformed" in r.getMessage()]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "2 malformed" in message
    assert str(label_path) in message
    assert "line 2" in message
    assert "deprecated in 0.6.1" in message


def test_yolo_obb_malformed_lines_are_counted_and_warned_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """H3-2a: the OBB loader (9 fields) counts a cxcywh-shaped line too."""
    args, label_path = _yolo_args(
        tmp_path,
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n0 0.5 0.5 0.2 0.2\n",
        obb=True,
    )
    with caplog.at_level(logging.WARNING, logger="autovision"):
        rows = list(args.list_image_info("train"))

    assert len(rows) == 1
    assert json.loads(rows[0]["bboxes"]) == [[0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2]]
    warnings = [r for r in caplog.records if "malformed" in r.getMessage()]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "1 malformed" in message
    assert str(label_path) in message
    assert "line 2" in message


@pytest.mark.parametrize(
    "obb", [pytest.param(False, id="det"), pytest.param(True, id="obb")]
)
def test_yolo_wellformed_labels_emit_no_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, obb: bool
) -> None:
    """H3-2a: clean labels (including blank lines) stay silent."""
    text = "0 0.5 0.5 0.2 0.2\n\n1 0.1 0.1 0.3 0.3\n"
    if obb:
        text = "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n\n"
    args, _ = _yolo_args(tmp_path, text, obb=obb)
    with caplog.at_level(logging.WARNING, logger="autovision"):
        rows = list(args.list_image_info("train"))
    assert len(rows) == 1
    assert not [r for r in caplog.records if "malformed" in r.getMessage()]


# ── H3-2b: duplicate image ids fail loudly; undeclared categories warn ────


def _coco_args(tmp_path: Path, payload: dict[str, Any]) -> COCOObjectDetectionDataArgs:
    """Write the payload and wrap it in detection data args (val = train)."""
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(payload))
    return COCOObjectDetectionDataArgs(
        train=SplitArgs(annotations=annotations_path, images=tmp_path / "images"),
        val=SplitArgs(annotations=annotations_path, images=tmp_path / "images"),
    )


def test_coco_duplicate_image_id_raises_with_both_file_names(tmp_path: Path) -> None:
    """H3-2b: a duplicate image id used to silently overwrite one row; it
    must now fail construction naming the id and both file names."""
    payload = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 10, "height": 10},
            {"id": 1, "file_name": "b.jpg", "width": 10, "height": 10},
        ],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [],
    }
    args = _coco_args(tmp_path, payload)
    with pytest.raises(
        AutoVisionError, match=r"image id 1 twice: 'a\.jpg' and 'b\.jpg'"
    ):
        list(args.list_image_info("train"))


def test_coco_undeclared_category_id_warns_and_drops(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """H3-2b: annotations with an undeclared category_id are dropped with a
    counted warning (merge_coco style), while ignore_classes stays silent."""
    payload = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": 1, "category_id": 99, "bbox": [30, 30, 10, 10]},
        ],
    }
    args = _coco_args(tmp_path, payload)
    with caplog.at_level(logging.WARNING, logger="autovision"):
        rows = list(args.list_image_info("train"))
    assert len(rows) == 1
    assert json.loads(rows[0]["class_labels"]) == [0]  # only the declared one
    assert len(json.loads(rows[0]["bboxes"])) == 1
    warnings = [r for r in caplog.records if "category_id" in r.getMessage()]
    assert len(warnings) == 1
    assert "1 annotation(s)" in warnings[0].getMessage()
    assert str(tmp_path / "annotations.json") in warnings[0].getMessage()

    # A deliberate ignore_classes drop must NOT trip the undeclared warning:
    # this payload only declares-and-ignores, it has no undeclared ids.
    ignored_payload = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
        ],
    }
    ignored_args = _coco_args(tmp_path, ignored_payload)
    ignored_args.ignore_classes = {1}
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="autovision"):
        ignored_rows = list(ignored_args.list_image_info("train"))
    assert json.loads(ignored_rows[0]["bboxes"]) == []
    assert not [r for r in caplog.records if "category_id" in r.getMessage()]


# ── H3-2c: instance segmentation pre-grouping stays output-identical ──────


def _interleaved_seg_payload() -> dict[str, Any]:
    """Small COCO payload with annotations interleaved across images, an
    undeclared category, a malformed polygon, and an unusable instance."""
    polygon = [1.0, 1.0, 20.0, 1.0, 20.0, 20.0, 1.0, 20.0]
    return {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 32, "height": 32},
            {"id": 2, "file_name": "b.png", "width": 32, "height": 32},
            {"id": 3, "file_name": "c.png", "width": 32, "height": 32},
        ],
        "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
        "annotations": [
            {"id": 1, "image_id": 2, "category_id": 1, "bbox": [1, 1, 19, 19],
             "segmentation": [polygon]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [2, 2, 15, 15],
             "segmentation": [polygon]},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [5, 5, 10, 10],
             "segmentation": [polygon]},
            {"id": 4, "image_id": 2, "category_id": 2, "bbox": [8, 8, 12, 12],
             "segmentation": [polygon]},
            {"id": 5, "image_id": 1, "category_id": 99, "bbox": [0, 0, 5, 5],
             "segmentation": [polygon]},
            {"id": 6, "image_id": 3, "category_id": 1, "bbox": [0, 0, 6, 6],
             "segmentation": [[1.0, 2.0, 3.0]]},
            {"id": 7, "image_id": 3, "category_id": 1, "bbox": None,
             "segmentation": []},
        ],
    }


def _reference_instance_rows(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """The pre-H3-2c O(images × annotations) scan, verbatim semantics.

    Grouping (which annotations land on which image, in which order) is
    what changed; mask decoding is shared with the loader via the same
    ``_decode_mask`` static method.
    """
    categories = sorted(int(c["id"]) for c in payload.get("categories", []))
    user_to_internal = {c: i for i, c in enumerate(categories)}
    rows: list[dict[str, Any]] = []
    for image in payload.get("images", []):
        image_id = int(image["id"])
        width, height = int(image["width"]), int(image["height"])
        boxes: list[list[float]] = []
        classes: list[int] = []
        masks: list[list[list[float]]] = []
        for annotation in payload.get("annotations", []):
            if int(annotation["image_id"]) != image_id:
                continue
            if int(annotation["category_id"]) not in user_to_internal:
                continue
            mask, _fell_back = COCOInstanceSegmentationDataset._decode_mask(
                annotation, height, width
            )
            if mask is None:
                continue
            boxes.append([float(v) for v in annotation["bbox"]])
            classes.append(user_to_internal[int(annotation["category_id"])])
            masks.append(mask.tolist())
        rows.append({"bboxes": boxes, "classes": classes, "masks": masks})
    return rows


def test_instance_segmentation_rows_match_reference_grouping(tmp_path: Path) -> None:
    """H3-2c: pre-grouped construction must match the naive double scan
    field for field (bboxes, classes, masks, per-image annotation order)."""
    payload = _interleaved_seg_payload()
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(payload))

    dataset = COCOInstanceSegmentationDataset(annotations_path, images_dir=tmp_path)
    expected = _reference_instance_rows(payload)

    assert len(dataset._rows) == len(expected)
    file_names = [image["file_name"] for image in payload["images"]]
    for row, want, file_name in zip(dataset._rows, expected, file_names):
        assert row["bboxes"] == want["bboxes"]
        assert row["classes"] == want["classes"]
        # Rows carry compact RLE since DATA-04; decoding must reproduce the
        # eager-decode reference masks bit for bit.
        assert [_decode_row_rle(rle) for rle in row["rles"]] == want["masks"]
        assert row["size"] == (32, 32)
        # image_path stays anchored to the same file as the pre-fix join.
        assert Path(row["image_path"]).resolve() == (tmp_path / file_name).resolve()
    # The bbox-fallback instance survived; the geometry-less one did not.
    assert [len(row["bboxes"]) for row in dataset._rows] == [2, 2, 1]


def _decode_row_rle(rle: dict[str, Any]) -> list[list[bool]]:
    """Decode a row RLE to nested bool lists (the eager-decode oracle form)."""
    from pycocotools import mask as mask_utils

    return mask_utils.decode(rle).astype(bool).tolist()


def test_instance_segmentation_rows_stay_rle_not_decoded_arrays(tmp_path: Path) -> None:
    """DATA-04: rows must hold compact RLE dicts, never (H, W) arrays.

    The eager variant materialized every instance mask at construction —
    hundreds of GB at COCO train2017 scale. A regression back to decoded
    arrays (or a mixed representation) fails here.
    """
    import numpy as np

    payload = _interleaved_seg_payload()
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(payload))
    dataset = COCOInstanceSegmentationDataset(annotations_path, images_dir=tmp_path)
    for row in dataset._rows:
        for rle in row["rles"]:
            assert isinstance(rle, dict) and "counts" in rle
            assert not isinstance(rle, np.ndarray)
        assert "masks" not in row  # the old eager key must not resurface


# ── H3-2d: resolve_under traversal guard across the COCO loaders ──────────


class TestResolveUnder:
    """The shared containment guard used by every COCO loader (H3-2d)."""

    def test_rejects_traversal_and_absolute_paths(self, tmp_path: Path) -> None:
        """``..`` components and absolute file names must fail loudly."""
        root = tmp_path / "root"
        root.mkdir()
        with pytest.raises(AutoVisionError, match=r"'\.\./evil\.png' escapes"):
            resolve_under(root, "../evil.png")
        with pytest.raises(AutoVisionError, match="escapes the dataset root"):
            resolve_under(root, "/etc/passwd")
        with pytest.raises(AutoVisionError, match="escapes the dataset root"):
            resolve_under(root, "sub/../../outside.png")

    def test_legal_relative_paths_resolve_inside_root(self, tmp_path: Path) -> None:
        """Plain relative names resolve to ``root / name`` unchanged."""
        root = tmp_path / "root"
        resolved = resolve_under(root, "sub/ok.png")
        assert resolved == (root / "sub" / "ok.png").resolve()


def test_coco_detection_rejects_traversal_file_name(tmp_path: Path) -> None:
    """H3-2d: the detection loader keeps its path-traversal guard."""
    payload = {
        "images": [{"id": 1, "file_name": "../evil.png", "width": 8, "height": 8}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [],
    }
    args = _coco_args(tmp_path, payload)
    with pytest.raises(AutoVisionError, match="escapes the dataset root"):
        list(args.list_image_info("train"))


def test_coco_instance_segmentation_rejects_traversal_file_name(
    tmp_path: Path,
) -> None:
    """H3-2d: the instance loader gains the guard it was missing."""
    payload = {
        "images": [{"id": 1, "file_name": "../evil.png", "width": 8, "height": 8}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [],
    }
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(payload))
    with pytest.raises(AutoVisionError, match="escapes the dataset root"):
        COCOInstanceSegmentationDataset(annotations_path, images_dir=tmp_path / "images")


def _keypoint_payload(file_name: str) -> dict[str, Any]:
    """Minimal person_keypoints payload with one fully-labelled person."""
    keypoints: list[float] = []
    for index in range(17):
        keypoints += [5.0 + index, 10.0 + index, 2]
    return {
        "images": [{"id": 1, "file_name": file_name, "width": 32, "height": 32}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [5.0, 10.0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0,
                "num_keypoints": 17,
                "keypoints": keypoints,
            }
        ],
    }


def test_coco_keypoint_rejects_traversal_file_name(tmp_path: Path) -> None:
    """H3-2d: the keypoint loader gains the guard it was missing."""
    from autovision.data.coco_keypoint_dataset import CocoKeypointDataset

    annotations_path = tmp_path / "person_keypoints.json"
    annotations_path.write_text(json.dumps(_keypoint_payload("../evil.png")))
    dataset = CocoKeypointDataset(annotations_path, tmp_path / "images")
    with pytest.raises(AutoVisionError, match="escapes the dataset root"):
        dataset[0]


def test_coco_keypoint_legal_relative_path_unchanged(tmp_path: Path) -> None:
    """H3-2d: legitimate relative file names keep resolving and decoding."""
    from autovision.data.coco_keypoint_dataset import CocoKeypointDataset

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    created = create_image(images_dir / "img1.jpg")
    annotations_path = tmp_path / "person_keypoints.json"
    annotations_path.write_text(json.dumps(_keypoint_payload(created.name)))
    dataset = CocoKeypointDataset(annotations_path, images_dir)
    item = dataset[0]
    assert item["image"].shape[0] == 3
    assert item["classes"].tolist() == [1]
    assert item["bboxes"].shape == (1, 4)


# ── H3-2e: broken annotation json fails loudly with the file name ─────────


def _detection_args_with_bytes(
    tmp_path: Path, content: bytes
) -> COCOObjectDetectionDataArgs:
    """Detection args whose annotations file holds raw bytes."""
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_bytes(content)
    return COCOObjectDetectionDataArgs(
        train=SplitArgs(annotations=annotations_path, images=tmp_path / "images"),
        val=SplitArgs(annotations=annotations_path, images=tmp_path / "images"),
    )


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    [
        pytest.param(b"", "is not valid JSON", id="empty-json"),
        pytest.param(
            b'{"images": [{"id": 1, "file_',
            "is not valid JSON",
            id="truncated-json",
        ),
        pytest.param(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "a.jpg", "width": 64}],
                    "categories": [{"id": 1, "name": "cat"}],
                    "annotations": [],
                }
            ).encode(),
            "missing the required key 'height'",
            id="missing-height-key",
        ),
        pytest.param(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 48}],
                    "categories": [{"id": 1, "name": "cat"}],
                    "annotations": [
                        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20]}
                    ],
                }
            ).encode(),
            "expected 4 values [x, y, w, h], got 3",
            id="bbox-wrong-length",
        ),
    ],
)
def test_coco_detection_loader_reports_broken_annotations(
    tmp_path: Path, mutation: bytes, expected_fragment: str
) -> None:
    """H3-2e: a 0-byte/truncated json, a missing 'width'/'height' key or a
    wrong-length bbox raise AutoVisionError naming the annotations file —
    never a bare KeyError/JSONDecodeError."""
    args = _detection_args_with_bytes(tmp_path, mutation)
    with pytest.raises(AutoVisionError, match=r"annotations\.json") as excinfo:
        list(args.list_image_info("train"))
    assert expected_fragment in str(excinfo.value)


def test_coco_detection_loader_empty_json_names_file_on_classes(tmp_path: Path) -> None:
    """H3-2e: the class-list path wraps the same way (file name included)."""
    args = _detection_args_with_bytes(tmp_path, b"")
    with pytest.raises(AutoVisionError, match=r"annotations\.json"):
        args.classes()


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    [
        pytest.param(b"", "is not valid JSON", id="empty-json"),
        pytest.param(
            b'{"images": [',
            "is not valid JSON",
            id="truncated-json",
        ),
        pytest.param(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "a.png", "width": 32}],
                    "categories": [{"id": 1, "name": "cat"}],
                    "annotations": [],
                }
            ).encode(),
            "missing the required key 'height'",
            id="missing-height-key",
        ),
        pytest.param(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "a.png", "width": 32, "height": 32}],
                    "categories": [{"id": 1, "name": "cat"}],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [1, 1, 5, 5, 9],
                            "segmentation": [[1.0, 1.0, 9.0, 1.0, 9.0, 9.0, 1.0, 9.0]],
                        }
                    ],
                }
            ).encode(),
            "expected 4 values [x, y, w, h], got 5",
            id="bbox-wrong-length",
        ),
    ],
)
def test_coco_instance_loader_reports_broken_annotations(
    tmp_path: Path, mutation: bytes, expected_fragment: str
) -> None:
    """H3-2e: the instance loader fails loudly (AutoVisionError with the
    file name) on broken json, missing keys and wrong-length bboxes."""
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_bytes(mutation)
    with pytest.raises(AutoVisionError, match=r"annotations\.json") as excinfo:
        COCOInstanceSegmentationDataset(annotations_path, images_dir=tmp_path)
    assert expected_fragment in str(excinfo.value)


def _keypoint_payload_bytes(bbox: Any) -> bytes:
    """A valid person payload whose bbox is replaced by ``bbox``."""
    keypoints: list[float] = []
    for index in range(17):
        keypoints += [5.0 + index, 10.0 + index, 2]
    payload = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 32, "height": 32}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": bbox,
                "iscrowd": 0,
                "num_keypoints": 17,
                "keypoints": keypoints,
            }
        ],
    }
    return json.dumps(payload).encode()


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    [
        pytest.param(b"", "is not valid JSON", id="empty-json"),
        pytest.param(
            b'{"images": [{"id": 1',
            "is not valid JSON",
            id="truncated-json",
        ),
        pytest.param(
            _keypoint_payload_bytes([5.0, 10.0, 20.0]),
            "expected 4 values [x, y, w, h], got 3",
            id="bbox-wrong-length",
        ),
    ],
)
def test_coco_keypoint_loader_reports_broken_annotations(
    tmp_path: Path, mutation: bytes, expected_fragment: str
) -> None:
    """H3-2e: the keypoint loader wraps broken json at construction and a
    wrong-length bbox at item time, both naming the annotations file."""
    from autovision.data.coco_keypoint_dataset import CocoKeypointDataset

    annotations_path = tmp_path / "person_keypoints.json"
    annotations_path.write_bytes(mutation)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    # The bbox check happens per item; a decodable image is needed for the
    # cases that survive construction (broken json raises before decode).
    create_image(images_dir / "a.jpg", size=(32, 32), seed=0)
    with pytest.raises(AutoVisionError, match=r"person_keypoints\.json") as excinfo:
        dataset = CocoKeypointDataset(annotations_path, images_dir)
        dataset[0]  # broken json never gets here (construction raised)
    assert expected_fragment in str(excinfo.value)


# ── H3-2f: label files read with explicit utf-8-sig encoding ──────────────


def test_yolo_labels_parse_with_utf8_bom(tmp_path: Path) -> None:
    """H3-2f: a UTF-8 BOM on the first label line must not corrupt the class
    id — the training loaders read labels with explicit ``utf-8-sig``."""
    args, label_path = _yolo_args(tmp_path, "")
    label_path.write_bytes(b"\xef\xbb\xbf" + b"0 0.5 0.5 0.2 0.2\n")
    rows = list(args.list_image_info("train"))
    assert json.loads(rows[0]["bboxes"]) == [[0.5, 0.5, 0.2, 0.2]]
    assert json.loads(rows[0]["class_labels"]) == [0]


def test_yolo_non_ascii_label_line_does_not_crash_decoding(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """H3-2f: explicit encoding means a utf-8 comment line degrades to the
    counted malformed-line warning instead of a UnicodeDecodeError."""
    args, label_path = _yolo_args(
        tmp_path, "0 0.5 0.5 0.2 0.2\n# 注释行：坐标见说明\n"
    )
    with caplog.at_level(logging.WARNING, logger="autovision"):
        rows = list(args.list_image_info("train"))
    assert len(rows) == 1
    assert json.loads(rows[0]["bboxes"]) == [[0.5, 0.5, 0.2, 0.2]]
    warnings = [r for r in caplog.records if "malformed" in r.getMessage()]
    assert len(warnings) == 1
    assert str(label_path) in warnings[0].getMessage()


# ── H1-9: dataset registry pins are mechanized ────────────────────────────

#: Registry URLs allowed to ship without a sha256 pin (H1-9 interim debt).
#: Once a pin is computed with ``scripts/compute_dataset_pins.py`` and the
#: digest recorded in the registry (with the provenance noted in
#: ``docs/provenance/LEDGER.md``), the URL MUST be removed from this set.
#: An empty allowlist is the target state.
UNPINNED_ALLOWLIST = frozenset(
    {
        "https://images.cocodataset.org/zips/val2017.zip",
        "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "https://images.cocodataset.org/zips/train2017.zip",
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
    }
)


def test_dataset_registry_entries_are_pinned() -> None:
    """H1-9: every direct-download DatasetFile must carry a sha256 pin.

    Entries that require manual acquisition are exempt; anything else
    unpinned must sit in ``UNPINNED_ALLOWLIST`` explicitly. Stale
    allowlist entries (URLs that gained a pin) are flagged too, so the
    set can actually shrink to empty.
    """
    unpinned: list[str] = []
    for name, spec in DATASET_REGISTRY.items():
        if spec.manual_instructions is not None:
            continue
        for entry in spec.files:
            if entry.sha256 is None:
                unpinned.append(entry.url)
    unallowlisted = sorted(set(unpinned) - UNPINNED_ALLOWLIST)
    assert not unallowlisted, (
        "Direct-download dataset artifacts without a sha256 pin and not in "
        f"UNPINNED_ALLOWLIST: {unallowlisted}. Compute the pin with "
        "scripts/compute_dataset_pins.py, record the digest in the registry "
        "and the provenance in docs/provenance/LEDGER.md, then remove the "
        "URL from UNPINNED_ALLOWLIST (an empty allowlist is the goal)."
    )
    stale = sorted(UNPINNED_ALLOWLIST - set(unpinned))
    assert not stale, (
        f"UNPINNED_ALLOWLIST lists now-pinned URLs; delete them: {stale}"
    )


def test_compute_dataset_pins_script_outputs_paste_ready_snippet(
    tmp_path: Path,
) -> None:
    """H1-9: the pin script hashes a file and prints a registry snippet."""
    payload = b"official archive bytes to pin"
    artifact = tmp_path / "artifact.zip"
    artifact.write_bytes(payload)
    expected_sha = hashlib.sha256(payload).hexdigest()
    script = Path(__file__).resolve().parents[1] / "scripts" / "compute_dataset_pins.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(artifact),
            "--url",
            "https://images.cocodataset.org/zips/val2017.zip",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert expected_sha in completed.stdout
    assert f"size_bytes={len(payload)}" in completed.stdout
    assert 'url="https://images.cocodataset.org/zips/val2017.zip"' in completed.stdout
    assert f'sha256="{expected_sha}"' in completed.stdout


def _load_pins_script() -> Any:
    """Import ``scripts/compute_dataset_pins.py`` as a module for in-process
    tests (URL hashing is monkeypatched, so no network is touched)."""
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "compute_dataset_pins.py"
    spec = importlib.util.spec_from_file_location("compute_dataset_pins", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_dataset_pins_script_hashes_from_url(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H1-9: URL targets stream through urllib and hash identically."""
    payload = b"streamed archive bytes"
    expected_sha = hashlib.sha256(payload).hexdigest()

    class _Response:
        """Minimal urlopen response serving ``payload`` once, then EOF."""

        status = 200
        _served = False

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self, size: int = -1) -> bytes:
            if self._served:
                return b""
            self._served = True
            return payload

    def _fake_urlopen(url: object, timeout: float | None = None) -> _Response:
        assert str(getattr(url, "full_url", url)).startswith("https://example.invalid/")
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    module = _load_pins_script()
    exit_code = module.main(["https://example.invalid/a.zip"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert expected_sha in output
    assert f"size_bytes={len(payload)}" in output
    assert 'url="https://example.invalid/a.zip"' in output


def test_compute_dataset_pins_script_reports_missing_files(tmp_path: Path) -> None:
    """H1-9: a missing local file fails with a non-zero exit, not a trace."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "compute_dataset_pins.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "missing.zip")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "FAILED" in completed.stderr


class TestOpenImageChannelContract:
    """DATA-09: returned channels always match the requested count.

    The torchvision UNCHANGED fast path used to leak the source's own
    channel count (an RGB JPEG for a 1-channel request came back 3-channel,
    an RGBA PNG 4-channel), so a num_input_channels=1 model silently
    received the wrong tensor shape depending on the file format.
    """

    def test_one_channel_request_on_rgb_source_returns_one_channel(self, tmp_path):
        image_path = tmp_path / "rgb.jpg"
        create_image(image_path, size=(32, 32), seed=1)
        tensor = open_image(image_path, num_channels=1)
        assert tensor.shape == (1, 32, 32)

    def test_one_channel_request_on_rgba_source_returns_one_channel(self, tmp_path):
        from PIL import Image

        image_path = tmp_path / "rgba.png"
        Image.new("RGBA", (16, 16), (255, 0, 0, 128)).save(image_path)
        tensor = open_image(image_path, num_channels=1)
        assert tensor.shape == (1, 16, 16)

    def test_three_channel_request_on_rgba_source_returns_three_channels(self, tmp_path):
        from PIL import Image

        image_path = tmp_path / "rgba.png"
        Image.new("RGBA", (16, 16), (255, 0, 0, 128)).save(image_path)
        tensor = open_image(image_path, num_channels=3)
        assert tensor.shape == (3, 16, 16)

    def test_grayscale_source_one_channel_request_passes_through(self, tmp_path):
        from PIL import Image

        image_path = tmp_path / "gray.png"
        Image.new("L", (16, 16), 128).save(image_path)
        tensor = open_image(image_path, num_channels=1)
        assert tensor.shape == (1, 16, 16)

    def test_invalid_channel_request_fails_loud(self, tmp_path):
        image_path = tmp_path / "rgb.jpg"
        create_image(image_path, size=(16, 16), seed=2)
        with pytest.raises(AutoVisionError, match="num_channels"):
            open_image(image_path, num_channels=2)


# ── DATA-06: value-parse errors ride the malformed-line channel ───────────


@pytest.mark.parametrize(
    "obb", [pytest.param(False, id="det"), pytest.param(True, id="obb")]
)
def test_yolo_garbage_values_count_as_malformed_not_crash(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, obb: bool
) -> None:
    """DATA-06: a field-count-legal line with garbage values is counted
    and warned with file/line context — the hot path must not crash with
    a bare ``ValueError`` that bypasses the deprecation channel."""
    if obb:
        text = "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n0 0.5 not_a_number 0.2 0.2 0.1 0.2 0.1 0.2\n"
    else:
        text = "0 0.5 0.5 0.2 0.2\n0 0.5 not_a_number 0.2 0.2\n"
    args, label_path = _yolo_args(tmp_path, text, obb=obb)
    with caplog.at_level(logging.WARNING, logger="autovision"):
        rows = list(args.list_image_info("train"))

    assert len(rows) == 1
    assert len(json.loads(rows[0]["bboxes"])) == 1
    message = next(
        r.getMessage() for r in caplog.records if "malformed" in r.getMessage()
    )
    assert "1 malformed" in message
    assert str(label_path) in message
    assert "line 2" in message
    assert "deprecated in 0.6.1" in message


# ── DATA-07: unknown-class lines are counted and warned, ignore silent ────


def test_yolo_unknown_class_id_warns_with_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """DATA-07: a hand-slipped class id is dropped but reported once."""
    args, label_path = _yolo_args(
        tmp_path, "0 0.5 0.5 0.2 0.2\n9 0.1 0.1 0.1 0.1\n9 0.2 0.2 0.1 0.1\n"
    )
    with caplog.at_level(logging.WARNING, logger="autovision"):
        rows = list(args.list_image_info("train"))

    assert len(rows) == 1
    assert len(json.loads(rows[0]["bboxes"])) == 1
    message = next(
        r.getMessage() for r in caplog.records if "class ids" in r.getMessage()
    )
    assert "2 YOLO label line(s)" in message
    assert str(label_path) in message
    assert "class id 9" in message


def test_yolo_unknown_class_warning_never_says_malformed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """DATA-07: the well-formed regression test filters on 'malformed'; the
    unknown-class warning must live in its own channel so clean-data runs
    with a stray class id do not read as corruption."""
    args, _label = _yolo_args(tmp_path, "9 0.1 0.1 0.1 0.1\n")
    with caplog.at_level(logging.WARNING, logger="autovision"):
        list(args.list_image_info("train"))
    assert not [r for r in caplog.records if "malformed" in r.getMessage()]
    assert [r for r in caplog.records if "class ids" in r.getMessage()]


def test_yolo_ignore_classes_drop_stays_silent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """ignore_classes drops are the documented contract — no warning noise."""
    images_dir = tmp_path / "train" / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "img.jpg").write_bytes(b"\xff\xd8stub")
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "train" / "labels" / "img.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.1 0.1\n"
    )
    args = YOLOObjectDetectionDataArgs(
        path=tmp_path,
        train=tmp_path / "train" / "images",
        val=tmp_path / "val" / "images",
        names={0: "a", 2: "ignored"},
        ignore_classes={2},
    )
    with caplog.at_level(logging.WARNING, logger="autovision"):
        rows = list(args.list_image_info("train"))
    assert len(json.loads(rows[0]["bboxes"])) == 1
    assert not [r for r in caplog.records if "class ids" in r.getMessage()]
    assert not [r for r in caplog.records if "malformed" in r.getMessage()]


# ── DATA-08: duplicate image ids fail loudly on every COCO plane ──────────


def test_instance_segmentation_duplicate_image_id_raises_with_both_names(
    tmp_path: Path,
) -> None:
    """DATA-08: the instance plane matches detection's H3-2b semantics."""
    payload = {
        "images": [
            {"id": 1, "file_name": "first.png", "width": 32, "height": 32},
            {"id": 1, "file_name": "second.png", "width": 32, "height": 32},
        ],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [],
    }
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(payload))
    with pytest.raises(AutoVisionError, match="duplicate image id 1") as excinfo:
        COCOInstanceSegmentationDataset(annotations_path, images_dir=tmp_path)
    assert "first.png" in str(excinfo.value)
    assert "second.png" in str(excinfo.value)


def test_keypoint_duplicate_image_id_raises_with_both_names(tmp_path: Path) -> None:
    """DATA-08: the keypoint plane matches detection's H3-2b semantics."""
    payload = {
        "images": [
            {"id": 7, "file_name": "one.jpg", "width": 64, "height": 64},
            {"id": 7, "file_name": "two.jpg", "width": 64, "height": 64},
        ],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [],
    }
    annotations_path = tmp_path / "person_keypoints.json"
    annotations_path.write_text(json.dumps(payload))
    from autovision.data.coco_keypoint_dataset import CocoKeypointDataset

    with pytest.raises(AutoVisionError, match="duplicate image id 7") as excinfo:
        CocoKeypointDataset(annotations_path, tmp_path)
    assert "one.jpg" in str(excinfo.value)
    assert "two.jpg" in str(excinfo.value)


# ── DATA-04: instance rows stay lazy RLE; decode failures name the row ────


def test_instance_segmentation_compressed_rle_roundtrips_lazily(tmp_path: Path) -> None:
    """DATA-04: the compressed-RLE json form decodes through the lazy path."""
    import numpy as np
    from pycocotools import mask as mask_utils

    arr = np.zeros((8, 8), dtype=np.uint8)
    arr[2:6, 2:6] = 1
    rle = mask_utils.encode(np.asfortranarray(arr))
    payload = {
        "images": [{"id": 1, "file_name": "a.png", "width": 8, "height": 8}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2, 2, 4, 4],
                "segmentation": {
                    "size": rle["size"],
                    "counts": rle["counts"].decode("ascii"),
                },
            }
        ],
    }
    from PIL import Image

    Image.new("RGB", (8, 8), (10, 20, 30)).save(tmp_path / "a.png")
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(payload))
    dataset = COCOInstanceSegmentationDataset(annotations_path, images_dir=tmp_path)
    assert isinstance(dataset._rows[0]["rles"][0]["counts"], str)
    assert int(dataset[0]["masks"][0].sum()) == 16


def test_instance_segmentation_corrupt_rle_fails_naming_the_row(
    tmp_path: Path,
) -> None:
    """DATA-04: a corrupt RLE surfaces at decode time as AutoVisionError
    naming the row and image, not a bare pycocotools traceback."""
    payload = _interleaved_seg_payload()
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(payload))
    from PIL import Image

    Image.new("RGB", (32, 32), (0, 0, 0)).save(tmp_path / "a.png")
    dataset = COCOInstanceSegmentationDataset(annotations_path, images_dir=tmp_path)
    # pycocotools tolerates arbitrary BYTES (LEB128), so corrupt the
    # type instead: decode raises TypeError and the loader wraps it.
    dataset._rows[0]["rles"][0]["counts"] = None
    with pytest.raises(AutoVisionError, match="row 0"):
        dataset[0]


# ── DATA-10: Label Studio images go through the shared containment guard ──


def test_label_studio_absolute_image_name_outside_images_dir_rejected(
    tmp_path: Path,
) -> None:
    """DATA-10: an absolute task path pointing outside images_dir must fail
    loudly — the old ``images_dir / name`` join silently discarded the
    prefix and opened (then copied verbatim) the outside file name."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    from PIL import Image

    Image.new("RGB", (16, 16), (1, 2, 3)).save(outside / "outside.png")
    task = {
        "data": {"image": f"/abs/or/not/{outside / 'outside.png'}"},
        "annotations": [],
    }
    source = tmp_path / "tasks.json"
    source.write_text(json.dumps([task]))
    from autovision.data.interchange import label_studio_to_coco

    with pytest.raises(AutoVisionError, match="escapes"):
        label_studio_to_coco(source, tmp_path / "coco.json", images_dir=tmp_path / "images")


def test_label_studio_traversal_image_name_rejected(tmp_path: Path) -> None:
    """DATA-10: ``..`` traversal in a Label Studio task is refused."""
    from PIL import Image

    Image.new("RGB", (16, 16), (4, 5, 6)).save(tmp_path / "secret.png")
    task = {"data": {"image": "../secret.png"}, "annotations": []}
    source = tmp_path / "tasks.json"
    source.write_text(json.dumps([task]))
    from autovision.data.interchange import label_studio_to_coco

    with pytest.raises(AutoVisionError, match="escapes"):
        label_studio_to_coco(source, tmp_path / "coco.json", images_dir=tmp_path / "images")


def test_label_studio_legal_relative_name_still_converts(tmp_path: Path) -> None:
    """DATA-10: the guard must not break the legal relative-path flow."""
    from PIL import Image

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (16, 16), (7, 8, 9)).save(images / "photo.png")
    task = {
        "data": {"image": "/data/local-dir/?d=photo.png"},
        "annotations": [
            {
                "result": [
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": 10, "y": 10, "width": 50, "height": 50,
                            "rectanglelabels": ["cat"],
                        },
                    }
                ]
            }
        ],
    }
    source = tmp_path / "tasks.json"
    source.write_text(json.dumps([task]))
    from autovision.data.interchange import label_studio_to_coco

    out = label_studio_to_coco(source, tmp_path / "coco.json", images_dir=images)
    payload = json.loads(out.read_text())
    assert payload["images"][0]["file_name"] == "photo.png"
    assert len(payload["annotations"]) == 1


# ── DATA-11: image/mask spatial alignment is checked up front ─────────────


def test_semantic_pair_mismatch_fails_naming_both_files(tmp_path: Path) -> None:
    """DATA-11: a misaligned semantic pair is corruption, named loudly —
    not a confusing crop/resize failure inside the transform."""
    from PIL import Image

    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir(), masks.mkdir()
    Image.new("RGB", (32, 32), (0, 0, 0)).save(images / "a.jpg")
    Image.new("L", (16, 24), 0).save(masks / "a.png")
    from autovision.data.semantic_segmentation_dataset import (
        SemanticSegmentationDataset,
        list_segmentation_pairs,
    )

    pairs, _ = list_segmentation_pairs(images, masks)
    dataset = SemanticSegmentationDataset(pairs)
    with pytest.raises(AutoVisionError, match="misaligned") as excinfo:
        dataset[0]
    assert "a.jpg" in str(excinfo.value)
    assert "a.png" in str(excinfo.value)
    assert "(32, 32)" in str(excinfo.value) and "(24, 16)" in str(excinfo.value)


# ── DATA-20/21/29/30 quick wins ───────────────────────────────────────────


def test_coco_non_positive_image_size_fails_with_file_context(tmp_path: Path) -> None:
    """DATA-20: a zero declared size dies at construction with the image
    named — not a bare ZeroDivisionError inside the normalization."""
    payload = {
        "images": [{"id": 1, "file_name": "a.png", "width": 0, "height": 32}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [1, 1, 10, 10], "area": 100, "iscrowd": 0}
        ],
    }
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(payload))
    args = COCOObjectDetectionDataArgs(
        train=SplitArgs(annotations=annotations_path, images=tmp_path),
        val=SplitArgs(annotations=annotations_path, images=tmp_path),
    )
    with pytest.raises(AutoVisionError, match="non-positive size") as excinfo:
        list(args.list_image_info("train"))
    assert "a.png" in str(excinfo.value)


def test_yolo_binary_label_file_fails_naming_the_label(tmp_path: Path) -> None:
    """DATA-21: a GBK/binary label file reports the file, not a bare
    UnicodeDecodeError from inside the row loop."""
    images_dir = tmp_path / "train" / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "img.jpg").write_bytes(b"\xff\xd8stub")
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "train" / "labels" / "img.txt").write_bytes(b"\xa5\xd6\xd0\xc4\n")
    args = YOLOObjectDetectionDataArgs(
        path=tmp_path,
        train=tmp_path / "train" / "images",
        val=tmp_path / "val" / "images",
        names={0: "a"},
    )
    with pytest.raises(AutoVisionError, match="not valid UTF-8") as excinfo:
        list(args.list_image_info("train"))
    assert "img.txt" in str(excinfo.value)


def test_coco_orphan_annotations_warn_with_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """DATA-29: annotations naming undeclared image ids are counted and
    reported (merge_coco parity), never silently ignored."""
    payload = {
        "images": [{"id": 1, "file_name": "a.png", "width": 32, "height": 32}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [1, 1, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 2, "image_id": 99, "category_id": 1,
             "bbox": [1, 1, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 3, "image_id": 99, "category_id": 1,
             "bbox": [2, 2, 5, 5], "area": 25, "iscrowd": 0},
        ],
    }
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(payload))
    args = COCOObjectDetectionDataArgs(
        train=SplitArgs(annotations=annotations_path, images=tmp_path),
        val=SplitArgs(annotations=annotations_path, images=tmp_path),
    )
    with caplog.at_level(logging.WARNING, logger="autovision"):
        rows = list(args.list_image_info("train"))
    assert len(rows) == 1
    assert len(json.loads(rows[0]["bboxes"])) == 1
    message = next(
        r.getMessage() for r in caplog.records if "not declared in 'images'" in r.getMessage()
    )
    assert "2 annotation(s)" in message


def test_coco_unknown_ignore_classes_warn_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """DATA-30: an ignore_classes entry naming no declared class warns once
    (a typo silently KEEPING the class is the dangerous failure mode)."""
    payload = {
        "images": [{"id": 1, "file_name": "a.png", "width": 32, "height": 32}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [1, 1, 10, 10], "area": 100, "iscrowd": 0}
        ],
    }
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(payload))

    def build() -> COCOObjectDetectionDataArgs:
        return COCOObjectDetectionDataArgs(
            train=SplitArgs(annotations=annotations_path, images=tmp_path),
            val=SplitArgs(annotations=annotations_path, images=tmp_path),
            ignore_classes={1, 42},
        )

    with caplog.at_level(logging.WARNING, logger="autovision"):
        first = list(build().list_image_info("train"))
    records = [r for r in caplog.records if "no-op" in r.getMessage()]
    assert len(records) == 1
    assert "42" in records[0].getMessage()
    with caplog.at_level(logging.WARNING, logger="autovision"):
        second = list(build().list_image_info("train"))
    assert len([r for r in caplog.records if "no-op" in r.getMessage()]) == 1  # deduped
    assert len(first) == len(second) == 1
    # class id 1 stays in the row: the unknown entry no-ops, the real one applies
    assert json.loads(first[0]["class_labels"]) == []


def test_instance_segmentation_orphan_annotations_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Review P3-9: instance-seg parity for DATA-29 — annotations naming
    undeclared image ids are counted and reported."""
    payload = {
        "images": [{"id": 1, "file_name": "a.png", "width": 32, "height": 32}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10],
             "segmentation": [1.0, 1.0, 20.0, 1.0, 20.0, 20.0, 1.0, 20.0]},
            {"id": 2, "image_id": 77, "category_id": 1, "bbox": [1, 1, 5, 5],
             "segmentation": [1.0, 1.0, 5.0, 1.0, 5.0, 5.0, 1.0, 5.0]},
        ],
    }
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(payload))
    with caplog.at_level(logging.WARNING, logger="autovision"):
        dataset = COCOInstanceSegmentationDataset(annotations_path, images_dir=tmp_path)
    assert len(dataset._rows) == 1
    message = next(
        r.getMessage()
        for r in caplog.records
        if "not declared in 'images'" in r.getMessage()
    )
    assert "1 annotation(s)" in message
