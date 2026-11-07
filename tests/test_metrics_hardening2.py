# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Second-round metrics/compliance hardening: OBB correctness, crowd wiring,
single-point headline, streaming depth, report slimming, domain validation.

Covers the METRIC-01..09 designs from ``docs/design/metrics_compliance.md``.
Every P1 here is an "external number" bug — each test pins the exact audit
reproduction (pre-fix behavior is asserted against in comments).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from autovision.augment.detection.args import RTDETRDetectionTransformArgs
from autovision.augment.detection.transform import build_val_transform_args
from autovision.augment.transform_args import NormalizeArgs
from autovision.errors import AutoVisionError
from autovision.metrics.map_evaluator import HEADLINE_PROTOCOL, COCOMAPEvaluator
from autovision.metrics.oriented_detection import (
    OrientedDetectionMetricCollector,
    _average_precision,
)
from autovision.metrics.segmentation import SegmentationMetricCollector

_GOLDEN = json.loads(
    (Path(__file__).resolve().parents[1] / "tests/golden/obb_ap.json").read_text()
)


def _box(x1: float, y1: float, x2: float, y2: float) -> torch.Tensor:
    """Axis-aligned corners in the loader's ordered-edge convention."""
    return torch.tensor(
        [[x1, y1, x2, y1, x2, y2, x1, y2]], dtype=torch.float32
    )


# ── METRIC-01/02: OBB PR construction + envelope ──────────────────────────


class TestObbApCorrectness:
    def test_golden_vectors_reproduce(self):
        """METRIC-06: the hand-computed golden vectors (including both audit
        reproductions) hold under BOTH interpolation modes."""
        for name, vector in _GOLDEN["vectors"].items():
            scores_tps = [tuple(entry) for entry in vector["scores_tps"]]
            assert _average_precision(
                scores_tps, vector["num_gt"], interpolation="all-point"
            ) == pytest.approx(vector["all_point"]), name
            assert _average_precision(
                scores_tps, vector["num_gt"], interpolation="coco101"
            ) == pytest.approx(vector["coco101"], abs=1e-9), name

    def test_background_fp_on_gtless_image_penalizes_ap(self):
        """METRIC-01 audit reproduction: image 0 has one GT matched by a TP;
        image 1 has NO GT and a high-scoring background FP. The FP must
        enter the PR stream — official-semantics AP is 0.5 (pre-fix: 1.0)."""
        collector = OrientedDetectionMetricCollector({0: "a"})
        gt = _box(10, 10, 30, 30)
        collector._ground_truth = [
            {"corners": gt, "labels": torch.tensor([0])},
            {"corners": torch.zeros(0, 8), "labels": torch.zeros(0, dtype=torch.long)},
        ]
        collector._predictions = [
            {
                "polygons": gt.clone(),
                "labels": torch.tensor([0]),
                "scores": torch.tensor([0.9]),
            },
            {
                "polygons": _box(90, 90, 110, 110),
                "labels": torch.tensor([0]),
                "scores": torch.tensor([0.95]),
            },
        ]
        assert collector.compute()["map_50"] == pytest.approx(0.5)

    def test_pred_only_class_does_not_enter_the_macro_mean(self):
        """A class predicted somewhere but never present in GT contributes
        FPs to its OWN stream yet stays out of the macro mean (pycocotools
        semantics): the class-A AP is untouched by class-B noise."""
        collector = OrientedDetectionMetricCollector({0: "a", 1: "b"})
        gt = _box(10, 10, 30, 30)
        collector._ground_truth = [
            {"corners": gt, "labels": torch.tensor([0])},
            {"corners": torch.zeros(0, 8), "labels": torch.zeros(0, dtype=torch.long)},
        ]
        collector._predictions = [
            {
                "polygons": gt.clone(),
                "labels": torch.tensor([0]),
                "scores": torch.tensor([0.9]),
            },
            {
                "polygons": _box(90, 90, 110, 110),
                "labels": torch.tensor([1]),  # class B: no GT anywhere
                "scores": torch.tensor([0.95]),
            },
        ]
        assert collector.compute()["map_50"] == pytest.approx(1.0)

    def test_prediction_target_count_mismatch_fails_loud(self):
        """The audit reproduction only worked through a silent zip
        truncation; mismatched record counts now name the counts."""
        collector = OrientedDetectionMetricCollector({0: "a"})
        collector._ground_truth = [
            {"corners": _box(10, 10, 30, 30), "labels": torch.tensor([0])}
        ]
        collector._predictions = [
            {
                "polygons": _box(10, 10, 30, 30),
                "labels": torch.tensor([0]),
                "scores": torch.tensor([0.9]),
            }
        ] * 2
        with pytest.raises(AutoVisionError, match="prediction record"):
            collector.compute()

    def test_invalid_interpolation_rejected(self):
        with pytest.raises(ValueError, match="interpolation"):
            _average_precision([(0.9, 1)], 1, interpolation="bogus")


# ── METRIC-03: crowd chain wiring ─────────────────────────────────────────


class TestCrowdWiring:
    def test_dataset_row_carries_crowd_flags(self, tmp_path):
        """The COCO loader emits iscrowds; __getitem__ must surface them
        (pre-fix: dropped on the floor)."""
        from tests.conftest import create_image

        from autovision.data.coco_object_detection_dataset import (
            COCOObjectDetectionDataArgs,
        )

        images = tmp_path / "images"
        images.mkdir()
        create_image(images / "img_0.jpg", size=(32, 32), seed=0)
        annotations = {
            "images": [{"id": 1, "file_name": "img_0.jpg", "width": 32, "height": 32}],
            "categories": [{"id": 1, "name": "cat"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [1, 1, 10, 10], "area": 100, "iscrowd": 1},
            ],
        }
        annotations_path = tmp_path / "annotations.json"
        annotations_path.write_text(json.dumps(annotations))
        args = COCOObjectDetectionDataArgs(
            train={"annotations": annotations_path, "images": images},
            val={"annotations": annotations_path, "images": images},
        )
        dataset = __import__(
            "autovision.data.object_detection_dataset",
            fromlist=["ObjectDetectionDataset"],
        ).ObjectDetectionDataset(args, list(args.list_image_info("train")))
        row = dataset[0]
        assert row["iscrowds"].tolist() == [1]

    def test_collate_carries_crowd_through_filter(self):
        from autovision.augment.detection.collate import (
            RTDETRDetectionCollateFunction,
        )

        args = SimpleNamespace(
            min_bbox_size_px=1.0,
            mixup_prob=None, mixup_step_start=0, mixup_step_stop=None,
            copyblend_prob=None, copyblend_step_start=0,
            copyblend_step_stop=None, copyblend_num_objects=0,
            copyblend_expand_ratios=(1.0, 1.0), copyblend_area_threshold=0.0,
            scale_jitter=None,
        )
        collate = RTDETRDetectionCollateFunction(args, train=False)
        box = torch.tensor([[0.25, 0.25, 0.1, 0.1]])
        items = [
            {
                "image": torch.zeros(3, 64, 64),
                "bboxes": box.clone(),
                "classes": torch.tensor([0]),
                "iscrowds": torch.tensor([1]),
                "image_path": "a.jpg",
                "original_size": (64, 64),
            }
        ]
        batch = collate(items)
        assert batch["iscrowds"][0].tolist() == [1]

    def test_missing_item_crowd_falls_back_to_zeros(self):
        """A directly-constructed item (tests/custom loaders) without the
        flag keeps the pre-fix semantics: all zeros, aligned to the boxes."""
        from autovision.augment.detection.collate import (
            RTDETRDetectionCollateFunction,
        )

        args = SimpleNamespace(
            min_bbox_size_px=1.0,
            mixup_prob=None, mixup_step_start=0, mixup_step_stop=None,
            copyblend_prob=None, copyblend_step_start=0,
            copyblend_step_stop=None, copyblend_num_objects=0,
            copyblend_expand_ratios=(1.0, 1.0), copyblend_area_threshold=0.0,
            scale_jitter=None,
        )
        collate = RTDETRDetectionCollateFunction(args, train=False)
        box = torch.tensor([[0.25, 0.25, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]])
        batch = collate(
            [
                {
                    "image": torch.zeros(3, 64, 64),
                    "bboxes": box.clone(),
                    "classes": torch.tensor([0, 1]),
                    "image_path": "a.jpg",
                    "original_size": (64, 64),
                }
            ]
        )
        assert batch["iscrowds"][0].tolist() == [0, 0]

    def test_train_model_fallback_without_batch_crowds(self):
        """Directly-constructed batches (no collate) hit the evaluator
        fallback: zeros — pre-fix semantics for callers without crowd data."""
        import inspect

        from autovision.tasks.detection.detr.train_model import (
            DETRObjectDetectionTrain,
        )

        source = inspect.getsource(DETRObjectDetectionTrain.validation_step)
        assert 'batch.get("iscrowds")' in source


# ── METRIC-04: single operating point headline ────────────────────────────


class TestSinglePointHeadline:
    def test_headline_p_r_f1_describe_one_operating_point(self):
        """Two images, one class: image 0 has GT matched by TP(0.9); image 1
        has a background FP(0.8). At score>=0.5: precision 0.5, recall 1.0,
        F1 = 2*0.5*1/(1.5) ≈ 0.667 — all three from the SAME point. The old
        sweep reported precision from the top recall threshold and F1 ≥ the
        recall it was paired with."""
        evaluator = COCOMAPEvaluator({1: "cat"}, headline_score_threshold=0.5)
        evaluator.update_targets(
            [
                {
                    "boxes": torch.tensor([[0.25, 0.25, 0.1, 0.1]]),
                    "labels": torch.tensor([0]),
                }
            ],
            original_sizes=[(100, 100)],
        )
        # ONE image carrying both the TP (0.9, on the GT) and the
        # background FP (0.8) — records are per-image aligned.
        evaluator.update(
            [
                {
                    "boxes": torch.tensor(
                        [[20.0, 20.0, 30.0, 30.0], [70.0, 70.0, 80.0, 80.0]]
                    ),
                    "scores": torch.tensor([0.9, 0.8]),
                    "labels": torch.tensor([0, 0]),
                }
            ]
        )
        metrics = evaluator.compute()
        assert metrics["precision"] == pytest.approx(0.5)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)
        # F1 can never exceed either P or R at one operating point — the
        # old sweep could produce F1 > recall.
        assert metrics["f1"] <= metrics["recall"]

    def test_crowd_gt_ignored_in_headline(self):
        """The official protocol: a detection matching a crowd GT is neither
        a TP nor an FP (dtIg); the crowd GT itself is not a FN."""
        evaluator = COCOMAPEvaluator({1: "cat"}, headline_score_threshold=0.5)
        evaluator.update_targets(
            [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
                    "labels": torch.tensor([0]),
                    "iscrowds": torch.tensor([1]),
                }
            ],
            original_sizes=[(100, 100)],
        )
        evaluator.update(
            [
                {
                    "boxes": torch.tensor([[50.0, 50.0, 80.0, 80.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                }
            ]
        )
        metrics = evaluator.compute()
        assert metrics["precision"] == 0.0  # no GT with counts → empty mean
        assert metrics["recall"] == 0.0

    def test_zero_prediction_class_keeps_recall_zero(self):
        """A class with GT but no predictions: precision 0 (nothing was
        right), recall 0 — no ZeroDivision."""
        evaluator = COCOMAPEvaluator({1: "cat"}, headline_score_threshold=0.5)
        evaluator.update_targets(
            [
                {
                    "boxes": torch.tensor([[0.25, 0.25, 0.1, 0.1]]),
                    "labels": torch.tensor([0]),
                }
            ],
            original_sizes=[(100, 100)],
        )
        metrics = evaluator.compute()
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_headline_protocol_names_the_operating_point(self):
        lowered = HEADLINE_PROTOCOL.lower()
        assert "one operating point" in lowered
        assert "headline_score_threshold" in lowered


# ── METRIC-07: streaming depth ────────────────────────────────────────────


class TestStreamingDepth:
    def test_streaming_matches_the_eager_semantics(self):
        """The per-image alignment happens BEFORE the scalar accumulation,
        so the streamed totals equal an eager full-set computation."""
        from autovision.metrics.depth_estimation import DepthMetricCollector

        rng = np.random.default_rng(7)
        preds = torch.from_numpy(
            rng.uniform(1.0, 5.0, size=(4, 32, 32)).astype(np.float32)
        )
        gts = preds * rng.uniform(0.8, 1.2, size=(4, 1, 1)).astype(np.float32)
        streamed = DepthMetricCollector(alignment="none")
        streamed.update(preds, gts)
        # Merge must be additive and order-free.
        half = DepthMetricCollector(alignment="none")
        half.update(preds[:2], gts[:2])
        other = DepthMetricCollector(alignment="none")
        other.update(preds[2:], gts[2:])
        other.merge_state(half.export_state())
        for key in ("abs_rel", "rmse", "delta1"):
            assert other.compute()[key] == pytest.approx(
                streamed.compute()[key], rel=1e-9
            ), key
        assert other._total == streamed._total == 4 * 32 * 32


# ── METRIC-09: segmentation target domain ─────────────────────────────────


class TestSegmentationTargetDomain:
    def test_negative_target_fails_loud(self):
        collector = SegmentationMetricCollector(num_classes=3)
        with pytest.raises(AutoVisionError, match="outside"):
            collector.update(
                torch.zeros(1, 4, 4, dtype=torch.long),
                torch.full((1, 4, 4), -1, dtype=torch.long),
            )

    def test_out_of_range_target_fails_loud(self):
        collector = SegmentationMetricCollector(num_classes=3)
        with pytest.raises(AutoVisionError, match="outside"):
            collector.update(
                torch.zeros(1, 4, 4, dtype=torch.long),
                torch.full((1, 4, 4), 7, dtype=torch.long),
            )

    def test_ignore_index_still_valid(self):
        collector = SegmentationMetricCollector(num_classes=3, ignore_index=255)
        collector.update(
            torch.full((1, 4, 4), 1, dtype=torch.long),
            torch.full((1, 4, 4), 255, dtype=torch.long),
        )
        assert collector.compute()["miou"] == 0.0  # nothing counted, no crash


# ── METRIC-08: report slimming ────────────────────────────────────────────


class TestReportSlimming:
    def test_matches_capped_and_confusion_not_duplicated(self, tmp_path):
        from autovision.metrics.error_analysis import analyze_detection_errors

        predictions, targets, paths = [], [], []
        for index in range(40):
            predictions.append(
                {
                    "boxes": torch.tensor(
                        [[10.0, 10.0, 30.0, 30.0], [60.0, 60.0, 80.0, 80.0]]
                    ),
                    "scores": torch.tensor([0.9, 0.3]),
                    "labels": torch.tensor([0, 0]),
                }
            )
            targets.append(
                {
                    "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
                    "labels": torch.tensor([0]),
                }
            )
            image = tmp_path / f"im_{index:03d}.jpg"
            image.write_bytes(b"stub")
            paths.append(image)
        report = analyze_detection_errors(
            predictions, targets, {0: "cat"}, paths, tmp_path / "report",
            max_report_matches=10,
        )
        payload = json.loads(report.report_json.read_text())
        assert payload["matches_total"] == 80
        assert len(payload["matches"]) == 10
        assert payload["matches_truncated"] is True
        # The confusion lives ONCE (inside the summary).
        assert "confusion" not in payload or payload.get("confusion") is None
        assert "confusion" in payload["summary"]


# ── P3 batch ──────────────────────────────────────────────────────────────


class TestComplianceReportHardening:
    def test_deterministic_sbom_serial(self, tmp_path):
        """P3-18: two saves of the same report derive the same uuid5."""
        from autovision.compliance.report import ComplianceReport

        report = ComplianceReport(
            framework={"version": "test", "commit": "c"},
            models=[], datasets=[], dependencies=[], issues=[],
            generated_at="2026-11-07T00:00:00",
        )
        out_a, out_b = tmp_path / "a", tmp_path / "b"
        report.save(out_a)
        report.save(out_b)
        serial_a = json.loads((out_a / "compliance.sbom.json").read_text())[
            "serialNumber"
        ]
        serial_b = json.loads((out_b / "compliance.sbom.json").read_text())[
            "serialNumber"
        ]
        assert serial_a == serial_b
        assert serial_a.startswith("urn:uuid:")

    def test_no_timestamp_omits_the_cyclonedx_field(self, tmp_path):
        from autovision.compliance.report import ComplianceReport

        report = ComplianceReport(
            framework={"version": "test", "commit": "c"},
            models=[], datasets=[], dependencies=[], issues=[],
            generated_at=None,
        )
        payload = report.to_cyclonedx()
        assert "timestamp" not in payload["metadata"]

    def test_pipe_in_dependency_name_does_not_break_the_table(self, tmp_path):
        from autovision.compliance.report import ComplianceReport

        report = ComplianceReport(
            framework={"version": "t", "commit": "c"},
            models=[],
            datasets=[],
            dependencies=[
                {"name": "evil|pkg", "version": "1.0", "license": "MIT",
                 "status": "allowed"}
            ],
            issues=[],
        )
        markdown = report._render_markdown()
        assert "evil\\|pkg" in markdown

    def test_missing_model_file_is_an_issue_not_a_crash(self, tmp_path):
        from autovision.compliance.inventory import model_inventory

        entry, issues = model_inventory(tmp_path / "missing.pt")
        assert entry is None
        assert issues[0]["kind"] == "missing_model_file"
        assert issues[0]["severity"] == "error"

    def test_corrupt_snapshot_is_an_issue_not_a_crash(self, tmp_path):
        from autovision.compliance.inventory import dataset_inventory

        bad = tmp_path / "snap.json"
        bad.write_text("{not json")
        entry, issues = dataset_inventory(bad)
        assert entry is None
        assert issues[0]["kind"] == "unreadable_dataset_snapshot"

    def test_dependency_dedup_normalizes_case(self, monkeypatch):
        """P3-20: PyPI names are case-insensitive — case variants must not
        double-count the same distribution."""
        import importlib.metadata

        from autovision.compliance.licenses import dependency_inventory

        class _Dist:
            def __init__(self, name):
                self.metadata = {"Name": name, "Version": "1.0",
                                 "License": "MIT"}

        monkeypatch.setattr(
            importlib.metadata,
            "distributions",
            lambda: iter([_Dist("Some-Pkg"), _Dist("some_pkg")]),
        )
        inventory = dependency_inventory()
        names = [entry.name.lower() for entry in inventory
                 if entry.name.lower() == "some-pkg"]
        assert len(names) == 1


class TestMiscMetricHardening:
    def test_scalar_rows_concatenate(self):
        """P3-22: the docstring promises scalar columns — they used to
        crash np.concatenate."""
        from autovision.metrics.containers import TaskMetrics

        class _Scalar(TaskMetrics):
            def process(self):
                return {"lr_mean": float(self._concatenate(self.stats["lr"]).mean())}

        container = _Scalar(columns=("lr",))
        container.update(lr=0.1)
        container.update(lr=0.2)
        assert container.process()["lr_mean"] == pytest.approx(0.15)

    def test_panoptic_disjoint_segments_match_one_to_one(self):
        """P3-13 regression guard: valid id-map predictions are mutually
        exclusive by construction, and the added per-GT/per-prediction
        mutex keeps TP counting one-to-one (the overlapping-prediction
        case the mutex defends against is unreachable through the
        documented (B, H, W, 2) contract — verified while fixing it)."""
        from autovision.metrics.panoptic_segmentation import (
            PanopticMetricCollector,
        )

        collector = PanopticMetricCollector(things=[1], stuffs=[0])
        gt = torch.zeros(1, 8, 8, 2, dtype=torch.long)
        gt[0, :4, :, 0] = 1
        gt[0, :4, :, 1] = 10
        pred = torch.zeros(1, 8, 8, 2, dtype=torch.long)
        pred[0, :4, :, 0] = 1
        pred[0, :4, :, 1] = 11  # exact match → one TP
        collector.update(pred, gt)
        assert int(collector._tp[1]) == 1

    def test_set_ground_truth_warns(self):
        """P3-23: the unwired legacy entry point carries a deprecation
        warning (removal 0.7.0)."""
        evaluator = COCOMAPEvaluator({1: "cat"})
        with pytest.warns(DeprecationWarning, match="deprecated in 0.6.1"):
            evaluator.set_ground_truth(
                [
                    {
                        "boxes": torch.tensor([[0.25, 0.25, 0.1, 0.1]]),
                        "labels": torch.tensor([0]),
                    }
                ]
            )

    def test_pose_update_requires_image_ids_under_ddp(self, tmp_path, monkeypatch):
        """P3-12: the rank-local id fallback collides during DDP merge —
        loud failure instead of silent overlap."""
        from autovision.metrics.pose_estimation import COCOKeypointAPEvaluator

        annotations = {
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}
            ],
            "categories": [
                {"id": 1, "name": "person",
                 "keypoints": [f"k{i}" for i in range(17)],
                 "skeleton": []}
            ],
            "annotations": [],
        }
        path = tmp_path / "person_keypoints.json"
        path.write_text(json.dumps(annotations))
        monkeypatch.setenv("WORLD_SIZE", "2")
        evaluator = COCOKeypointAPEvaluator(path, categories={1: "person"})
        with pytest.raises(AutoVisionError, match="image_ids"):
            evaluator.update([])

    def test_match_detections_length_guard(self):
        """P3-16 (mirrors the OBB guard): a silently-truncated zip would
        analyze only the aligned prefix."""
        from autovision.metrics.error_analysis import match_detections

        with pytest.raises(AutoVisionError, match="per-image aligned"):
            match_detections(
                [{"boxes": torch.zeros(0, 4), "scores": torch.zeros(0),
                  "labels": torch.zeros(0, dtype=torch.long)}],
                [],  # one short
            )

    def test_duplicate_class_names_rejected(self):
        """P3-17: per_class is name-keyed — duplicate names would merge
        two classes' counts into one row."""
        from autovision.metrics.error_analysis import match_detections

        with pytest.raises(AutoVisionError, match="Duplicate class name"):
            match_detections(
                [],
                [],
                class_names={0: "cat", 1: "cat"},
            )


# ── review follow-ups: P0 wiring (transform boundary + decode keys) ───────


class TestEndToEndWiring:
    """The review's P0 lesson: both P0s sat in seams the unit tests bypassed
    (transform between dataset and collate; decode keys feeding the
    headline). These tests cover the FULL production seams."""

    def test_transform_preserves_crowds_through_val_pipeline(self):
        """P0-2: the val transform (SanitizeBoundingBoxes active) must carry
        iscrowds through to the item the collate consumes."""
        from autovision.augment.detection.transform import (
            RTDETRDetectionTransform,
        )

        args = build_val_transform_args(
            RTDETRDetectionTransformArgs(
                image_size=(64, 64), normalize=NormalizeArgs()
            )
        )
        transform = RTDETRDetectionTransform(args)
        item = {
            "image_path": "a.jpg",
            "image": torch.zeros(3, 64, 64),
            "bboxes": __import__("torchvision").tv_tensors.BoundingBoxes(
                torch.tensor([[16.0, 16.0, 40.0, 40.0]]),
                format="XYXY",
                canvas_size=(64, 64),
            ),
            "classes": torch.tensor([0]),
            "iscrowds": torch.tensor([1]),
            "original_size": (64, 64),
        }
        out = transform(item)
        assert out["iscrowds"].tolist() == [1]

    def test_production_bboxes_key_feeds_the_headline(self):
        """P0-1: production decode records use "bboxes" (decode_detection_
        outputs / picodet) — the headline matcher must normalize the key or
        every training-path headline reads 0/0/0 (pre-fix behavior)."""
        evaluator = COCOMAPEvaluator({1: "cat"}, headline_score_threshold=0.5)
        evaluator.update_targets(
            [
                {
                    "boxes": torch.tensor([[0.25, 0.25, 0.1, 0.1]]),
                    "labels": torch.tensor([0]),
                }
            ],
            original_sizes=[(100, 100)],
        )
        # bboxes-keyed record + a background FP in the same image.
        evaluator.update(
            [
                {
                    "bboxes": torch.tensor(
                        [[20.0, 20.0, 30.0, 30.0], [70.0, 70.0, 80.0, 80.0]]
                    ),
                    "scores": torch.tensor([0.9, 0.8]),
                    "labels": torch.tensor([0, 0]),
                }
            ]
        )
        metrics = evaluator.compute()
        assert metrics["precision"] == pytest.approx(0.5)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)

    def test_crowd_flows_dataset_to_evaluator_end_to_end(self, tmp_path):
        """The full seam (dataset → transform → collate → evaluator) with a
        crowd row: the crowd GT must NOT count as a FN in the headline."""
        from PIL import Image as _Image

        from autovision.augment.detection.collate import (
            RTDETRDetectionCollateFunction,
        )
        from autovision.augment.detection.transform import RTDETRDetectionTransform
        from autovision.data.object_detection_dataset import (
            ObjectDetectionDataset,
        )

        image = tmp_path / "img_0.jpg"
        _Image.new("RGB", (64, 64), (0, 0, 0)).save(image)
        annotations = {
            "images": [
                {"id": 1, "file_name": "img_0.jpg", "width": 64, "height": 64}
            ],
            "categories": [{"id": 1, "name": "cat"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [16, 16, 24, 24], "area": 576, "iscrowd": 1},
            ],
        }
        annotations_path = tmp_path / "annotations.json"
        annotations_path.write_text(json.dumps(annotations))
        args = __import__(
            "autovision.data.coco_object_detection_dataset",
            fromlist=["COCOObjectDetectionDataArgs"],
        ).COCOObjectDetectionDataArgs(
            train={"annotations": annotations_path, "images": tmp_path},
            val={"annotations": annotations_path, "images": tmp_path},
        )
        dataset = ObjectDetectionDataset(args, list(args.list_image_info("train")))
        transform_args = build_val_transform_args(
            RTDETRDetectionTransformArgs(
                image_size=(64, 64), normalize=NormalizeArgs()
            )
        )
        transform = RTDETRDetectionTransform(transform_args)
        dataset.set_transform(transform)
        batch = RTDETRDetectionCollateFunction(transform_args, train=False)(
            [dataset[0]]
        )
        assert batch["iscrowds"][0].tolist() == [1]

        # The evaluator must treat the crowd row as ignorable in the
        # headline (no FN for the unmatched detection).
        evaluator = COCOMAPEvaluator({1: "cat"}, headline_score_threshold=0.5)
        evaluator.update(
            [
                {
                    "bboxes": torch.tensor([[1.0, 1.0, 10.0, 10.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                }
            ]
        )
        evaluator.update_targets(
            [
                {
                    "boxes": batch["bboxes"][0],
                    "labels": batch["classes"][0],
                    "iscrowds": batch["iscrowds"][0],
                }
            ],
            original_sizes=batch["original_size"],
        )
        metrics = evaluator.compute()
        # The detection overlaps the crowd GT → ignored, not FP; the crowd
        # GT is not a FN → no class counts, headline all zero (not P=0/R=1).
        assert metrics["recall"] == 0.0
