# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""InferenceResult: record normalization, selection, and mask serialization."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from autovision.errors import AutoVisionError, ResultValidationError
from autovision.inference import InferenceResult
from autovision.results import mask_to_bool


def _detection(n: int = 2) -> InferenceResult:
    return InferenceResult.from_record(
        {
            "bboxes": torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float32)[:n],
            "scores": torch.tensor([0.9, 0.8], dtype=torch.float32)[:n],
            "labels": torch.tensor([0, 1], dtype=torch.long)[:n],
        },
        metadata={"source": "clip.mp4", "frame": 3},
    )


class TestConstruction:
    def test_detection_record_round_trips_and_infers_task(self):
        result = _detection()
        assert len(result) == 2
        assert result.inferred_task == "object_detection"
        record = result.to_record()
        assert set(record) == {"bboxes", "scores", "labels", "metadata"}
        assert record["metadata"]["source"] == "clip.mp4"

    def test_boxes_alias_is_normalized(self):
        result = InferenceResult.from_record(
            {"boxes": torch.zeros(1, 4), "scores": torch.ones(1)}
        )
        assert result.bboxes is not None
        assert result.to_record()["bboxes"].shape == (1, 4)

    def test_unknown_fields_are_preserved_in_extra(self):
        result = InferenceResult.from_record(
            {
                "bboxes": torch.zeros(2, 4),
                "scores": torch.ones(2),
                "labels": torch.zeros(2, dtype=torch.long),
                "custom": {"nested": 1},
            }
        )
        assert result.extra["custom"] == {"nested": 1}
        assert result.to_record()["custom"] == {"nested": 1}

    def test_extra_cannot_shadow_typed_fields(self):
        with pytest.raises(ValueError, match="must not duplicate"):
            InferenceResult(bboxes=torch.zeros(1, 4), extra={"bboxes": []})

    def test_pose_and_classification_cardinality(self):
        pose = InferenceResult(
            keypoints=torch.zeros(3, 17, 3),
            scores=torch.ones(3),
            labels=torch.zeros(3, dtype=torch.long),
        )
        assert len(pose) == 3
        assert pose.inferred_task == "pose_estimation"
        classification = InferenceResult(
            scores=torch.tensor([0.7]), labels=torch.tensor([4])
        )
        assert len(classification) == 1
        assert classification.inferred_task == "image_classification"

    def test_obb_polygons_accept_flat_form(self):
        result = InferenceResult(
            polygons=torch.tensor([[0, 0, 10, 0, 10, 10, 0, 10]], dtype=torch.float32),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([0]),
        )
        assert result.polygons is not None
        assert tuple(result.polygons.shape) == (1, 4, 2)
        assert result.inferred_task == "oriented_object_detection"

    def test_panoptic_cardinality_uses_segment_ids(self):
        result = InferenceResult(
            masks=torch.zeros(8, 8, 2, dtype=torch.long),
            segment_ids=torch.tensor([0, 1, 2]),
            scores=torch.tensor([0.9, 0.8, 0.7]),
        )
        assert len(result) == 3
        assert result.inferred_task == "panoptic_segmentation"
        with pytest.raises(ValueError, match="not defined"):
            result.select(0)

    def test_depth_and_semantic_masks_are_image_level(self):
        depth = InferenceResult(depth=torch.zeros(4, 5))
        assert len(depth) == 1
        assert depth.inferred_task == "depth_estimation"
        semantic = InferenceResult(mask=torch.zeros(4, 5, dtype=torch.long))
        assert len(semantic) == 1
        assert semantic.inferred_task == "semantic_segmentation"

    def test_empty_detection(self):
        empty = InferenceResult.from_record({"bboxes": [], "scores": [], "labels": []})
        assert empty.is_empty
        assert empty.inferred_task == "object_detection"


class TestSelection:
    def test_select_returns_fresh_metadata_and_rows(self):
        result = _detection()
        selected = result.select([1])
        assert selected.bboxes.tolist() == [[20.0, 20.0, 30.0, 30.0]]
        selected.metadata["frame"] = 99
        assert result.metadata["frame"] == 3

    def test_filter_scores_and_boolean_lists(self):
        result = _detection()
        filtered = result.filter_scores(0.85)
        assert filtered.labels.tolist() == [0]
        filtered = result.filter([True, False])
        assert filtered.scores.tolist() == pytest.approx([0.9])

    def test_filter_scores_warns_once_for_nan_rows(self) -> None:
        """H3-1j: 含 NaN 分数的行被丢弃时输出一次 warning，不再静默丢行。"""
        result = InferenceResult(
            bboxes=torch.zeros(3, 4),
            scores=torch.tensor([0.9, float("nan"), 0.2]),
            labels=torch.zeros(3, dtype=torch.long),
        )
        with pytest.warns(UserWarning, match="1 row") as caught:
            filtered = result.filter_scores(0.5)
        assert len(caught) == 1
        assert filtered.labels.tolist() == [0]

    def test_select_int_on_empty_result_raises_framework_error(self) -> None:
        """H3-1f: 空结果 select(0) 报 AutoVisionError 而非裸 IndexError。"""
        empty = InferenceResult.from_record({"bboxes": [], "scores": [], "labels": []})
        with pytest.raises(AutoVisionError, match="non-empty"):
            empty.select(0)

    def test_select_error_names_segment_based_results_accurately(self) -> None:
        """H3-1f: segment_ids-only 结果的报错文案不再谎称 panoptic。"""
        result = InferenceResult(
            segment_ids=torch.tensor([0, 1]),
            scores=torch.tensor([0.9, 0.8]),
        )
        with pytest.raises(ValueError, match="segment-based"):
            result.select([0])

    def test_select_does_not_slice_image_level_masks(self):
        result = InferenceResult(
            bboxes=torch.zeros(2, 4),
            scores=torch.ones(2),
            labels=torch.zeros(2, dtype=torch.long),
            mask=torch.zeros(4, 4, dtype=torch.long),
        )
        selected = result.select(0)
        assert selected.mask is result.mask
        assert selected.bboxes.shape == (1, 4)


class TestSerialization:
    def test_instance_masks_default_to_rle(self):
        from pycocotools import mask as mask_utils

        masks = torch.zeros(2, 6, 8, dtype=torch.bool)
        masks[0, 1:3, 2:5] = True
        masks[1, 4:6, 0:2] = True
        result = InferenceResult(
            bboxes=torch.tensor([[0, 0, 8, 6], [0, 0, 8, 6]], dtype=torch.float32),
            scores=torch.tensor([0.9, 0.8]),
            labels=torch.tensor([0, 1]),
            masks=masks,
        )
        payload = result.to_json()
        assert len(payload["masks"]) == 2
        decoded = mask_utils.decode(
            {
                "size": payload["masks"][0]["size"],
                "counts": payload["masks"][0]["counts"].encode("ascii"),
            }
        )
        assert np.array_equal(decoded.astype(bool), masks[0].numpy())

    def test_masks_none_omits_the_payload(self):
        masks = torch.ones(1, 2, 2, dtype=torch.bool)
        result = InferenceResult(
            bboxes=torch.zeros(1, 4),
            scores=torch.ones(1),
            labels=torch.zeros(1, dtype=torch.long),
            masks=masks,
        )
        assert "masks" not in result.to_json(masks="none")

    def test_dense_mode_is_capped_for_large_arrays(self):
        result = InferenceResult(depth=torch.arange(100, dtype=torch.float32).reshape(10, 10))
        payload = result.to_json(max_elements=10)
        assert payload["depth"]["summarized"] is True
        assert payload["depth"]["shape"] == [10, 10]

    def test_dense_mode_over_100x_budget_raises(self) -> None:
        """H3-1a: dense 模式超过 max_elements*100 硬闸 → AutoVisionError。"""
        result = InferenceResult(
            bboxes=torch.zeros(1, 4),
            scores=torch.ones(1),
            labels=torch.zeros(1, dtype=torch.long),
            masks=torch.zeros(1, 30, 30, dtype=torch.bool),  # 900 > 1*100
        )
        with pytest.raises(AutoVisionError, match="masks='rle'"):
            result.to_json(masks="dense", max_elements=1)

    def test_dense_mode_at_budget_boundary_passes(self) -> None:
        """H3-1a: 恰好 max_elements*100 的 dense 输出保持完整不被摘要。"""
        masks = torch.zeros(1, 10, 10, dtype=torch.bool)  # 100 == 1*100
        masks[0, 0, 0] = True
        result = InferenceResult(
            bboxes=torch.zeros(1, 4),
            scores=torch.ones(1),
            labels=torch.zeros(1, dtype=torch.long),
            masks=masks,
        )
        payload = result.to_json(masks="dense", max_elements=1)
        assert payload["masks"][0][0][0] is True

    def test_dense_mode_gate_covers_panoptic_maps(self) -> None:
        """H3-1a: panoptic 图的 dense 模式同样受硬闸约束。"""
        result = InferenceResult(
            masks=torch.zeros(30, 30, 2, dtype=torch.long),
            segment_ids=torch.tensor([0, 1]),
            scores=torch.tensor([0.9, 0.8]),
        )
        with pytest.raises(AutoVisionError, match="safety limit"):
            result.to_json(masks="dense", max_elements=1)

    def test_non_numeric_extra_array_summarizes_without_crashing(self) -> None:
        """H3-1d: extra 的字符串数组摘要只输出 shape/dtype/length，不崩。"""
        result = InferenceResult(
            bboxes=torch.zeros(1, 4),
            extra={"names": np.array([f"id-{i}" for i in range(50)])},
        )
        summary = result.to_json(max_elements=10)["names"]
        assert summary["summarized"] is True
        assert summary["shape"] == [50]
        assert summary["dtype"] == "<U5"
        assert summary["length"] == 50
        assert "min" not in summary
        assert "max" not in summary
        assert "mean" not in summary

    def test_object_string_array_summary_handles_value_error(self) -> None:
        """H3-1d: object 字符串数组在 float(np.min) 处抛 ValueError，同样降级。"""
        result = InferenceResult(
            bboxes=torch.zeros(1, 4),
            extra={"tags": np.array([f"t{i}" for i in range(50)], dtype=object)},
        )
        summary = result.to_json(max_elements=10)["tags"]
        assert summary["summarized"] is True
        assert summary["length"] == 50
        assert "min" not in summary

    def test_mask_to_bool_integer_negatives_are_background(self) -> None:
        """H3-1g: 整数路径 > 0，-1（panoptic unassigned 约定）不算前景。"""
        array = np.array([[0, -1, 2]], dtype=np.int64)
        assert mask_to_bool(array).tolist() == [[False, False, True]]

    def test_semantic_bool_mask_encodes_as_rle(self):
        result = InferenceResult(mask=torch.zeros(4, 5, dtype=torch.bool))
        payload = result.to_json()
        assert set(payload["mask"]) == {"size", "counts"}

    def test_semantic_label_map_uses_lossless_rle(self):
        labels = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
        result = InferenceResult(mask=labels)
        payload = result.to_json()["mask"]
        assert payload["size"] == [2, 3]
        assert payload["counts"] == [2, 2, 2]
        assert payload["values"] == [0, 1, 2]

    def test_panoptic_map_encodes_both_channels(self):
        masks = torch.zeros(4, 5, 2, dtype=torch.long)
        masks[:, :, 0] = 1
        masks[:, :, 1] = 4
        result = InferenceResult(
            masks=masks,
            segment_ids=torch.tensor([0, 1, 2, 3]),
            scores=torch.tensor([0.9, 0.8, 0.7, 0.6]),
        )
        payload = result.to_json()["masks"]
        assert payload["size"] == [4, 5]
        assert payload["class_ids"]["values"] == [1]
        assert payload["segment_ids"]["values"] == [4]

    def test_metadata_and_task_are_serialized_when_present(self):
        result = InferenceResult(
            bboxes=torch.zeros(1, 4),
            scores=torch.ones(1),
            labels=torch.zeros(1, dtype=torch.long),
            metadata={"source": "a.mp4"},
            task="object_detection",
        )
        payload = result.to_json()
        assert payload["metadata"] == {"source": "a.mp4"}
        assert payload["task"] == "object_detection"

    def test_to_numpy_converts_typed_fields(self):
        result = _detection().to_numpy()
        assert isinstance(result.bboxes, np.ndarray)
        assert result.to_json()["bboxes"] == [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]

    def test_invalid_masks_mode_raises(self):
        with pytest.raises(ValueError, match="masks must be"):
            _detection().to_json(masks="png")

    def test_integer_and_float_masks_are_rle_encoded(self):
        from pycocotools import mask as mask_utils

        masks = torch.zeros(2, 4, 4, dtype=torch.uint8)
        masks[0, 1:3, 1:3] = 255
        masks[1, 1:3, 1:3] = 1
        floats = torch.zeros(1, 4, 4)
        floats[0, 0:2, 0:2] = 0.9
        for value in (masks, floats):
            result = InferenceResult(
                bboxes=torch.zeros(value.shape[0], 4),
                scores=torch.ones(value.shape[0]),
                labels=torch.zeros(value.shape[0], dtype=torch.long),
                masks=value,
            )
            payload = result.to_json()
            decoded = mask_utils.decode(
                {
                    "size": payload["masks"][0]["size"],
                    "counts": payload["masks"][0]["counts"].encode("ascii"),
                }
            )
            assert decoded.sum() == 4

    def test_per_instance_fields_are_never_summarized(self):
        result = InferenceResult(
            bboxes=torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=torch.float32),
            scores=torch.tensor([0.9, 0.8]),
            labels=torch.tensor([0, 1]),
            depth=torch.arange(100, dtype=torch.float32).reshape(10, 10),
        )
        payload = result.to_json(max_elements=4)
        assert payload["bboxes"] == [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]]
        assert payload["labels"] == [0, 1]
        assert payload["depth"]["summarized"] is True

    def test_non_finite_summary_is_json_safe(self):
        depth = torch.tensor([[0.0, float("nan")], [float("inf"), 2.0]])
        payload = InferenceResult(depth=depth).to_json(max_elements=1)
        assert payload["depth"]["summarized"] is True
        assert payload["depth"]["min"] == 0.0
        assert payload["depth"]["max"] == 2.0

    def test_filter_scores_moves_index_to_mask_device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required to exercise mixed-device selection")
        result = InferenceResult(
            bboxes=torch.zeros(2, 4),
            scores=torch.tensor([0.9, 0.1], device="cuda"),
            labels=torch.zeros(2, dtype=torch.long),
            masks=torch.zeros(2, 4, 4, dtype=torch.bool),
        )
        filtered = result.filter_scores(0.5)
        assert filtered.masks is not None
        assert filtered.masks.shape[0] == 1

    def test_panoptic_filter_scores_is_a_noop(self):
        result = InferenceResult(
            masks=torch.zeros(4, 4, 2, dtype=torch.long),
            segment_ids=torch.tensor([0, 1]),
            scores=torch.tensor([0.9, 0.8]),
        )
        assert result.filter_scores(0.95) is result

    def test_masks_only_result_is_instance_segmentation(self):
        result = InferenceResult(
            labels=torch.tensor([0]),
            scores=torch.tensor([0.9]),
            masks=torch.ones(1, 4, 4, dtype=torch.bool),
        )
        assert result.inferred_task == "instance_segmentation"

    def test_numpy_bool_list_filters_rows(self):
        result = _detection()
        filtered = result.filter([np.True_, np.False_])
        assert filtered.labels.tolist() == [0]


class TestValidation:
    def test_bbox_shape_is_validated(self):
        with pytest.raises(ValueError, match="bboxes must have shape"):
            InferenceResult(bboxes=torch.zeros(2, 3))

    def test_row_counts_must_match(self):
        with pytest.raises(ValueError, match="labels has 1 rows"):
            InferenceResult(
                bboxes=torch.zeros(2, 4),
                labels=torch.zeros(1, dtype=torch.long),
            )

    def test_instance_masks_must_match_label_rows(self):
        with pytest.raises(ValueError, match="masks has 2 rows"):
            InferenceResult(
                labels=torch.zeros(1, dtype=torch.long),
                masks=torch.zeros(2, 4, 4, dtype=torch.bool),
            )

    def test_panoptic_segment_scores_must_match(self):
        with pytest.raises(ValueError, match="segment_ids has 2 entries"):
            InferenceResult(
                masks=torch.zeros(4, 4, 2, dtype=torch.long),
                segment_ids=torch.tensor([0, 1]),
                scores=torch.ones(3),
            )

    def test_classification_scores_must_match_labels(self):
        with pytest.raises(ValueError, match="labels has 2 rows"):
            InferenceResult(
                labels=torch.tensor([0, 1]),
                scores=torch.ones(1),
            )

    def test_validation_errors_are_framework_and_value_errors(self) -> None:
        """H3-1h: 校验异常双继承 AutoVisionError/ValueError 且 MRO 正确。"""
        assert issubclass(ResultValidationError, AutoVisionError)
        assert issubclass(ResultValidationError, ValueError)
        assert ResultValidationError.__mro__[:3] == (
            ResultValidationError,
            AutoVisionError,
            ValueError,
        )
        with pytest.raises(ResultValidationError, match="bboxes must have shape"):
            InferenceResult(bboxes=torch.zeros(2, 3))
        try:
            InferenceResult(bboxes=torch.zeros(2, 3))
        except ValueError as error:  # 既有 except ValueError 调用方不破坏
            assert isinstance(error, AutoVisionError)
        else:
            pytest.fail("ResultValidationError must stay catchable as ValueError")


class TestPanopticShapeHeuristic:
    """H3-1b: (H, W, 2) masks 缺 segment_ids 时的共用判定与构造期告警。"""

    def test_hw2_masks_without_segment_ids_are_not_instance_rows(self) -> None:
        """H3-1b: 480×640×2 无 segment_ids 不再误判为 480 个实例行。"""
        with pytest.warns(UserWarning, match="segment_ids"):
            result = InferenceResult(masks=torch.zeros(480, 640, 2))
        assert result.is_panoptic is True
        assert result.has_instance_masks is False
        assert len(result) == 1
        assert result.inferred_task == "panoptic_segmentation"

    def test_ambiguous_panoptic_shape_serializes_as_label_map_rle(self) -> None:
        """H3-1b: to_json 走 label-map RLE 组合而非 480 条实例 RLE。"""
        with pytest.warns(UserWarning):
            result = InferenceResult(masks=torch.zeros(480, 640, 2))
        payload = result.to_json()
        assert isinstance(payload["masks"], dict)
        assert payload["masks"]["size"] == [480, 640]
        assert set(payload["masks"]) == {"size", "class_ids", "segment_ids"}

    def test_segment_ids_present_keeps_the_legacy_path_warning_free(self) -> None:
        """H3-1b: 带 segment_ids 的 panoptic 路径行为逐位不变且不告警。"""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = InferenceResult(
                masks=torch.zeros(8, 8, 2, dtype=torch.long),
                segment_ids=torch.tensor([0, 1, 2]),
                scores=torch.tensor([0.9, 0.8, 0.7]),
            )
        assert result.is_panoptic is True
        assert len(result) == 3
        assert result.to_json()["masks"]["size"] == [8, 8]

    def test_degenerate_two_column_instance_stack_is_documented_ambiguity(self) -> None:
        """H3-1b: W==2 的真实例栈是启发式的已知歧义边界（取舍记录）。

        无 bboxes/labels/segment_ids 的 (N, H, 2) 栈与 panoptic (H, W, 2)
        图仅凭 shape 不可区分，按设计归 panoptic 侧（构造期 warning 指向
        segment_ids）；显式 bboxes 行契约在场时仍按实例栈处理，不回归。
        """
        with pytest.warns(UserWarning, match="segment_ids"):
            result = InferenceResult(masks=torch.zeros(3, 480, 2))
        assert result.is_panoptic is True
        assert result.has_instance_masks is False
        boxed = InferenceResult(
            bboxes=torch.zeros(3, 4),
            scores=torch.ones(3),
            labels=torch.zeros(3, dtype=torch.long),
            masks=torch.zeros(3, 480, 2, dtype=torch.bool),
        )
        assert boxed.has_instance_masks is True
