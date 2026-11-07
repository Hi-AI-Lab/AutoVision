# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""``decode_detection_outputs`` must accept every raw-output shape.

The structured object (``.logits`` / ``.boxes``), the
``(logits, boxes)`` tuple, and the flat ``list`` the compiled runners
(OpenVINO / TensorRT) return must all decode identically.

Regression: the OpenVINO backend returns a ``list``, which previously crashed
the postprocess with
``AttributeError: 'list' object has no attribute 'logits'``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from autovision.tasks.detection.shared.decode import decode_detection_outputs


def _make() -> tuple[torch.Tensor, torch.Tensor, list[dict[str, int]]]:
    torch.manual_seed(0)
    logits = torch.randn(1, 10, 3)  # (B, Q, C) raw class scores
    boxes = torch.rand(1, 10, 4)  # (B, Q, 4) normalized cxcywh
    metadata = [{"orig_w": 100, "orig_h": 50}]
    return logits, boxes, metadata


def test_object_tuple_and_list_decode_identically() -> None:
    logits, boxes, metadata = _make()
    reference = decode_detection_outputs(
        SimpleNamespace(logits=logits, boxes=boxes), metadata, threshold=0.0
    )
    via_tuple = decode_detection_outputs(
        (logits, boxes), metadata, threshold=0.0
    )
    via_list = decode_detection_outputs([logits, boxes], metadata, threshold=0.0)

    for got in (via_tuple, via_list):
        assert got[0]["labels"].equal(reference[0]["labels"])
        assert torch.allclose(got[0]["bboxes"], reference[0]["bboxes"])
        assert torch.allclose(got[0]["scores"], reference[0]["scores"])


def test_list_order_is_logits_then_boxes() -> None:
    # The compiled runners return [logits, boxes] (the export output-port
    # order). Index 0 is consumed as logits, so swapping the two changes the
    # decoded boxes — proving the documented contract is what decode consumes.
    # Both tensors are 4-wide so the swapped run still flows through decode
    # (a 3-wide logits would make the "boxes" slot unpackable).
    torch.manual_seed(1)
    logits = torch.randn(1, 10, 4)
    boxes = torch.rand(1, 10, 4)
    metadata = [{"orig_w": 100, "orig_h": 50}]
    correct = decode_detection_outputs([logits, boxes], metadata, threshold=0.0)
    swapped = decode_detection_outputs([boxes, logits], metadata, threshold=0.0)
    assert not torch.allclose(correct[0]["bboxes"], swapped[0]["bboxes"])
