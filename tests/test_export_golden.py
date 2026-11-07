# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Export golden snapshots: ORT outputs pinned per task face (H4-5).

For every deployable task face a seeded tiny TaskNetwork variant is exported
through the real gated pipeline (``export_onnx(..., verify=True)`` — which
also runs the torch-vs-ORT parity gate), then a fixed seeded input is run
through the exported graph with onnxruntime CPU and the outputs are compared
against ``tests/golden/export_<face>.json`` at ``atol=1e-5``.

Structured parity gates compare torch-vs-ORT of the *same* weights, so they
cannot catch cross-version numeric drift. The golden pins the ORT outputs
themselves: decode/postprocess changes, opset bumps, simplification changes
and fp16-blacklist shifts that keep parity self-consistent still fail here.

Recording is fail-fast (H4-5): a missing golden is written and then *fails*
the run — the second run performs the comparison. Record-and-skip would let
a deleted golden pass CI on its first round.

Delete the corresponding ``tests/golden/export_<face>.json`` to re-record
after an *intentional* numeric change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch

onnxruntime = pytest.importorskip(
    "onnxruntime", reason="onnxruntime not installed (pip install -e '.[export]')"
)
pytest.importorskip("onnxslim", reason="onnxslim not installed (pip install -e '.[export]')")

GOLDEN_DIR = Path(__file__).parent / "golden"
_ATOL = 1e-5
_WEIGHT_SEED = 0
_INPUT_SEED = 2026


def _build_detection():
    from autovision.tasks.detection.detr.task_model import DETRObjectDetection

    return DETRObjectDetection(
        num_classes=2,
        model_name="gtrvit-s",
        image_size=(64, 64),
        decoder="dfine_official",
        num_queries=10,
        load_weights=False,
        backbone_pretrained=False,
    )


def _build_pose():
    from autovision.tasks.pose_estimation.gtr import GTRPoseEstimation

    return GTRPoseEstimation(variant="s", image_size=(64, 64), load_checkpoint=False)


def _build_depth():
    from autovision.tasks.depth_estimation.gtr import GTRDepthEstimation

    return GTRDepthEstimation(variant="s", image_size=(64, 64), load_checkpoint=False)


def _build_semseg():
    from autovision.tasks.semantic_segmentation.gtr import GTRSemanticSegmentation

    # 5 classes instead of the ADE20K default 19: the golden stores the full
    # (1, C, 64, 64) logit map, and C=19 would make the JSON >1MB for no
    # extra drift coverage (class count is pinned by shape assertions).
    return GTRSemanticSegmentation(
        num_classes=5, variant="s", image_size=(64, 64), load_checkpoint=False
    )


def _build_obb():
    from autovision.tasks.oriented_detection.gtr import GTRROBBOrientedDetection

    model = GTRROBBOrientedDetection(
        variant="s", image_size=(64, 64), num_queries=10, load_checkpoint=False
    )
    # Break the zero-init square-box degeneracy: at initialization every
    # decoded box is exactly square, and the long-edge normalization's
    # ``w < h`` comparison flips on ulp-level ORT differences (angle jumps
    # by exactly pi/2). Same workaround as the export gate suite.
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for module in model.modules():
            for name, child in module.named_children():
                if "bbox_head" in name or "pre_bbox_head" in name or "enc_bbox_head" in name:
                    for parameter in child.parameters():
                        parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.02)
    return model


def _build_classification():
    from autovision.tasks.classification.task_model import ImageClassification

    return ImageClassification(
        num_classes=2,
        backbone="torchvision/resnet18",
        image_size=(64, 64),
        load_weights=False,
    )


def _build_instance_segmentation():
    from autovision.tasks.instance_segmentation.dinov2_eomt.task_model import (
        Dinov2EomtInstanceSegmentation,
    )

    return Dinov2EomtInstanceSegmentation(
        num_classes=3,
        image_size=64,
        num_queries=6,
        num_blocks=2,
        num_upscale_blocks=1,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
    )


def _build_panoptic_segmentation():
    from autovision.tasks.panoptic_segmentation.dinov2_eomt.task_model import (
        Dinov2EomtPanopticSegmentation,
    )

    return Dinov2EomtPanopticSegmentation(
        num_classes=2,
        num_thing_classes=1,
        image_size=64,
        num_queries=6,
        num_blocks=2,
        num_upscale_blocks=1,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
    )


def _gtr_available() -> bool:
    from autovision.tasks.detection.shared.gtr import gtr_available

    return gtr_available()


# The GTR backbones need the optional fla dependency; CI installs the test
# extras without it, so only the non-GTR faces run there.
_GTR_SKIP = pytest.mark.skipif(
    not _gtr_available(),
    reason="GTR backbones require flash-linear-attention (pip install -e '.[gtr]')",
)

_BUILDERS: list[pytest.param] = [
    pytest.param(_build_detection, id="detection", marks=_GTR_SKIP),
    pytest.param(_build_obb, id="obb", marks=_GTR_SKIP),
    pytest.param(_build_pose, id="pose", marks=_GTR_SKIP),
    pytest.param(_build_depth, id="depth", marks=_GTR_SKIP),
    pytest.param(_build_semseg, id="semseg", marks=_GTR_SKIP),
    pytest.param(_build_classification, id="classification"),
    pytest.param(_build_instance_segmentation, id="instance_segmentation"),
    pytest.param(_build_panoptic_segmentation, id="panoptic_segmentation"),
]


@pytest.mark.parametrize(("build",), _BUILDERS)
def test_export_golden_snapshot(build: Callable[[], object], request, tmp_path):
    """H4-5: ORT outputs for one task face match the recorded golden.

    Missing golden → recorded, then this run fails; re-run verifies. Any
    output-shape, output-name/order, or value drift beyond ``atol=1e-5``
    fails the comparison.
    """
    face = request.node.callspec.id
    torch.manual_seed(_WEIGHT_SEED)
    model = build()
    model.eval()
    out_path = tmp_path / f"{face}.onnx"
    model.export_onnx(out_path, verify=True)

    generator = torch.Generator().manual_seed(_INPUT_SEED)
    images = torch.randn(1, 3, 64, 64, generator=generator)
    session = onnxruntime.InferenceSession(
        str(out_path), providers=["CPUExecutionProvider"]
    )
    output_names = [output.name for output in session.get_outputs()]
    arrays = session.run(None, {session.get_inputs()[0].name: images.numpy()})
    # Round to 6 decimals: rounding error (<=5e-7) stays far below the
    # comparison atol while keeping the JSON compact and diff-readable.
    snapshot = {
        name: np.asarray(array, dtype=np.float64).round(6).tolist()
        for name, array in zip(output_names, arrays)
    }

    golden = GOLDEN_DIR / f"export_{face}.json"
    if not golden.exists():
        golden.write_text(
            json.dumps(snapshot, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        pytest.fail(
            f"golden for '{face}' was missing; recorded {golden} — "
            "re-run the test to verify against it"
        )

    expected = json.loads(golden.read_text(encoding="utf-8"))
    assert list(expected) == list(snapshot), (
        f"output names/order changed for '{face}': "
        f"{list(expected)} -> {list(snapshot)}"
    )
    for name, stored in expected.items():
        actual = np.asarray(snapshot[name], dtype=np.float64)
        stored_array = np.asarray(stored, dtype=np.float64)
        assert stored_array.shape == actual.shape, (
            f"output '{name}' shape changed for '{face}': "
            f"{stored_array.shape} -> {actual.shape}"
        )
        np.testing.assert_allclose(
            actual,
            stored_array,
            atol=_ATOL,
            rtol=0,
            err_msg=f"numeric drift in output '{name}' for face '{face}'",
        )
