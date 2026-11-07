# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Default-checkpoint name mapping for the GTR task faces.

Regression context: before the fix, ``checkpoint=None`` +
``load_checkpoint=True`` (the defaults) loaded NOTHING — a RANDOM model ran
with plausible-looking predictions. The fix resolves each face's pinned
asset by name; these tests pin that name mapping against the asset
registry WITHOUT the weights on disk (a drifted name must fail here as
"Unknown GTR asset" on every bare construction, not only on machines
holding the checkpoints).
"""

from __future__ import annotations

import importlib

import pytest

from autovision.registry.gtr_assets import GTR_ASSETS

_fla = True
try:
    from autovision.tasks.detection.shared.gtr import gtr_available as _gtr_available

    _fla = _gtr_available()
except Exception:  # noqa: BLE001 - fla missing entirely counts as unavailable
    _fla = False

requires_fla = pytest.mark.skipif(
    not _fla,
    reason="GTR task-face imports need flash-linear-attention (pip install -e '.[gtr]')",
)


@pytest.mark.parametrize(
    ("module_path", "class_name", "variant", "expected"),
    [
        pytest.param(
            "autovision.tasks.depth_estimation.gtr",
            "GTRDepthEstimation",
            "s",
            "gtrdepth_s",
            id="depth-s",
        ),
        pytest.param(
            "autovision.tasks.instance_segmentation.gtr",
            "GTRSegObjectDetection",
            "m",
            "gtrseg_m_coco",
            id="seg-m",
        ),
        pytest.param(
            "autovision.tasks.oriented_detection.gtr",
            "GTRROBBOrientedDetection",
            "x",
            "gtrobb_x_dota",
            id="obb-x",
        ),
        pytest.param(
            "autovision.tasks.pose_estimation.gtr",
            "GTRPoseEstimation",
            "l",
            "gtrpose_l_coco",
            id="pose-l",
        ),
        pytest.param(
            "autovision.tasks.semantic_segmentation.gtr",
            "GTRSemanticSegmentation",
            "s",
            "gtrsemseg_s_cityscapes",
            id="semseg-s",
        ),
    ],
)
@requires_fla
def test_default_checkpoint_name_is_registered(
    module_path: str, class_name: str, variant: str, expected: str
) -> None:
    """``default_checkpoint_name(variant)`` must hit a registered asset."""
    task_cls = getattr(importlib.import_module(module_path), class_name)

    assert task_cls.default_checkpoint_name(variant) == expected
    assert expected in GTR_ASSETS


@requires_fla
def test_unknown_variant_default_name_is_not_registered() -> None:
    """An invalid variant must not map onto a registered asset by accident
    (the faces validate variants in __init__; this guards the mapping only)."""
    from autovision.tasks.depth_estimation.gtr import GTRDepthEstimation

    assert GTRDepthEstimation.default_checkpoint_name("does-not-exist") not in GTR_ASSETS
