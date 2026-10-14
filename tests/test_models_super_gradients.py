# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Tests for the super_gradients model package (optional extra).

SuperGradients is a heavy, tightly pinned dependency; the tests here cover
the package skeleton without it (empty list, helpful errors) and the full
wrapper path when the package happens to be installed.
"""

from __future__ import annotations

import pytest

from autovision.encoders.package_helpers import get_package
from autovision.encoders.super_gradients import SUPER_GRADIENTS_PACKAGE
from autovision.encoders.super_gradients.super_gradients_package import (
    CustomizableDetectorModelWrapper,
    _load_customizable_detector_cls,
)
from autovision.errors import UnknownModelError

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

try:
    import super_gradients  # noqa: F401

    HAS_SG = True
except ImportError:
    HAS_SG = False


def test_registered():
    assert get_package("super_gradients") is SUPER_GRADIENTS_PACKAGE


@pytest.mark.skipif(HAS_SG, reason="super_gradients installed")
class TestSkeletonWithoutSuperGradients:
    def test_model_list_empty(self):
        assert SUPER_GRADIENTS_PACKAGE.list_model_names() == []

    def test_get_model_raises_with_guidance(self):
        with pytest.raises(ValueError, match="super-gradients"):
            SUPER_GRADIENTS_PACKAGE.get_model("yolo_nas_s")

    def test_unsupported_channel_count_checked_before_build(self):
        # Without the package this still raises — but the channel error is
        # only reachable with it installed; skip in that case.
        with pytest.raises((ValueError, ModuleNotFoundError)):
            SUPER_GRADIENTS_PACKAGE.get_model("yolo_nas_s", num_input_channels=5)

    def test_wrapper_rejects_unknown_model(self):
        class _Opaque:
            pass

        with pytest.raises(UnknownModelError):
            SUPER_GRADIENTS_PACKAGE.get_model_wrapper(_Opaque())


@pytest.mark.skipif(HAS_SG, reason="super_gradients installed")
class TestWrapperSkeletonWithoutSuperGradients:
    def test_supported_classes_empty(self):
        assert _load_customizable_detector_cls() == ()
        assert (
            CustomizableDetectorModelWrapper.is_supported_model_cls(object) is False
        )
