# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Tests for the timm model package (optional extra)."""

from __future__ import annotations

import os

import pytest
import torch

timm = pytest.importorskip("timm", reason="timm not installed")

from autovision._models.package_helpers import get_package, get_wrapped_model  # noqa: E402
from autovision._models.timm import TIMM_PACKAGE  # noqa: E402
from autovision._models.timm.timm_wrapper import (  # noqa: E402
    TIMMModelWrapper,
    architecture_info_for_name,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


class TestPackage:
    def test_registered(self):
        assert get_package("timm") is TIMM_PACKAGE

    def test_list_model_names_nonempty(self):
        names = TIMM_PACKAGE.list_model_names()
        assert any(name.startswith("timm/") for name in names)

    def test_get_model_untrained_by_default(self):
        model = TIMM_PACKAGE.get_model("resnet18", load_weights=False)
        assert model.__module__.split(".")[0] == "timm"

    def test_get_model_nondefault_channels(self):
        model = TIMM_PACKAGE.get_model("resnet18", num_input_channels=5, load_weights=False)
        assert model.conv1.in_channels == 5

    def test_wrapper_roundtrip(self):
        model = TIMM_PACKAGE.get_model("resnet18", load_weights=False)
        wrapper = TIMM_PACKAGE.get_model_wrapper(model)
        assert isinstance(wrapper, TIMMModelWrapper)
        assert wrapper.get_model() is model
        assert model.num_features == wrapper.feature_dim()

    def test_via_public_resolver(self):
        wrapper = get_wrapped_model("timm/resnet18", load_weights=False)
        assert isinstance(wrapper, TIMMModelWrapper)

    def test_pretrained_requires_explicit_args(self):
        # Pretrained weights need an explicit opt-in; real downloads are
        # gated behind AUTOVISION_HF_TESTS=1 (see README) so restricted
        # networks don't hang the suite.
        if not os.environ.get("AUTOVISION_HF_TESTS"):
            pytest.skip("set AUTOVISION_HF_TESTS=1 to test real weight downloads")
        model = TIMM_PACKAGE.get_model("resnet18", model_args={"pretrained": True})
        assert model.num_features > 0

    def test_export_model(self, tmp_path):
        model = TIMM_PACKAGE.get_model("resnet18", load_weights=False)
        out = tmp_path / "timm_model.pt"
        TIMM_PACKAGE.export_model(model, out, log_example=False)
        assert out.exists()
        reference = TIMM_PACKAGE.get_model("resnet18", load_weights=False)
        reference.load_state_dict(torch.load(out, weights_only=True))


class TestWrapperProtocol:
    def _wrapper(self) -> TIMMModelWrapper:
        model = timm.create_model("resnet18", pretrained=False)
        return TIMMModelWrapper(model)

    def test_forward_features_nchw(self):
        wrapper = self._wrapper()
        output = wrapper.forward_features(torch.randn(2, 3, 32, 32))
        features = output["features"]
        assert features.ndim == 4
        assert features.shape[0] == 2
        assert features.shape[1] == wrapper.feature_dim()

    def test_forward_pool_shape(self):
        wrapper = self._wrapper()
        output = wrapper.forward_features(torch.randn(2, 3, 32, 32))
        pooled = wrapper.forward_pool(output)["pooled_features"]
        assert pooled.shape[:2] == (2, wrapper.feature_dim())

    def test_vit_features_drop_class_token(self):
        model = timm.create_model(
            "vit_tiny_patch16_224", pretrained=False, dynamic_img_size=True
        )
        wrapper = TIMMModelWrapper(model)
        output = wrapper.forward_features(torch.randn(1, 3, 32, 32))
        features = output["features"]
        assert features.ndim == 4  # tokens reshaped to a grid, class token gone
        grid_tokens = features.shape[2] * features.shape[3]
        sequence_tokens = 1 + (32 // 16) ** 2  # cls + patches
        assert grid_tokens == sequence_tokens - 1


class TestArchitectureInfo:
    def test_prefix_table(self):
        assert architecture_info_for_name("resnet18") == {
            "model_type": "convolutional", "norm_type": "batchnorm",
        }
        assert architecture_info_for_name("vit_base_patch16_224") == {
            "model_type": "transformer", "norm_type": "layernorm",
        }
        assert architecture_info_for_name("convnext_tiny") == {
            "model_type": "convolutional", "norm_type": "layernorm",
        }
        assert architecture_info_for_name("maxvit_tiny_224") == {
            "model_type": "hybrid", "norm_type": "layernorm",
        }

    def test_wrapper_architecture_info(self):
        model = timm.create_model("resnet18", pretrained=False)
        wrapper = TIMMModelWrapper(model)
        assert wrapper.architecture_info()["norm_type"] == "batchnorm"
