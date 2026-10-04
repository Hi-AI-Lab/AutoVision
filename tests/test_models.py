# Copyright (c) 2026 Autovision contributors. Apache-2.0.
import pytest
import torch
from torch import nn

from autovision._models import (
    EmbeddingModel,
    ModelWrapper,
    get_package,
    get_package_from_model,
    get_wrapped_model,
    list_model_names,
    missing_model_wrapper_attrs,
    parse_model_name,
    register_package,
)
from autovision._models.model_wrapper import ArchitectureInfoGettable
from autovision._models.package_helpers import list_base_packages, list_packages
from autovision.errors import UnknownModelError


class TinyWrapper(nn.Module):
    """Minimal ModelWrapper implementation used across tests."""

    def __init__(self, dim: int = 8, spatial: int = 2, model_type: str = "convolutional"):
        super().__init__()
        self.dim = dim
        self.spatial = spatial
        self._model_type = model_type
        self.conv = nn.Conv2d(3, dim, 3, padding=1)

    def forward_features(self, x):
        return {"features": self.conv(x), "cls_token": self.conv(x).mean(dim=(2, 3))}

    def forward_pool(self, x):
        pooled = torch.nn.functional.adaptive_avg_pool2d(x["features"], 1)
        return {"pooled_features": pooled}

    def feature_dim(self) -> int:
        return self.dim

    def get_model(self):
        return self.conv

    def architecture_info(self):
        return {"model_type": self._model_type, "norm_type": "batchnorm"}


class TestProtocols:
    def test_tiny_wrapper_is_model_wrapper(self):
        assert isinstance(TinyWrapper(), ModelWrapper)
        assert isinstance(TinyWrapper(), ArchitectureInfoGettable)

    def test_missing_attrs_diagnostic(self):
        class Incomplete(nn.Module):
            def forward_features(self, x): ...

        missing = missing_model_wrapper_attrs(Incomplete(), exclude_module_attrs=True)
        assert "feature_dim" in missing
        assert "forward_pool" in missing
        assert "get_model" in missing
        assert "forward_features" not in missing  # implemented

    def test_module_attrs_not_reported(self):
        assert missing_model_wrapper_attrs(TinyWrapper()) == []


class TestParseModelName:
    def test_simple(self):
        assert parse_model_name("torchvision/resnet18") == ("torchvision", "resnet18")

    def test_nested_name_keeps_rest(self):
        assert parse_model_name("pkg/a/b/c") == ("pkg", "a/b/c")

    def test_missing_slash_raises(self):
        with pytest.raises(ValueError, match="must have the format"):
            parse_model_name("resnet18")


class TestPackageRegistry:
    def test_torchvision_registered(self):
        names = [p.name for p in list_base_packages()]
        assert "torchvision" in names
        assert names[-1] == "custom"  # custom always last

    def test_get_package(self):
        package = get_package("torchvision")
        assert package.name == "torchvision"

    def test_get_unknown_package(self):
        with pytest.raises(ValueError, match="Unknown package"):
            get_package("nonexistent")

    def test_list_packages_excludes_custom(self):
        assert "custom" not in [p.name for p in list_packages()]

    def test_list_model_names_sorted(self):
        names = list_model_names()
        assert names == sorted(names)
        assert "torchvision/resnet18" in names
        assert "torchvision/convnext_tiny" in names

    def test_duplicate_registration_raises(self):
        from autovision._models.package import BasePackage

        class Duplicate(BasePackage):
            name = "torchvision"

            @classmethod
            def export_model(cls, model, out, log_example=True): ...

            @classmethod
            def is_supported_model(cls, model):
                return False

        with pytest.raises(ValueError, match="already registered"):
            register_package(Duplicate())


class TestTorchvisionPackage:
    def test_build_resnet18(self):
        wrapper = get_wrapped_model("torchvision/resnet18")
        assert isinstance(wrapper, ModelWrapper)
        model = wrapper.get_model()
        assert model.conv1.in_channels == 3

    def test_build_adapted_channels(self):
        wrapper = get_wrapped_model("torchvision/resnet18", num_input_channels=1)
        assert wrapper.get_model().conv1.in_channels == 1
        wrapper4 = get_wrapped_model("torchvision/resnet18", num_input_channels=5)
        assert wrapper4.get_model().conv1.in_channels == 5

    def test_forward_features_shapes(self):
        wrapper = get_wrapped_model("torchvision/resnet18")
        x = torch.randn(2, 3, 64, 64)
        out = wrapper.forward_features(x)
        assert out["features"].shape == (2, 512, 2, 2)
        pooled = wrapper.forward_pool(out)
        assert pooled["pooled_features"].shape == (2, 512, 1, 1)
        assert wrapper.feature_dim() == 512

    def test_architecture_info(self):
        wrapper = get_wrapped_model("torchvision/resnet18")
        info = wrapper.architecture_info()
        assert info == {"model_type": "convolutional", "norm_type": "batchnorm"}

    def test_convnext_wrapper(self):
        wrapper = get_wrapped_model("torchvision/convnext_tiny")
        assert wrapper.architecture_info() == {
            "model_type": "convolutional",
            "norm_type": "layernorm",
        }
        out = wrapper.forward_features(torch.randn(1, 3, 64, 64))
        assert out["features"].shape == (1, 768, 2, 2)

    def test_unsupported_model_name(self):
        with pytest.raises(ValueError):
            get_wrapped_model("torchvision/vit_h_14_does_not_exist_pattern")

    def test_export_model(self, tmp_path):
        wrapper = get_wrapped_model("torchvision/resnet18")
        out = tmp_path / "exported" / "model.pt"
        from autovision._models.torchvision.torchvision_package import (
            TORCHVISION_PACKAGE,
        )

        TORCHVISION_PACKAGE.export_model(wrapper, out)
        assert out.exists()
        state = torch.load(out, weights_only=True)
        assert "conv1.weight" in state


class TestWrappedModelResolution:
    def test_instance_passthrough(self):
        wrapper = TinyWrapper()
        assert get_wrapped_model(wrapper) is wrapper

    def test_nn_module_attributed(self):
        from autovision._models.torchvision.torchvision_package import (
            TORCHVISION_PACKAGE,
        )

        raw = TORCHVISION_PACKAGE.get_model("resnet18")
        package = get_package_from_model(raw, include_custom=False)
        assert package.name == "torchvision"

    def test_unknown_module_raises_with_diagnostic(self):
        with pytest.raises(UnknownModelError, match="Missing"):
            get_package_from_model(nn.Sequential(nn.Linear(2, 2)), include_custom=False, fallback_custom=False)

    def test_incomplete_wrapper_raises_with_missing_list(self):
        class Incomplete(nn.Module):
            pass

        with pytest.raises(UnknownModelError, match="implement the ModelWrapper"):
            get_wrapped_model(Incomplete())


class TestEmbeddingModel:
    def test_identity_head(self):
        wrapper = TinyWrapper(dim=8)
        embedding_model = EmbeddingModel(wrapped_model=wrapper)
        assert embedding_model.embed_dim == 8
        x = torch.randn(2, 3, 16, 16)
        pooled = embedding_model(x, pool=True)
        assert pooled.shape == (2, 8, 1, 1)
        features = embedding_model(x, pool=False)
        assert features.shape == (2, 8, 16, 16)

    def test_projection_head(self):
        wrapper = TinyWrapper(dim=8)
        embedding_model = EmbeddingModel(wrapped_model=wrapper, embed_dim=16)
        assert embedding_model.embed_dim == 16
        out = embedding_model(torch.randn(2, 3, 8, 8), pool=True)
        assert out.shape == (2, 16, 1, 1)
