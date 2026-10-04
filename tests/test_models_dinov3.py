# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Tests for the transformers-gated DINOv3 package.

The installed transformers does not ship DINOv3 classes yet, so the package
logic is exercised by injecting a fake ``transformers`` module (replacing
``sys.modules["transformers"]``) that provides the expected DINOv3 classes.
When a future transformers version ships real DINOv3 support, the package
auto-enables against it.
"""

import hashlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from autovision._models import get_wrapped_model
from autovision._models.dinov3 import (
    DINOV3_PACKAGE,
    TransformersDinov3ConvNextWrapper,
    TransformersDinov3ViTWrapper,
)

# ── current environment: transformers without DINOv3 ────────────────────


class TestGating:
    def test_no_support_lists_empty(self):
        # The real installed transformers has no DINOv3 classes.
        assert DINOV3_PACKAGE.list_model_names() == []

    def test_no_support_raises_on_get_model(self):
        with pytest.raises(ImportError, match="DINOv3 support"):
            DINOV3_PACKAGE.get_model("vits16", load_weights=False)

    def test_graceful_without_transformers(self, monkeypatch):
        import importlib

        monkeypatch.delitem(sys.modules, "transformers", raising=False)

        def _raise(name, package=None):
            raise ImportError(name)

        monkeypatch.setattr(importlib, "import_module", _raise)
        assert DINOV3_PACKAGE.list_model_names() == []


# ── fake transformers with DINOv3 classes ───────────────────────────────


class FakeDinov3ViTConfig:
    def __init__(self, **kwargs) -> None:
        self.model_type = "dinov3_vit"
        self.num_channels = 3
        defaults = dict(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            image_size=224,
            patch_size=16,
            num_register_tokens=4,
        )
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)


class FakeDinov3ViTModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embeddings = SimpleNamespace(
            patch_embeddings=SimpleNamespace(
                projection=nn.Conv2d(
                    config.num_channels, config.hidden_size, config.patch_size, config.patch_size
                )
            )
        )

    def forward(self, pixel_values, **kwargs):
        patches = self.embeddings.patch_embeddings.projection(pixel_values)
        b, d, h, w = patches.shape
        seq = patches.flatten(2).transpose(1, 2)
        prefix = 1 + getattr(self.config, "num_register_tokens", 0)
        hidden = torch.cat(
            [torch.ones(b, prefix, d, device=seq.device), seq], dim=1
        )
        return SimpleNamespace(last_hidden_state=hidden)


class FakeDinov3ConvNextConfig:
    def __init__(self, **kwargs) -> None:
        self.model_type = "dinov3_convnext"
        self.num_channels = 3
        defaults = dict(
            hidden_sizes=[96, 192, 384, 768],
            image_size=224,
            patch_size=16,
        )
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)


class FakeDinov3ConvNextModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embeddings = SimpleNamespace(
            # Stem projects to the LAST stage channels: the wrapper's
            # feature_dim is hidden_sizes[-1].
            patch_embeddings=nn.Conv2d(
                config.num_channels, config.hidden_sizes[-1], config.patch_size, config.patch_size
            )
        )

    def forward(self, pixel_values, **kwargs):
        return SimpleNamespace(
            last_hidden_state=self.embeddings.patch_embeddings(pixel_values)
        )


@pytest.fixture
def fake_dinov3_transformers(monkeypatch):
    """Replace sys.modules['transformers'] with a stub exposing DINOv3."""
    fake = types.ModuleType("transformers")
    fake.Dinov3ViTConfig = FakeDinov3ViTConfig
    fake.Dinov3ViTModel = FakeDinov3ViTModel
    fake.Dinov3ConvNextConfig = FakeDinov3ConvNextConfig
    fake.Dinov3ConvNextModel = FakeDinov3ConvNextModel
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return fake


class TestWithInjectedSupport:
    def test_list_model_names(self, fake_dinov3_transformers):
        names = DINOV3_PACKAGE.list_model_names()
        assert len(names) == 24  # 12 models x (pretrained, -notpretrained)
        assert names == sorted(names)
        assert "dinov3/vits16" in names
        assert "dinov3/vits16-notpretrained" in names
        assert "dinov3/convnext-tiny" in names
        assert "dinov3/vitl16-sat493m" in names

    def test_registration_before_custom(self, fake_dinov3_transformers):
        from autovision._models.package_helpers import list_base_packages

        names = [p.name for p in list_base_packages()]
        assert names.index("dinov3") < names.index("custom")

    def test_vit_notpretrained_build(self, fake_dinov3_transformers):
        model = DINOV3_PACKAGE.get_model(
            "vits16-notpretrained", load_weights=False
        )
        assert isinstance(model, FakeDinov3ViTModel)
        assert model.config.num_register_tokens == 4
        wrapper = DINOV3_PACKAGE.get_model_wrapper(model)
        assert isinstance(wrapper, TransformersDinov3ViTWrapper)
        out = wrapper.forward_features(torch.randn(1, 3, 224, 224))
        assert out["features"].shape == (1, 384, 14, 14)
        assert out["cls_token"].shape == (1, 384)
        pooled = wrapper.forward_pool(out)
        assert pooled["pooled_features"].shape == (1, 384, 1, 1)
        assert wrapper.feature_dim() == 384
        assert wrapper.patch_size() == 16
        assert wrapper.architecture_info() == {
            "model_type": "transformer",
            "norm_type": "layernorm",
        }

    def test_convnext_notpretrained_build(self, fake_dinov3_transformers):
        model = DINOV3_PACKAGE.get_model(
            "convnext-tiny-notpretrained", load_weights=False
        )
        assert isinstance(model, FakeDinov3ConvNextModel)
        wrapper = DINOV3_PACKAGE.get_model_wrapper(model)
        assert isinstance(wrapper, TransformersDinov3ConvNextWrapper)
        out = wrapper.forward_features(torch.randn(1, 3, 224, 224))
        assert out["features"].shape == (1, 768, 14, 14)
        assert wrapper.feature_dim() == 768
        assert wrapper.architecture_info() == {
            "model_type": "convolutional",
            "norm_type": "layernorm",
        }

    def test_vit_input_channels(self, fake_dinov3_transformers):
        model = DINOV3_PACKAGE.get_model(
            "vits16-notpretrained", num_input_channels=2, load_weights=False
        )
        assert (
            model.embeddings.patch_embeddings.projection.in_channels == 2
        )
        wrapper = DINOV3_PACKAGE.get_model_wrapper(model)
        out = wrapper.forward_features(torch.randn(1, 2, 224, 224))
        assert out["features"].shape == (1, 384, 14, 14)

    def test_convnext_input_channels(self, fake_dinov3_transformers):
        model = DINOV3_PACKAGE.get_model(
            "convnext-tiny-notpretrained", num_input_channels=1, load_weights=False
        )
        assert model.embeddings.patch_embeddings.in_channels == 1
        wrapper = DINOV3_PACKAGE.get_model_wrapper(model)
        out = wrapper.forward_features(torch.randn(1, 1, 224, 224))
        assert out["features"].shape == (1, 768, 14, 14)

    def test_unknown_model(self, fake_dinov3_transformers):
        with pytest.raises(ValueError, match="Unknown dinov3 model"):
            DINOV3_PACKAGE.get_model("vitm16", load_weights=False)

    def test_pretrained_uses_from_pretrained(
        self, fake_dinov3_transformers, tmp_path, monkeypatch
    ):
        # The verified-loading flow: the repo id goes to the download layer,
        # and from_pretrained receives the verified local snapshot dir.
        requested = []
        downloaded = []

        class Spy(FakeDinov3ViTModel):
            @classmethod
            def from_pretrained(cls, name, **kwargs):
                requested.append(name)
                return cls(FakeDinov3ViTConfig(hidden_size=768))

        fake_dinov3_transformers.Dinov3ViTModel = Spy
        snapshot = tmp_path / "snap"
        snapshot.mkdir()
        (snapshot / "model.safetensors").write_bytes(b"weights")
        monkeypatch.setattr(
            "autovision._models.weight_verification._download_snapshot",
            lambda repo_id, revision, *a, **k: downloaded.append(repo_id)
            or snapshot,
        )
        monkeypatch.setattr(
            "autovision._models.weight_verification._lfs_sha256",
            lambda repo_id, revision: {
                "model.safetensors": hashlib.sha256(b"weights").hexdigest()
            },
        )
        DINOV3_PACKAGE.get_model("vitb16")  # pretrained by default
        assert downloaded == ["facebook/dinov3-vitb16-pretrain-lvd1689m"]
        assert requested == [snapshot]

    def test_is_supported_model(self, fake_dinov3_transformers):
        wrapper = DINOV3_PACKAGE.get_model_wrapper(
            DINOV3_PACKAGE.get_model("vits16-notpretrained", load_weights=False)
        )
        assert DINOV3_PACKAGE.is_supported_model(wrapper)
        by_name = get_wrapped_model("dinov3/vits16-notpretrained")
        assert isinstance(by_name, TransformersDinov3ViTWrapper)

    def test_export(self, tmp_path, fake_dinov3_transformers):
        wrapper = DINOV3_PACKAGE.get_model_wrapper(
            DINOV3_PACKAGE.get_model("vits16-notpretrained", load_weights=False)
        )
        out = tmp_path / "exported" / "model.pt"
        DINOV3_PACKAGE.export_model(wrapper, out, log_example=False)
        assert out.exists()
