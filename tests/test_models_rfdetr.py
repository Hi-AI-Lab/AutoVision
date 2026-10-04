# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Tests for the transformers-backed RF-DETR package.

The package gates on the ``transformers`` RfDetr classes at runtime, so the
logic is exercised by injecting a fake ``transformers`` module (replacing
``sys.modules["transformers"]``) that provides the expected classes — same
pattern as the DINOv3 tests. The pretrained flow is covered with a spy
``from_pretrained`` against the verified-loading layer; real-weight
downloads stay behind ``AUTOVISION_HF_TESTS`` like the other HF packages.

Note: the DINOv2 backbone inside RF-DETR needs input sizes divisible by the
patch size; tests use 224 (divisible by both patch 14 and 16).
"""

import hashlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from autovision._models import get_wrapped_model
from autovision._models.model_wrapper import ModelWrapper
from autovision._models.rfdetr import RFDETR_PACKAGE, RFDETRModelWrapper

# ── current environment: no transformers / transformers without RfDetr ──


class TestGating:
    def test_no_support_lists_empty(self):
        # The real installed transformers has no RfDetr classes (or no
        # transformers at all).
        assert RFDETR_PACKAGE.list_model_names() == []

    def test_no_support_raises_on_get_model(self):
        with pytest.raises(ImportError, match="RfDetr"):
            RFDETR_PACKAGE.get_model("rf-detr-base", load_weights=False)

    def test_graceful_without_transformers(self, monkeypatch):
        import importlib

        monkeypatch.delitem(sys.modules, "transformers", raising=False)

        def _raise(name, package=None):
            raise ImportError(name)

        monkeypatch.setattr(importlib, "import_module", _raise)
        assert RFDETR_PACKAGE.list_model_names() == []


# ── fake transformers with the RfDetr classes ────────────────────────────


class FakeRfDetrDinov2Config:
    def __init__(self, **kwargs) -> None:
        self.model_type = "rf_detr_dinov2"
        self.num_channels = 3
        defaults = dict(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            image_size=224,
            patch_size=14,
            num_register_tokens=4,
            num_windows=4,
            window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
            out_indices=[2, 5, 8, 11],
            out_features=["stage2", "stage5", "stage8", "stage11"],
            reshape_hidden_states=True,
            use_mask_token=True,
        )
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)


class FakeRfDetrDinov2Backbone(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Conv2d(
            config.num_channels, config.hidden_size,
            config.patch_size, config.patch_size,
        )

    def forward(self, pixel_values, **kwargs):
        x = self.projection(pixel_values)
        # One feature map per out_features stage, all ViT-width (channels
        # are constant across stages for the reshaped DINOv2 backbone).
        return SimpleNamespace(
            feature_maps=[x.clone() for _ in self.config.out_features]
        )


class FakeRfDetrConfig:
    def __init__(self, backbone_config=None, **kwargs) -> None:
        self.model_type = "rf_detr"
        self.backbone_config = backbone_config or FakeRfDetrDinov2Config()
        self.decoder_layers = 3
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeRfDetrModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.backbone = FakeRfDetrDinov2Backbone(config.backbone_config)
        # Head-equivalent params beyond the backbone prove exports and
        # state dict round-trips stay full-model.
        self.head = nn.Linear(config.backbone_config.hidden_size, 4)


@pytest.fixture
def fake_rfdetr_transformers(monkeypatch):
    """Replace sys.modules['transformers'] with a stub exposing RfDetr."""
    fake = types.ModuleType("transformers")
    fake.RfDetrConfig = FakeRfDetrConfig
    fake.RfDetrModel = FakeRfDetrModel
    fake.RfDetrDinov2Config = FakeRfDetrDinov2Config
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return fake


def make_wrapper(
    name: str = "rf-detr-base-notpretrained",
    **kwargs,
) -> RFDETRModelWrapper:
    model = RFDETR_PACKAGE.get_model(name, load_weights=False, **kwargs)
    return RFDETR_PACKAGE.get_model_wrapper(model)


class TestWithInjectedSupport:
    def test_list_model_names(self, fake_rfdetr_transformers):
        names = RFDETR_PACKAGE.list_model_names()
        assert len(names) == 10  # 5 variants x (pretrained, -notpretrained)
        assert names == sorted(names)
        for expected in (
            "rfdetr/rf-detr-base",
            "rfdetr/rf-detr-base-notpretrained",
            "rfdetr/rf-detr-nano",
            "rfdetr/rf-detr-large-notpretrained",
        ):
            assert expected in names
        # The seg variant has no converted checkpoint; it stays unlisted.
        assert "rfdetr/rf-detr-seg-preview" not in names

    def test_registration_before_custom(self, fake_rfdetr_transformers):
        from autovision._models.package_helpers import list_base_packages

        names = [p.name for p in list_base_packages()]
        assert names.index("rfdetr") < names.index("custom")

    def test_base_notpretrained_build(self, fake_rfdetr_transformers):
        model = RFDETR_PACKAGE.get_model(
            "rf-detr-base-notpretrained", load_weights=False
        )
        assert isinstance(model, FakeRfDetrModel)
        backbone_config = model.config.backbone_config
        assert backbone_config.patch_size == 14
        assert backbone_config.num_windows == 4
        assert backbone_config.window_block_indexes == [0, 1, 3, 4, 6, 7, 9, 10]
        assert backbone_config.out_indices == [2, 5, 8, 11]
        assert backbone_config.use_mask_token is True
        assert model.config.decoder_layers == 3

    def test_variant_dispatch(self, fake_rfdetr_transformers):
        nano = RFDETR_PACKAGE.get_model(
            "rf-detr-nano-notpretrained", load_weights=False
        )
        assert nano.config.decoder_layers == 2
        assert nano.config.backbone_config.patch_size == 16
        assert nano.config.backbone_config.num_windows == 2
        assert nano.config.backbone_config.out_indices == [3, 6, 9, 12]
        large = RFDETR_PACKAGE.get_model(
            "rf-detr-large-notpretrained", load_weights=False
        )
        assert large.config.decoder_layers == 4

    def test_model_args_override(self, fake_rfdetr_transformers):
        model = RFDETR_PACKAGE.get_model(
            "rf-detr-nano-notpretrained",
            load_weights=False,
            model_args={"decoder_layers": 6},
        )
        assert model.config.decoder_layers == 6

    def test_three_channels_required(self, fake_rfdetr_transformers):
        with pytest.raises(ValueError, match="3 input channels"):
            RFDETR_PACKAGE.get_model("rf-detr-base", num_input_channels=1)

    def test_unknown_model(self, fake_rfdetr_transformers):
        with pytest.raises(ValueError, match="Unknown rfdetr model"):
            RFDETR_PACKAGE.get_model("rf-detr-unknown", load_weights=False)

    def test_wrapper_is_model_wrapper(self, fake_rfdetr_transformers):
        assert isinstance(make_wrapper(), ModelWrapper)

    def test_forward_features_last_stage(self, fake_rfdetr_transformers):
        wrapper = make_wrapper()
        out = wrapper.forward_features(torch.randn(2, 3, 224, 224))
        assert out["features"].shape == (2, 384, 16, 16)  # 224 / patch 14

    def test_feature_dim_and_patch_size(self, fake_rfdetr_transformers):
        wrapper = make_wrapper("rf-detr-nano-notpretrained")
        assert wrapper.feature_dim() == 384
        assert wrapper.patch_size() == 16

    def test_pool(self, fake_rfdetr_transformers):
        wrapper = make_wrapper()
        out = wrapper.forward_features(torch.randn(1, 3, 224, 224))
        pooled = wrapper.forward_pool(out)
        assert pooled["pooled_features"].shape == (1, 384, 1, 1)

    def test_architecture_info(self, fake_rfdetr_transformers):
        assert make_wrapper().architecture_info() == {
            "model_type": "transformer",
            "norm_type": "layernorm",
        }

    def test_state_dict_roundtrip_strict(self, fake_rfdetr_transformers):
        wrapper = make_wrapper()
        state = wrapper.state_dict()
        assert len(state) > 0
        wrapper.load_state_dict(state, strict=True)

    def test_is_supported_model(self, fake_rfdetr_transformers):
        wrapper = make_wrapper()
        assert RFDETR_PACKAGE.is_supported_model(wrapper)
        bare = RFDETR_PACKAGE.get_model(
            "rf-detr-base-notpretrained", load_weights=False
        )
        assert RFDETR_PACKAGE.is_supported_model(bare)
        assert not RFDETR_PACKAGE.is_supported_model(nn.Linear(2, 2))

    def test_get_wrapped_model_by_name(self, fake_rfdetr_transformers):
        wrapper = get_wrapped_model("rfdetr/rf-detr-base-notpretrained")
        assert isinstance(wrapper, RFDETRModelWrapper)

    def test_make_teacher_zeroes_stochastic_depth(
        self, fake_rfdetr_transformers
    ):
        wrapper = make_wrapper()
        droppath = nn.Dropout(0.1)
        droppath.drop_prob = 0.1
        wrapper.backbone.register_module("droppath", droppath)
        wrapper.make_teacher()
        assert not wrapper.training
        assert droppath.p == 0.0
        assert droppath.drop_prob == 0.0

    def test_export_full_state_dict(
        self, tmp_path, fake_rfdetr_transformers
    ):
        wrapper = make_wrapper()
        out = tmp_path / "exported" / "model.pt"
        RFDETR_PACKAGE.export_model(wrapper, out, log_example=False)
        payload = torch.load(out, weights_only=True)
        assert any(key.startswith("backbone.") for key in payload)
        assert any(key.startswith("head.") for key in payload)

    def test_pretrained_uses_from_pretrained(
        self, fake_rfdetr_transformers, tmp_path, monkeypatch
    ):
        # The verified-loading flow: the repo id goes to the download layer,
        # and from_pretrained receives the verified local snapshot dir.
        requested = []
        downloaded = []

        class Spy(FakeRfDetrModel):
            @classmethod
            def from_pretrained(cls, name, **kwargs):
                requested.append(name)
                return cls(FakeRfDetrConfig())

        fake_rfdetr_transformers.RfDetrModel = Spy
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
        RFDETR_PACKAGE.get_model("rf-detr-base")  # pretrained by default
        assert downloaded == ["stevenbucaille/rf-detr-base"]
        assert requested == [snapshot]
