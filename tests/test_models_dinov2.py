# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Tests for the transformers-backed DINOv2 package (real transformers)."""

import os

import pytest
import torch

from autovision._models import EmbeddingModel, get_wrapped_model
from autovision._models._model_helpers import load_state_dict_compat
from autovision._models.dinov2 import DINOV2_PACKAGE, TransformersDinov2Wrapper
from autovision._models.model_wrapper import ModelWrapper

transformers = pytest.importorskip(
    "transformers", reason="transformers package not installed"
)

# Real from_pretrained tests download from the HF endpoint (set
# HF_ENDPOINT=https://hf-mirror.com behind restricted networks) and are
# gated so the default offline CI run skips them.
_NETWORK_TESTS = bool(os.environ.get("AUTOVISION_HF_TESTS"))


def make_wrapper(name: str = "vits14-notpretrained", **kwargs) -> TransformersDinov2Wrapper:
    model = DINOV2_PACKAGE.get_model(name, load_weights=False, **kwargs)
    return DINOV2_PACKAGE.get_model_wrapper(model)


class TestLoadStateDictCompat:
    """Cross-version loading: old exports predate embeddings.mask_token."""

    def test_accepts_old_export_missing_mask_token(self):
        model = DINOV2_PACKAGE.get_model("vits14-notpretrained", load_weights=False)
        state = model.state_dict()
        old_export = {k: v for k, v in state.items() if "mask_token" not in k}
        assert "embeddings.mask_token" in state  # guard the premise
        load_state_dict_compat(model, old_export)  # must not raise

    def test_missing_real_key_still_raises(self):
        model = DINOV2_PACKAGE.get_model("vits14-notpretrained", load_weights=False)
        state = dict(model.state_dict())
        del state["embeddings.cls_token"]
        with pytest.raises(RuntimeError, match="missing key"):
            load_state_dict_compat(model, state)

    def test_unexpected_key_still_raises(self):
        model = DINOV2_PACKAGE.get_model("vits14-notpretrained", load_weights=False)
        state = dict(model.state_dict())
        state["embeddings.bogus_token"] = torch.zeros(1)
        with pytest.raises(RuntimeError, match="unexpected key"):
            load_state_dict_compat(model, state)


class TestListModelNames:
    def test_counts_and_names(self):
        names = DINOV2_PACKAGE.list_model_names()
        # 4 sizes x (pretrained, -noreg, -notpretrained)
        assert len(names) == 12
        assert names == sorted(names)
        for size in ("vits14", "vitb14", "vitl14", "vitg14"):
            assert f"dinov2/{size}" in names
            assert f"dinov2/{size}-noreg" in names
            assert f"dinov2/{size}-notpretrained" in names
        # No TiPSv2 (no transformers counterpart).
        assert not any("tips" in n for n in names)

    def test_registration_before_custom(self):
        from autovision._models.package_helpers import list_base_packages

        names = [p.name for p in list_base_packages()]
        assert names.index("dinov2") < names.index("custom")


class TestGetModel:
    def test_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown dinov2 model"):
            DINOV2_PACKAGE.get_model("vitm14", load_weights=False)

    def test_notpretrained_local_build(self):
        model = DINOV2_PACKAGE.get_model("vits14-notpretrained", load_weights=False)
        assert isinstance(model, transformers.Dinov2WithRegistersModel)
        config = model.config
        assert config.hidden_size == 384
        assert config.patch_size == 14
        assert config.num_register_tokens == 4

    def test_noreg_build(self):
        # load_weights=False must build locally (no from_pretrained call).
        model = DINOV2_PACKAGE.get_model("vits14-noreg", load_weights=False)
        assert isinstance(model, transformers.Dinov2Model)
        assert model.config.num_register_tokens == 0
        assert model.config.name_or_path == ""  # never touched the hub

    @pytest.mark.skipif(not _NETWORK_TESTS, reason="set AUTOVISION_HF_TESTS=1")
    def test_pretrained_with_registers(self):
        model = DINOV2_PACKAGE.get_model("vits14")
        assert isinstance(model, transformers.Dinov2WithRegistersModel)
        assert model.config.hidden_size == 384
        assert model.config.num_register_tokens == 4
        # Registers actually enter the token sequence.
        hidden_len = model(torch.randn(1, 3, 224, 224)).last_hidden_state.shape[1]
        assert hidden_len == 1 + 4 + (224 // 14) ** 2

    @pytest.mark.skipif(not _NETWORK_TESTS, reason="set AUTOVISION_HF_TESTS=1")
    def test_pretrained_noreg_checkpoint(self):
        model = DINOV2_PACKAGE.get_model("vits14-noreg")
        assert isinstance(model, transformers.Dinov2Model)
        assert model.config.name_or_path == "facebook/dinov2-small"
        # No register tokens: sequence is 1 CLS + patches only.
        hidden_len = model(torch.randn(1, 3, 224, 224)).last_hidden_state.shape[1]
        assert hidden_len == 1 + (224 // 14) ** 2

    def test_arbitrary_input_channels(self):
        model = DINOV2_PACKAGE.get_model(
            "vits14-notpretrained", num_input_channels=5, load_weights=False
        )
        projection = model.embeddings.patch_embeddings.projection
        assert projection.in_channels == 5
        wrapper = DINOV2_PACKAGE.get_model_wrapper(model)
        out = wrapper.forward_features(torch.randn(1, 5, 224, 224))
        assert out["features"].shape == (1, 384, 16, 16)


class TestWrapper:
    def test_is_model_wrapper(self):
        assert isinstance(make_wrapper(), ModelWrapper)

    def test_forward_shapes(self):
        wrapper = make_wrapper()
        out = wrapper.forward_features(torch.randn(2, 3, 224, 224))
        assert out["features"].shape == (2, 384, 16, 16)
        assert out["cls_token"].shape == (2, 384)
        pooled = wrapper.forward_pool(out)
        assert pooled["pooled_features"].shape == (2, 384, 1, 1)

    def test_pool_uses_cls_token(self):
        wrapper = make_wrapper()
        out = wrapper.forward_features(torch.randn(1, 3, 224, 224))
        pooled = wrapper.forward_pool(out)["pooled_features"]
        assert torch.allclose(pooled.flatten(1), out["cls_token"])

    def test_feature_dim_and_patch_size(self):
        wrapper = make_wrapper()
        assert wrapper.feature_dim() == 384
        assert wrapper.patch_size() == 14

    def test_architecture_info(self):
        assert make_wrapper().architecture_info() == {
            "model_type": "transformer",
            "norm_type": "layernorm",
        }

    def test_make_teacher(self):
        wrapper = make_wrapper()
        assert wrapper.make_teacher() is wrapper
        assert not wrapper.training
        assert all(
            m.p == 0.0 for m in wrapper.modules() if isinstance(m, torch.nn.Dropout)
        )

    def test_make_teacher_disables_stochastic_depth(self):
        # HF Dinov2DropPath subclasses nn.Module (attr drop_prob), not
        # nn.Dropout — make_teacher must zero it structurally so a later
        # train() call cannot silently re-enable stochastic depth.
        model = DINOV2_PACKAGE.get_model(
            "vits14-notpretrained", model_args={"drop_path_rate": 0.1},
            load_weights=False,
        )
        wrapper = DINOV2_PACKAGE.get_model_wrapper(model)
        drop_paths = [m for m in wrapper.modules() if hasattr(m, "drop_prob")]
        assert drop_paths and any(m.drop_prob > 0.0 for m in drop_paths)
        wrapper.make_teacher()
        assert all(m.drop_prob == 0.0 for m in drop_paths)

    def test_mask_token_enabled_by_default(self):
        # The package default enables the HF mask token (baseline DINOv2
        # architecture component; required by the iBOT method).
        for name in ("vits14-notpretrained", "vits14-noreg"):
            model = DINOV2_PACKAGE.get_model(name, load_weights=False)
            assert model.config.use_mask_token is True
            assert model.embeddings.mask_token is not None

    def test_forward_features_mask_requires_mask_token(self):
        # transformers silently ignores bool_masked_pos without the mask
        # token — fail loudly instead (package default enables the token;
        # users may explicitly disable it via model_args).
        wrapper = make_wrapper(model_args={"use_mask_token": False})
        mask = torch.zeros(1, 256, dtype=torch.bool)
        with pytest.raises(ValueError, match="use_mask_token"):
            wrapper.forward_features(torch.randn(1, 3, 224, 224), bool_masked_pos=mask)

    def test_forward_features_masked_positions(self):
        wrapper = make_wrapper()  # mask token enabled by default
        x = torch.randn(1, 3, 224, 224)
        mask = torch.zeros(1, 256, dtype=torch.bool)
        mask[:, :16] = True  # mask 16 of the 256 patches
        masked = wrapper.forward_features(x, bool_masked_pos=mask)
        assert masked["features"].shape == (1, 384, 16, 16)
        assert masked["cls_token"].shape == (1, 384)
        plain = wrapper.forward_features(x)
        # Masking actually reaches the token sequence.
        assert not torch.allclose(masked["features"], plain["features"])

    def test_multiscale_features_shapes(self):
        wrapper = make_wrapper()
        taps = wrapper.forward_multiscale_features(
            torch.randn(1, 3, 224, 224), (0, 5, 11)
        )
        assert len(taps) == 3
        for tap in taps:
            assert tap["features"].shape == (1, 384, 16, 16)
            assert tap["cls_token"].shape == (1, 384)

    def test_multiscale_final_tap_pre_layernorm(self):
        # Baseline get_intermediate_layers semantics: taps are block outputs
        # BEFORE the final layernorm, unlike forward_features.
        wrapper = make_wrapper()
        x = torch.randn(1, 3, 224, 224)
        final_tap = wrapper.forward_multiscale_features(x, (11,))[0]["features"]
        plain = wrapper.forward_features(x)["features"]
        assert not torch.allclose(final_tap, plain)

    def test_multiscale_accepts_iterator(self):
        # A generator must not be consumed by the validation loop and
        # silently yield zero taps.
        wrapper = make_wrapper()
        taps = wrapper.forward_multiscale_features(
            torch.randn(1, 3, 224, 224), iter((0, 5, 11))
        )
        assert len(taps) == 3

    def test_multiscale_rejects_out_of_range_indices(self):
        wrapper = make_wrapper()  # vits14: 12 blocks
        x = torch.randn(1, 3, 224, 224)
        with pytest.raises(ValueError, match="out of range"):
            wrapper.forward_multiscale_features(x, (12,))
        with pytest.raises(ValueError, match="out of range"):
            wrapper.forward_multiscale_features(x, (-1,))

    def test_embedding_model_roundtrip(self):
        wrapper = make_wrapper()
        embedding_model = EmbeddingModel(wrapped_model=wrapper, embed_dim=64)
        state = embedding_model.state_dict()
        embedding_model.load_state_dict(state, strict=True)
        out = embedding_model(torch.randn(1, 3, 224, 224), pool=True)
        assert out.shape == (1, 64, 1, 1)

    def test_get_wrapped_model_by_name(self):
        wrapper = get_wrapped_model("dinov2/vits14-notpretrained")
        assert isinstance(wrapper, TransformersDinov2Wrapper)

    def test_is_supported_model(self):
        wrapper = make_wrapper()
        assert DINOV2_PACKAGE.is_supported_model(wrapper)
        assert DINOV2_PACKAGE.is_supported_model(wrapper.hf_model)
        assert not DINOV2_PACKAGE.is_supported_model("dinov2/vits14")


class TestExport:
    def test_export_state_dict(self, tmp_path):
        wrapper = make_wrapper()
        out = tmp_path / "exported" / "model.pt"
        DINOV2_PACKAGE.export_model(wrapper, out, log_example=False)
        assert out.exists()
        state = torch.load(out, weights_only=True)
        assert any("patch_embeddings" in k for k in state)
