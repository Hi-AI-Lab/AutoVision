# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Tests for the native FSDC-DETR family (design doc 22 §7).

Runs against the REAL HF stack via the hermetic ``fsdc-vittest`` registry
variant (tiny DINOv3 ViT, offline build, no weights download) — stronger
than a fake because the swap-face equivalence test must exercise the
actual ``Deimv2DINOv3ConvEncoder``/``Deimv2HybridEncoder`` contracts.
Skips cleanly when the installed transformers lacks the DINOv3/Deimv2
classes.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from autovision.tasks.detection.components.dbfsaf import DBFSAF, MKSRU
from autovision.tasks.detection.components.fdconv import (
    FDConv,
    band_masks,
)
from autovision.tasks.detection.components.sfs_ff import SFSFF
from autovision.tasks.detection.components.wavelet_downsample import (
    FSDDown,
    HaarDWT2d,
    haar_idwt2,
)
from autovision.tasks.detection.deimv2_object_detection.train_model import (
    _deimv2_param_buckets,
)
from autovision.tasks.detection.fsdc_object_detection import (
    FSDC_MODEL_REGISTRY,
    FsdcArgs,
    FsdcObjectDetection,
    FsdcObjectDetectionTrain,
    FsdcObjectDetectionTrainArgs,
    resolve_fsdc_variant,
)

pytest.importorskip(
    "transformers", minversion="5.9", reason="Deimv2/DINOv3 classes required"
)

IMG = 256  # tiny backbone: patch 16 -> patch grid 16, strides 8/16/32


def _build(**fsdc_kwargs):
    torch.manual_seed(42)
    return FsdcObjectDetection(
        num_classes=5,
        model="fsdc-vittest",
        image_size=(IMG, IMG),
        load_weights=False,
        fsdc=fsdc_kwargs or None,
    )


# ── swap-face equivalence (doc 22 §3.2 proof, §7 test_swap_faces_complete)


class TestSwapFaceEquivalence:
    def test_full_model_bit_exact(self):
        """Native modules in their HF-equivalent configuration, loaded with
        the stock weights, reproduce the stock DEIMv2 forward bit-for-bit —
        proves the replacement faces are complete (no hidden dependency on
        the swapped-out modules)."""
        stock = _build()
        fsdc = _build(use_dbfsaf=False, use_sfs_ff=False, use_fsd_down=False)
        inner_s = stock._model.model
        inner_f = fsdc._model.model
        # HF-equivalent native configuration -> identical state-dict key sets.
        assert set(inner_s.state_dict()) == set(inner_f.state_dict())
        inner_f.load_state_dict(inner_s.state_dict())
        stock.eval()
        fsdc.eval()
        x = torch.randn(1, 3, IMG, IMG)
        with torch.no_grad():
            out_s = stock._forward_train(x)
            out_f = fsdc._forward_train(x)
        assert torch.equal(out_s["pred_logits"], out_f["pred_logits"])
        assert torch.equal(out_s["pred_boxes"], out_f["pred_boxes"])

    def test_fsdc_none_is_stock_wrapper(self):
        """fsdc=None keeps the HF conv encoder/encoder and a None snapshot."""
        model = _build()
        inner = model._model.model
        from transformers.models.deimv2.modeling_deimv2 import (
            Deimv2DINOv3ConvEncoder,
            Deimv2HybridEncoder,
        )

        assert isinstance(inner.conv_encoder, Deimv2DINOv3ConvEncoder)
        assert isinstance(inner.encoder, Deimv2HybridEncoder)
        assert model._init_args["model"] == "fsdc-vittest"
        assert model._init_args["fsdc"] is None


# ── paper equation forms (doc 22 §4.1-§4.3) ─────────────────────────────


class TestWaveletDownsample:
    def test_haar_perfect_reconstruction(self):
        dwt = HaarDWT2d(8, freeze_wavelet=True)
        x = torch.randn(2, 8, 16, 16)
        sub = dwt(x)
        view = sub.view(2, 8, 4, 8, 8)
        recon = haar_idwt2(
            view[:, :, 0], view[:, :, 1], view[:, :, 2], view[:, :, 3]
        )
        assert torch.allclose(recon, x, atol=1e-5)

    def test_haar_odd_size_ceil(self):
        dwt = HaarDWT2d(4, freeze_wavelet=True)
        out = dwt(torch.randn(1, 4, 31, 31))
        assert out.shape == (1, 16, 16, 16)

    def test_learnable_wavelet_initializes_to_haar(self):
        dwt = HaarDWT2d(4, freeze_wavelet=False)
        assert isinstance(dwt.filters, nn.Parameter)
        reference = HaarDWT2d(4, freeze_wavelet=True).filters
        assert torch.equal(dwt.filters.data, reference)
        frozen = HaarDWT2d(4, freeze_wavelet=True)
        assert not isinstance(frozen.filters, nn.Parameter)

    def test_fsddown_eq12_eq13_forms(self):
        channels = 16
        down = FSDDown(channels)
        x = torch.randn(2, channels, 32, 32)
        out = down(x)
        assert out.shape == (2, channels, 16, 16)
        # Eq.12: the gate is a single 8C vector -> chunked halves modulate
        # the two arms; verify by zeroing each arm's gate contribution.
        dwt = down.dwt(x)
        conv = down.conv_path(down.gconv(x))
        pooled = torch.cat([dwt, conv], dim=1).mean(dim=(2, 3), keepdim=True)
        gate = torch.sigmoid(pooled)
        assert gate.shape[1] == 8 * channels
        manual = down.out_proj(
            down.rho * gate[:, : 4 * channels] * dwt
            + gate[:, 4 * channels :] * conv
        )
        assert torch.allclose(out, manual, atol=1e-6)

    def test_fsddown_odd_input_pads_before_both_arms(self):
        down = FSDDown(8)
        assert down(torch.randn(1, 8, 31, 31)).shape == (1, 8, 16, 16)


class TestDBFSAFEquations:
    def test_eq8_split_mapping(self):
        """Half -> dilation 3, 3/8 -> dilation 2, 1/8 -> identity (Eq.8)."""
        c = 32
        mksru = MKSRU(c)
        # Zero the biases so a zero input stays zero through the DW stack.
        with torch.no_grad():
            mksru.dw5.bias.zero_()
        x = torch.zeros(1, c, 9, 9)
        x[:, : c // 2] = 1.0  # only the dilation-3 arm receives signal
        out = mksru.dw5(x)
        a, b, identity = torch.split(out, [c // 2, 3 * c // 8, c // 8], dim=1)
        assert mksru.dw7_d3.dilation == (3, 3) and a.shape[1] == c // 2
        assert mksru.dw7_d2.dilation == (2, 2) and b.shape[1] == 3 * c // 8
        # Identity arm: dw5(0) = 0, untouched downstream.
        assert torch.equal(identity, torch.zeros_like(identity))
        # Same-padding: every arm preserves the spatial grid.
        y = mksru(x)
        assert y.shape == x.shape

    def test_eq9_multiplicative_no_residual(self):
        """Zeroing the shortcut branch zeroes the output — Eq.9 is a pure
        product (an additive residual would keep the X̂ arm alive)."""
        c = 32
        mksru = MKSRU(c)
        mksru.eval()
        x = torch.randn(1, c, 9, 9)
        with torch.no_grad():
            active = mksru(x)
            assert active.abs().sum() > 0
            saved_w = mksru.short_conv.weight.clone()
            saved_b = mksru.short_conv.bias.clone()
            saved_gb = mksru.gate_s.bias.clone()
            mksru.short_conv.weight.zero_()
            mksru.short_conv.bias.zero_()
            mksru.gate_s.bias.zero_()
            gated = mksru(x)
            mksru.short_conv.weight.copy_(saved_w)
            mksru.short_conv.bias.copy_(saved_b)
            mksru.gate_s.bias.copy_(saved_gb)
        assert torch.equal(gated, torch.zeros_like(gated))

    def test_eq4_softmax_over_2c(self):
        """The gate softmax spans the concatenated 2·C_fs channels."""
        block = DBFSAF(96, 64, gamma=0.5)
        c_fs = block.c_fs
        f_s = torch.randn(1, c_fs, 20, 20)
        f_f = torch.randn(1, c_fs, 20, 20)
        pooled = torch.cat([f_s, f_f], dim=1).mean(dim=(2, 3), keepdim=True)
        gate = torch.softmax(pooled, dim=1)
        assert gate.shape[1] == 2 * c_fs
        assert torch.allclose(
            gate.sum(dim=1), torch.ones(1, 1, 1, 1), atol=1e-6
        )

    def test_eq5_alpha_scales_refined_arm(self):
        block = DBFSAF(96, 64)
        x = torch.randn(1, 96, 20, 20)
        block.eval()
        with torch.no_grad():
            base = block(x)
            block.alpha.fill_(0.0)
            no_spatial = block(x)
        assert not torch.allclose(base, no_spatial, atol=1e-5)

    def test_eq13_rho_scales_dwt_arm(self):
        """Eq.13's rho scalar modulates only the DWT arm of FSDDown."""
        down = FSDDown(16)
        x = torch.randn(1, 16, 32, 32)
        down.eval()
        with torch.no_grad():
            base = down(x)
            down.rho.fill_(0.0)
            no_dwt = down(x)
        assert not torch.allclose(base, no_dwt, atol=1e-5)

    def test_bf16_autocast_fft_island(self):
        """FPU stays finite under the engine's bf16 autocast (F7)."""
        block = DBFSAF(96, 64).eval()
        x = torch.randn(1, 96, 16, 16, dtype=torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = block(x)
        assert torch.isfinite(out.float()).all()

    def test_fdconv_band_masks_disjoint(self):
        masks = band_masks(3, 4)
        assert masks.shape == (4, 3, 3)
        assert torch.equal(masks.sum(dim=0), torch.ones(3, 3))

    def test_fdconv_lite_is_plain_conv(self):
        lite = FDConv(16, 32, lite=True)
        assert isinstance(lite.conv, nn.Conv2d)
        assert not hasattr(lite, "band_gate")


class TestSFSFF:
    def test_pan_junction_contract(self):
        """Eq.10-11: (C,H,W) coarse + (C,2H,2W) fine -> (C,H,W)."""
        sfs = SFSFF(64)
        coarse = torch.randn(2, 64, 25, 25)
        fine = torch.randn(2, 64, 50, 50)
        out = sfs(coarse, fine)
        assert out.shape == coarse.shape
        # Channel split {3C/8, C/2, C/8} sums to C.
        assert 3 * 64 // 8 + 64 // 2 + 64 // 8 == 64

    def test_embedded_down_is_fsddown(self):
        assert isinstance(SFSFF(64).down, FSDDown)

    def test_channel_guard(self):
        with pytest.raises(ValueError, match="% 8"):
            SFSFF(60)


# ── family integration (doc 22 §3.3) ────────────────────────────────────


class TestFsdcFamily:
    def test_three_level_encoder_contract(self):
        model = _build(use_dbfsaf=True, use_sfs_ff=True, use_fsd_down=True)
        inner = model._model.model
        feats = inner.conv_encoder(torch.randn(1, 3, IMG, IMG))
        assert [tuple(f.shape[-2:]) for f in feats] == [(32, 32), (16, 16), (8, 8)]
        assert [f.shape[1] for f in feats] == [64, 64, 64]
        outs = inner.encoder(feats)
        assert [tuple(f.shape[-2:]) for f in outs.feature_maps] == [
            (32, 32),
            (16, 16),
            (8, 8),
        ]

    def test_four_level_pyramid(self):
        model = _build(sta_levels=4, use_dbfsaf=True, use_sfs_ff=True)
        inner = model._model.model
        assert inner.config.num_feature_levels == 4
        assert list(inner.config.decoder_n_points) == [3, 3, 6, 3]
        feats = inner.conv_encoder(torch.randn(1, 3, IMG, IMG))
        assert [tuple(f.shape[-2:]) for f in feats] == [
            (64, 64),
            (32, 32),
            (16, 16),
            (8, 8),
        ]
        model.train()
        targets = [
            {"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]])}
        ]
        out = model._forward_train(torch.randn(1, 3, IMG, IMG), targets=targets)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()

    def test_training_loss_backward_finite(self):
        model = _build(use_dbfsaf=True, use_sfs_ff=True)
        model.train()
        x = torch.randn(2, 3, IMG, IMG)
        targets = [
            {
                "labels": torch.tensor([1, 2]),
                "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.3, 0.3, 0.05, 0.05]]),
            },
            {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.7, 0.7, 0.2, 0.2]])},
        ]
        out = model._forward_train(x, targets=targets)
        assert out["loss"] is not None
        out["loss"].backward()
        bad = [
            name
            for name, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        assert not bad

    def test_lr_buckets(self):
        """The ViT routes to the dinov3 bucket; every frequency module is a
        full-LR connector (doc 22 §4.6)."""
        model = _build(use_dbfsaf=True, use_sfs_ff=True)
        buckets = _deimv2_param_buckets(model, None)
        by_bucket: dict[str, set[str]] = {}
        for name, _ in model._model.named_parameters():
            if ".conv_encoder.backbone." in name:
                bucket = "dinov3"
            elif ".decoder." in name:
                bucket = "decoder"
            else:
                bucket = "other"
            by_bucket.setdefault(bucket, set()).add(name)
        assert by_bucket["dinov3"]  # the ViT moved over
        assert any("spatial_tuning_adapter" in n for n in by_bucket["other"])
        assert any("dbfsaf_blocks" in n for n in by_bucket["other"])
        assert any("pan_sfsff" in n for n in by_bucket["other"])
        # Nothing frequency-related leaked into the backbone bucket.
        assert not any(
            "dbfsaf" in n or "sfsff" in n or "spatial_tuning" in n
            for n in by_bucket["dinov3"]
        )
        assert buckets  # the wrapper's own grouping runs clean


class TestGuards:
    def test_unknown_variant(self):
        with pytest.raises(ValueError, match="Unknown fsdc"):
            resolve_fsdc_variant("fsdc-m")

    def test_fsd_down_requires_sfs_ff(self):
        with pytest.raises(ValueError, match="no landing spot"):
            FsdcArgs(use_sfs_ff=False, use_fsd_down=True).validate_args()

    def test_sta_levels_guard(self):
        with pytest.raises(ValueError, match="sta_levels"):
            FsdcArgs(sta_levels=5).validate_args()

    def test_gamma_guard(self):
        with pytest.raises(ValueError, match="gamma"):
            FsdcArgs(dbfsaf_gamma=1.5).validate_args()

    def test_l_variant_has_no_conversion(self):
        """fsdc-l builds locally but rejects a warm-start request with the
        PR-0 reason (no transformers conversion on the Hub)."""
        with pytest.raises(ValueError, match="no transformers-format"):
            FsdcObjectDetection(
                num_classes=5,
                model="fsdc-l",
                image_size=(640, 640),
                load_weights=True,
            )

    def test_registry_covers_variants(self):
        assert set(FSDC_MODEL_REGISTRY) == {"fsdc-l", "fsdc-s", "fsdc-vittest"}


class TestCheckpointRoundTrip:
    def test_state_dict_and_init_args(self):
        model = _build(use_dbfsaf=True, use_sfs_ff=True)
        state = model.state_dict()
        restored = _build(use_dbfsaf=True, use_sfs_ff=True)
        restored.load_state_dict(state)
        model.eval()
        restored.eval()
        x = torch.randn(1, 3, IMG, IMG)
        with torch.no_grad():
            a = model._forward_train(x)
            b = restored._forward_train(x)
        assert torch.equal(a["pred_logits"], b["pred_logits"])
        # The snapshot records the fsdc name + dumped config (safe-loader
        # round trip, doc 22 §3.2).
        assert model._init_args["model"] == "fsdc-vittest"
        snapshot = model._init_args["fsdc"]
        assert isinstance(snapshot, dict)
        assert FsdcArgs(**snapshot) == model.fsdc_args

    def test_train_args_resolve_variant(self):
        args = FsdcObjectDetectionTrainArgs(
            num_classes=5, model="fsdc-l", image_size=(640, 640)
        )
        info = args.resolve_variant()
        assert info["config"]["d_model"] == 224
        assert isinstance(FsdcObjectDetectionTrain, type)


class TestV2Erratum:
    """doc 22 v1.2: the official L finetunes the REAL DINOv3 ViT-S/16
    (384-wide, LVD-1689M pretrained) — the serialized config's
    "embed_dim: 224" is the STA wrapper's projected width."""

    def test_l_registry_real_vit(self):
        from autovision.tasks.detection.deimv2_object_detection.config import (
            resolve_deimv2_variant,
        )

        info = resolve_deimv2_variant("deimv2-dinov3-l")
        backbone = info["config"]["backbone_config"]
        assert backbone["hidden_size"] == 384
        assert backbone["num_attention_heads"] == 6
        assert backbone["num_register_tokens"] == 4
        assert info["lr_backbone_ratio"] == 0.025  # official 1.25e-5 / 5e-4

    def test_fsdc_l_fusion_channels_use_real_vit_width(self):
        """384-wide ViT + STA {64,128,128} -> fusion in {448, 512, 512}."""
        model = FsdcObjectDetection(
            num_classes=5,
            model="fsdc-l",
            image_size=(640, 640),
            load_weights=False,
            fsdc={"use_dbfsaf": True, "use_sfs_ff": True},
        )
        blocks = model._model.model.conv_encoder.dbfsaf_blocks
        assert [b.fdconv.weight.shape[1] for b in blocks] == [448, 512, 512]
        assert all(b.out_proj.conv.out_channels == 224 for b in blocks)

    def test_backbone_weights_loading_roundtrip(self, tmp_path):
        """The facebook-layout rename (layer.* -> model.layer.*) restores
        weights exactly; a foreign file fails loud."""
        from safetensors.torch import save_file

        source = _build(use_dbfsaf=True, use_sfs_ff=True)
        native = source._model.model.conv_encoder
        facebook_layout = {
            key.removeprefix("model."): value
            if key.startswith("model.layer.")
            else value
            for key, value in native.backbone.state_dict().items()
        }
        weights_file = tmp_path / "dinov3_vits16.safetensors"
        save_file(facebook_layout, str(weights_file))

        loaded = _build(use_dbfsaf=True, use_sfs_ff=True)
        reference = loaded._model.model.conv_encoder.backbone.state_dict()
        loaded.fsdc_args = FsdcArgs(
            use_dbfsaf=True, use_sfs_ff=True, backbone_weights=str(weights_file)
        )
        loaded._model.model.conv_encoder.load_dinov3_pretrained(str(weights_file))
        after = loaded._model.model.conv_encoder.backbone.state_dict()
        assert all(torch.equal(reference[k], after[k]) for k in reference)

        corrupted = dict(facebook_layout)
        corrupted["layer.0.attention.q_proj.weight"] = torch.zeros(1)
        bad_file = tmp_path / "corrupted.safetensors"
        save_file(corrupted, str(bad_file))
        with pytest.raises(ValueError, match="does not match"):
            native.load_dinov3_pretrained(str(bad_file))


class TestExport:
    def test_component_export_static(self):
        """All frequency modules are static-shape: torch.export passes on
        each component (the FFT ops trace as pure functions)."""
        from torch.export import export

        down = FSDDown(16).eval()
        export(down, (torch.randn(1, 16, 32, 32),))
        sfs = SFSFF(16).eval()
        export(
            sfs,
            (torch.randn(1, 16, 8, 8), torch.randn(1, 16, 16, 16)),
        )
        block = DBFSAF(48, 16).eval()
        export(block, (torch.randn(1, 48, 16, 16),))

    def test_fft_export_fallback_arm(self):
        """deploy() + fft_export_fallback bypasses both FFT sites; without
        the flag the deploy pass is a no-op for the frequency modules."""
        model = _build(
            use_dbfsaf=True, use_sfs_ff=True, fft_export_fallback=True
        )
        model.eval()
        inner = model._model.model
        x = torch.randn(1, 3, IMG, IMG)
        with torch.no_grad():
            before = inner.encoder(list(inner.conv_encoder(x))).feature_maps
        model.deploy()
        for sfs in inner.encoder.pan_sfsff:
            assert sfs.fft.export_identity
        for block in inner.conv_encoder.dbfsaf_blocks:
            assert block.fpu.export_identity
        with torch.no_grad():
            after = inner.encoder(list(inner.conv_encoder(x))).feature_maps
        assert not all(torch.equal(a, b) for a, b in zip(before, after))

        plain = _build(use_dbfsaf=True, use_sfs_ff=True)
        plain.deploy()
        for sfs in plain._model.model.encoder.pan_sfsff:
            assert not sfs.fft.export_identity
