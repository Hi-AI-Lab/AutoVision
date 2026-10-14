# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Tests for the DEIMv2 object detection task model.

The real detector needs transformers >= 5.9 (Deimv2 classes), so the logic
runs against an injected fake ``transformers`` module that mirrors the HF
API surface the wrapper uses (forward with labels returning
loss/loss_dict/logits/pred_boxes, class_embed module list + encoder score
head) — same pattern as tests/test_rfdetr_object_detection.py.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from autovision.tasks.detection.deimv2_object_detection.config import (
    DEIMV2_MODEL_REGISTRY,
    Deimv2ObjectDetectionTrainArgs,
    resolve_deimv2_variant,
)
from autovision.tasks.detection.deimv2_object_detection.task_model import (
    Deimv2ObjectDetection,
    _reinitialize_class_head,
)
from autovision.tasks.detection.deimv2_object_detection.train_model import (
    Deimv2ObjectDetectionTrain,
    _deimv2_param_buckets,
)

# ── fake transformers mirroring the HF DEIMv2 API ────────────────────────


class FakeDeimv2Config:
    def __init__(self, backbone_config=None, **kwargs) -> None:
        self.model_type = "deimv2"
        self.backbone_config = backbone_config or {
            "model_type": "hgnet_v2",
            "out_indices": [3, 4],
        }
        defaults = dict(
            d_model=32,
            num_queries=16,
            decoder_layers=2,
            num_labels=80,
            encoder_type="hybrid",
        )
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)


class _FakeEncoderBlock(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv2d(width, width, 3, padding=1)


class _FakeDinov3Backbone(nn.Module):
    """Attribute tree mirroring the real DINOv3 names for bucketing tests."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Linear(8, 8)
        self.layernorm = nn.LayerNorm(8)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_FakeEncoderBlock() for _ in range(2)])


class _FakeDeimv2Inner(nn.Module):
    def __init__(self, config: FakeDeimv2Config) -> None:
        super().__init__()
        self.conv_encoder = nn.Module()
        self.conv_encoder.backbone = _FakeDinov3Backbone()
        # HGNetv2-style subtree: exercises the ".conv_encoder.model." bucket
        # fragment and the BatchNorm ("normalization") no-wd naming.
        self.conv_encoder.model = nn.Module()
        self.conv_encoder.model.embedder = nn.Module()
        self.conv_encoder.model.embedder.normalization = nn.BatchNorm2d(8)
        self.encoder = nn.Linear(32, 32)
        self.decoder = nn.Linear(32, 32)
        self.enc_score_head = nn.Linear(32, config.num_labels)
        self.class_embed = nn.ModuleList(
            [nn.Linear(32, config.num_labels) for _ in range(config.decoder_layers)]
        )


class FakeDeimv2ForObjectDetection(nn.Module):
    def __init__(self, config: FakeDeimv2Config) -> None:
        super().__init__()
        self.config = config
        self.model = _FakeDeimv2Inner(config)
        self.proj = nn.Conv2d(3, 32, 7, stride=7)
        self.class_embed = self.model.class_embed
        self.bbox_embed = nn.Linear(32, 4)

    def forward(self, pixel_values, pixel_mask=None, labels=None, **kwargs):
        features = self.proj(pixel_values)
        hidden = torch.nn.functional.adaptive_avg_pool2d(features, (4, 4))
        hidden = hidden.flatten(2).transpose(1, 2)  # (B, 16, 32)
        logits = self.class_embed[-1](hidden)
        boxes = torch.sigmoid(self.bbox_embed(hidden))
        if labels is None:
            return {
                "logits": logits,
                "pred_boxes": boxes,
                "loss": None,
                "loss_dict": {},
            }
        ce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits)
        )
        l1 = torch.nn.functional.l1_loss(boxes, torch.zeros_like(boxes))
        return {
            "logits": logits,
            "pred_boxes": boxes,
            "loss": ce + l1,
            "loss_dict": {"loss_ce": ce.detach(), "loss_bbox": l1.detach()},
        }


@pytest.fixture
def fake_deimv2_transformers(monkeypatch):
    fake = types.ModuleType("transformers")
    fake.Deimv2Config = FakeDeimv2Config
    fake.Deimv2ForObjectDetection = FakeDeimv2ForObjectDetection
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return fake


def make_task_model(num_classes: int = 3, **kwargs) -> Deimv2ObjectDetection:
    return Deimv2ObjectDetection(
        num_classes=num_classes,
        model="deimv2-hgnetv2-n",
        image_size=(224, 224),
        num_select=16,
        load_weights=False,
        **kwargs,
    )


def make_batch(batch_size: int = 2, image_size: int = 224) -> dict:
    return {
        "image": torch.randn(batch_size, 3, image_size, image_size),
        "bboxes": [torch.rand(3, 4) for _ in range(batch_size)],
        "classes": [torch.tensor([0, 1, 2]) for _ in range(batch_size)],
        "original_size": [(100, 150) for _ in range(batch_size)],
    }


# ── registry / config ────────────────────────────────────────────────────


class TestConfig:
    def test_unknown_variant(self):
        with pytest.raises(ValueError, match="Unknown deimv2 detection model"):
            resolve_deimv2_variant("deimv2-dinov3-x")

    def test_registry_variants(self):
        # deimv2-dinov3-l: design doc 22 PR-0 (official Intellindust config,
        # no transformers conversion); fsdc-vittest: hermetic FSDC build.
        assert set(DEIMV2_MODEL_REGISTRY) == {
            "deimv2-dinov3-s",
            "deimv2-dinov3-l",
            "deimv2-hgnetv2-n",
            "deimv2-hgnetv2-pico",
            "fsdc-vittest",
        }

    def test_auto_image_size_resolves_to_native(self):
        args = Deimv2ObjectDetectionTrainArgs(
            model="deimv2-hgnetv2-n", image_size="auto"
        )
        args.resolve_image_size()
        assert args.image_size == (640, 640)

    def test_custom_size_divisibility(self):
        args = Deimv2ObjectDetectionTrainArgs(
            model="deimv2-hgnetv2-n", image_size=(800, 800)
        )
        args.resolve_image_size()
        assert args.image_size == (800, 800)

        bad = Deimv2ObjectDetectionTrainArgs(
            model="deimv2-hgnetv2-n", image_size=(810, 810)
        )
        with pytest.raises(ValueError, match="not divisible"):
            bad.resolve_image_size()

    def test_resolve_auto_fills_variant_recipe(self):
        args = Deimv2ObjectDetectionTrainArgs(model="deimv2-hgnetv2-n")
        args.resolve_auto(
            total_steps=1000, train_num_batches=100, gradient_accumulation_steps=1
        )
        assert args.lr == pytest.approx(8e-4)
        assert args.lr_backbone == pytest.approx(8e-4 * 0.5)
        assert args.weight_decay == pytest.approx(1e-4)
        assert args.lr_scheduler == "cosine"
        assert args.lr_warmup_steps == 100
        assert args.image_size == (640, 640)

    def test_resolve_auto_dinov3_backbone_lr(self):
        args = Deimv2ObjectDetectionTrainArgs(model="deimv2-dinov3-s")
        args.resolve_auto(
            total_steps=1000, train_num_batches=100, gradient_accumulation_steps=1
        )
        assert args.lr == pytest.approx(5e-4)
        assert args.lr_backbone == pytest.approx(5e-4 * 0.05)

    def test_explicit_overrides_win(self):
        args = Deimv2ObjectDetectionTrainArgs(
            model="deimv2-hgnetv2-n", lr=1e-4, weight_decay=2e-4
        )
        args.resolve_auto(
            total_steps=1000, train_num_batches=100, gradient_accumulation_steps=1
        )
        assert args.lr == pytest.approx(1e-4)
        # lr_backbone still tracks the variant ratio from the explicit lr.
        assert args.lr_backbone == pytest.approx(1e-4 * 0.5)
        assert args.weight_decay == pytest.approx(2e-4)


# ── gating ───────────────────────────────────────────────────────────────


class TestGating:
    def test_no_support_raises_informative(self, monkeypatch):
        from autovision.tasks.detection.deimv2_object_detection import task_model

        monkeypatch.setattr(task_model, "_deimv2_classes", lambda: None)
        with pytest.raises(ImportError, match="transformers"):
            make_task_model()


# ── task model ───────────────────────────────────────────────────────────


class TestTaskModel:
    def test_forward_shapes(self, fake_deimv2_transformers):
        model = make_task_model(num_classes=3)
        model.eval()
        with torch.no_grad():
            logits, boxes = model(torch.randn(2, 3, 224, 224))
        assert logits.shape == (2, 16, 3)
        assert boxes.shape == (2, 16, 4)
        assert (boxes >= 0).all() and (boxes <= 1).all()

    def test_loss_backward(self, fake_deimv2_transformers):
        model = make_task_model(num_classes=3)
        batch = make_batch()
        targets = [
            {"boxes": batch["bboxes"][i], "labels": batch["classes"][i]}
            for i in range(2)
        ]
        out = model._forward_train(batch["image"], targets=targets)
        assert out["loss"] is not None
        assert torch.isfinite(out["loss"])
        out["loss"].backward()

    def test_postprocess_decodes_to_original_pixels(self, fake_deimv2_transformers):
        model = make_task_model(num_classes=3)
        model.eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 224, 224))
        decoded = model.postprocess(outputs, [{"orig_w": 150, "orig_h": 100}])
        assert len(decoded) == 1
        prediction = decoded[0]
        assert set(prediction) == {"labels", "bboxes", "scores"}
        assert prediction["bboxes"].shape[1] == 4
        assert prediction["bboxes"][:, 0].max() <= 150
        assert prediction["bboxes"][:, 1].max() <= 100

    def test_class_head_retile(self, fake_deimv2_transformers):
        model = make_task_model(num_classes=80)  # native head: 80 rows
        # The fake's per-layer heads are independently initialized (the real
        # checkpoint ties them), so compare each head against its own rows.
        old_weights = [
            head.weight.detach().clone() for head in model._model.class_embed
        ]
        _reinitialize_class_head(model._model, 3)  # shrink: truncation
        for index, head in enumerate(model._model.class_embed):
            assert head.weight.shape == (3, 32)
            torch.testing.assert_close(head.weight, old_weights[index][:3])
        assert model._model.model.enc_score_head.weight.shape == (3, 32)
        # Grow 3 -> 7: cyclic tiling of the surviving (shrunk) rows.
        shrunk = [head.weight.detach().clone() for head in model._model.class_embed]
        _reinitialize_class_head(model._model, 7)
        for index, head in enumerate(model._model.class_embed):
            assert head.weight.shape == (7, 32)
            torch.testing.assert_close(head.weight, shrunk[index][torch.arange(7) % 3])
        assert model._model.model.enc_score_head.weight.shape == (7, 32)

    def test_local_build_sets_eval_size(self, fake_deimv2_transformers):
        model = make_task_model()
        assert model._model.config.eval_size == [224, 224]
        # The backbone dict is preserved for the config (routing key intact).
        assert model._model.config.backbone_config["model_type"] == "hgnet_v2"
        # Official freeze_norm: False — backbone BN must stay trainable.
        assert model._model.config.freeze_backbone_batch_norms is False


# ── train model ──────────────────────────────────────────────────────────


def make_train_args(**overrides) -> Deimv2ObjectDetectionTrainArgs:
    defaults = dict(
        model="deimv2-hgnetv2-n",
        num_classes=3,
        image_size=(224, 224),
        num_select=16,
        load_weights=False,
        use_ema_model=True,
        ema_warmup_steps=2,
        eval_max_dets=10,
    )
    defaults.update(overrides)
    return Deimv2ObjectDetectionTrainArgs(**defaults)


class TestParamBuckets:
    def test_buckets_dinov3_naming(self, fake_deimv2_transformers):
        train_model = Deimv2ObjectDetectionTrain(make_train_args())
        buckets = _deimv2_param_buckets(train_model.model, make_train_args())
        names = {bucket for bucket, _ in buckets}
        assert "dinov3" in names  # conv_encoder.backbone.*
        assert "backbone" in names  # conv_encoder.model.* (HGNetv2 subtree)
        assert "decoder" in names and "other" in names
        # Norm/bias params split into the no-wd twin of every bucket.
        assert any(no_wd for _, no_wd in buckets)

    def test_backbone_batchnorm_lands_in_backbone_no_wd(self, fake_deimv2_transformers):
        # With freeze_backbone_batch_norms=False (official freeze_norm:
        # False) the HGNetv2 BN affine params become trainable parameters
        # and must land in the backbone bucket with weight decay exempt.
        train_model = Deimv2ObjectDetectionTrain(make_train_args())
        buckets = _deimv2_param_buckets(train_model.model, make_train_args())
        bn_params = [
            name
            for name, _ in train_model.model._model.named_parameters()
            if "normalization" in name
        ]
        assert bn_params, "BatchNorm params missing from the fake model"
        assert (("backbone", True)) in buckets
        assert len(buckets[("backbone", True)]) >= len(bn_params)


class TestTrainModel:
    def test_training_step_backward(self, fake_deimv2_transformers):
        train_model = Deimv2ObjectDetectionTrain(make_train_args())
        result = train_model.training_step(make_batch(), step=0)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert "train_loss/loss_ce" in result.log_dict

    def test_optimizer_param_groups_and_lrs(self, fake_deimv2_transformers):
        train_model = Deimv2ObjectDetectionTrain(make_train_args())
        optimizer, scheduler = train_model.get_optimizer(
            total_steps=100,
            global_batch_size=32,  # official total batch -> scale 1.0
            train_num_batches=10,
        )
        groups = {group["name"]: group for group in optimizer.param_groups}
        assert "dinov3" in groups
        assert "decoder" in groups
        assert "other" in groups
        # Backbone bucket tracks the variant ratio (0.5x for hgnetv2-n);
        # LambdaLR writes the warmup start into group["lr"], the base lrs
        # live in initial_lr.
        assert groups["dinov3"]["initial_lr"] == pytest.approx(8e-4 * 0.5)
        assert groups["decoder"]["initial_lr"] == pytest.approx(8e-4)
        assert groups["other"]["initial_lr"] == pytest.approx(8e-4)
        # Norm/bias parameters are exempt from weight decay.
        assert all(
            group.get("weight_decay", 1e-4) == 0.0
            for name, group in groups.items()
            if name.endswith("_no_wd")
        )
        assert all(
            group.get("weight_decay", 0.0) == pytest.approx(1e-4)
            for name, group in groups.items()
            if not name.endswith("_no_wd")
        )
        # Cosine schedule: first optimizer step is inside the 100-step
        # linear warmup, so every group's lr doubles from the 1/100 start.
        before = [group["lr"] for group in optimizer.param_groups]
        scheduler.step()
        after = [group["lr"] for group in optimizer.param_groups]
        assert all(a == pytest.approx(2 * b) for a, b in zip(after, before))

    def test_ema_and_export_state(self, fake_deimv2_transformers):
        train_model = Deimv2ObjectDetectionTrain(make_train_args())
        assert train_model.ema_model is not None
        train_model.training_step(make_batch(), step=0)
        train_model.on_train_batch_end()
        export_state = train_model.get_export_state_dict()
        assert any(key.startswith("ema_model.") for key in export_state)
