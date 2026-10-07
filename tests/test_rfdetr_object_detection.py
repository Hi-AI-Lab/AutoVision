# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Tests for the RF-DETR object detection task model.

The real detector needs transformers >= 5.9 (RfDetr classes), so the logic
runs against an injected fake ``transformers`` module that mirrors the HF
API surface the wrapper uses (forward with labels returning
loss/loss_dict/logits/pred_boxes, class_embed + grouped encoder heads) —
same pattern as tests/test_models_rfdetr.py.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from autovision._task_models.rfdetr_object_detection.config import (
    RFDETRObjectDetectionTrainArgs,
    resolve_rfdetr_variant,
)
from autovision._task_models.rfdetr_object_detection.task_model import (
    RFDETRObjectDetection,
    _reinitialize_class_head,
)
from autovision._task_models.rfdetr_object_detection.train_model import (
    RFDETRObjectDetectionTrain,
    _rfdetr_param_buckets,
)

# ── fake transformers mirroring the HF RF-DETR API ───────────────────────


class FakeRfDetrDinov2Config:
    def __init__(self, **kwargs) -> None:
        self.model_type = "rf_detr_dinov2"
        defaults = dict(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            image_size=224,
            patch_size=14,
            num_windows=4,
            out_indices=[2, 5, 8, 11],
            out_features=["stage2", "stage5", "stage8", "stage11"],
            use_mask_token=True,
        )
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)


class FakeRfDetrConfig:
    def __init__(self, backbone_config=None, **kwargs) -> None:
        self.model_type = "rf_detr"
        self.backbone_config = backbone_config or FakeRfDetrDinov2Config()
        self.d_model = 32
        self.num_queries = 16
        self.group_detr = 13
        self.num_labels = 80
        self.decoder_layers = 3
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeEncoderBlock(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv2d(width, width, 3, padding=1)


class _FakeRfDetrDinov2Backbone(nn.Module):
    """Attribute tree mirroring the real names for bucketing tests."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Linear(8, 8)
        self.layernorm = nn.LayerNorm(8)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_FakeEncoderBlock() for _ in range(2)])


class _FakeRfDetrConvEncoder(nn.Module):
    """Real structure: ConvEncoder{backbone: Dinov2Backbone, projector}."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _FakeRfDetrDinov2Backbone()
        self.projector = nn.Linear(8, 8)


class _FakeRfDetrInner(nn.Module):
    def __init__(self, config: FakeRfDetrConfig) -> None:
        super().__init__()
        self.backbone = _FakeRfDetrConvEncoder()
        self.decoder = nn.Linear(32, 32)
        self.query_feat = nn.Parameter(torch.randn(config.num_queries * 2, 32))
        self.enc_out_class_embed = nn.ModuleList(
            [nn.Linear(32, config.num_labels) for _ in range(2)]
        )
        self.enc_out_bbox_embed = nn.ModuleList([nn.Linear(32, 4) for _ in range(2)])


class FakeRfDetrForObjectDetection(nn.Module):
    def __init__(self, config: FakeRfDetrConfig) -> None:
        super().__init__()
        self.config = config
        self.model = _FakeRfDetrInner(config)
        self.proj = nn.Conv2d(3, 32, 7, stride=7)
        self.class_embed = nn.Linear(32, config.num_labels)
        self.bbox_embed = nn.Linear(32, 4)

    def forward(self, pixel_values, pixel_mask=None, labels=None, **kwargs):
        features = self.proj(pixel_values)
        hidden = torch.nn.functional.adaptive_avg_pool2d(features, (4, 4))
        hidden = hidden.flatten(2).transpose(1, 2)  # (B, 16, 32)
        logits = self.class_embed(hidden)
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
def fake_rfdetr_transformers(monkeypatch):
    fake = types.ModuleType("transformers")
    fake.RfDetrConfig = FakeRfDetrConfig
    fake.RfDetrForObjectDetection = FakeRfDetrForObjectDetection
    fake.RfDetrDinov2Config = FakeRfDetrDinov2Config
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return fake


def make_task_model(num_classes: int = 3, **kwargs) -> RFDETRObjectDetection:
    return RFDETRObjectDetection(
        num_classes=num_classes,
        model="rf-detr-nano",
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


# ── config ───────────────────────────────────────────────────────────────


class TestConfig:
    def test_unknown_variant(self):
        with pytest.raises(ValueError, match="Unknown rfdetr detection model"):
            resolve_rfdetr_variant("rf-detr-huge")

    def test_auto_image_size_resolves_to_variant_native(self):
        args = RFDETRObjectDetectionTrainArgs(model="rf-detr-nano", image_size="auto")
        args.resolve_image_size()
        assert args.image_size == (384, 384)

    def test_custom_size_divisibility(self):
        args = RFDETRObjectDetectionTrainArgs(
            model="rf-detr-nano", image_size=(800, 800)
        )
        args.resolve_image_size()
        assert args.image_size == (800, 800)

        base = RFDETRObjectDetectionTrainArgs(
            model="rf-detr-base", image_size=(800, 800)
        )
        with pytest.raises(ValueError, match="not divisible"):
            base.resolve_image_size()

    def test_resolve_auto_scheduler(self):
        args = RFDETRObjectDetectionTrainArgs(model="rf-detr-nano")
        args.resolve_auto(
            total_steps=1000, train_num_batches=100, gradient_accumulation_steps=1
        )
        assert args.lr_scheduler == "cosine"
        assert args.lr_warmup_steps == 100
        assert args.image_size == (384, 384)


# ── gating ───────────────────────────────────────────────────────────────


class TestGating:
    def test_no_support_raises_informative(self, monkeypatch):
        from autovision._task_models.rfdetr_object_detection import task_model

        monkeypatch.setattr(task_model, "_od_classes", lambda: None)
        with pytest.raises(ImportError, match="transformers"):
            make_task_model()


# ── task model ───────────────────────────────────────────────────────────


class TestTaskModel:
    def test_forward_shapes(self, fake_rfdetr_transformers):
        model = make_task_model(num_classes=3)
        model.eval()
        with torch.no_grad():
            logits, boxes = model(torch.randn(2, 3, 224, 224))
        assert logits.shape == (2, 16, 3)
        assert boxes.shape == (2, 16, 4)
        assert (boxes >= 0).all() and (boxes <= 1).all()

    def test_loss_backward(self, fake_rfdetr_transformers):
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

    def test_postprocess_decodes_to_original_pixels(self, fake_rfdetr_transformers):
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

    def test_class_head_retile(self, fake_rfdetr_transformers):
        model = make_task_model(num_classes=80)  # native head: 80 rows
        head = model._model.class_embed
        old_weight = head.weight.detach().clone()
        _reinitialize_class_head(model._model, 3)  # shrink: truncation
        assert model._model.class_embed.weight.shape == (3, 32)
        torch.testing.assert_close(model._model.class_embed.weight, old_weight[:3])
        # Grow 3 -> 7: cyclic tiling of the surviving (shrunk) rows.
        shrunk = model._model.class_embed.weight.detach().clone()
        _reinitialize_class_head(model._model, 7)
        assert model._model.class_embed.weight.shape == (7, 32)
        torch.testing.assert_close(
            model._model.class_embed.weight,
            shrunk[torch.arange(7) % 3],
        )
        # Grouped encoder heads are retiled too.
        assert model._model.model.enc_out_class_embed[0].weight.shape == (7, 32)


# ── train model ──────────────────────────────────────────────────────────


def make_train_args(**overrides) -> RFDETRObjectDetectionTrainArgs:
    defaults = dict(
        model="rf-detr-nano",
        num_classes=3,
        image_size=(224, 224),
        num_queries=16,
        num_select=16,
        load_weights=False,
        use_ema_model=True,
        ema_warmup_steps=2,
        eval_max_dets=10,
    )
    defaults.update(overrides)
    return RFDETRObjectDetectionTrainArgs(**defaults)


class TestTrainModel:
    def test_training_step_backward(self, fake_rfdetr_transformers):
        train_model = RFDETRObjectDetectionTrain(make_train_args())
        result = train_model.training_step(make_batch(), step=0)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert "train_loss/loss_ce" in result.log_dict

    def test_optimizer_param_groups_and_lrs(self, fake_rfdetr_transformers):
        train_model = RFDETRObjectDetectionTrain(
            make_train_args(lr=1e-4, lr_encoder=1.5e-4)
        )
        optimizer, scheduler = train_model.get_optimizer(
            total_steps=100,
            global_batch_size=16,
            train_num_batches=10,
        )
        groups = {group["name"]: group for group in optimizer.param_groups}
        # Bucket names from the fake's real-name attribute tree.
        assert "encoder_0" in groups  # embeddings (ViT layer 0)
        assert "encoder_1" in groups  # encoder.layer.0 -> layer 1
        assert "encoder_2" in groups  # encoder.layer.1 -> layer 2
        assert "decoder" in groups
        assert "other" in groups
        # LambdaLR writes the warmup start into group["lr"]; the base lrs
        # live in initial_lr. ViT layers: embeddings = layer 0, blocks
        # are 1-based (official backbone.py scheme).
        assert groups["encoder_0"]["initial_lr"] == pytest.approx(1.5e-4)
        assert groups["encoder_1"]["initial_lr"] == pytest.approx(1.5e-4 * 0.8)
        assert groups["encoder_2"]["initial_lr"] == pytest.approx(1.5e-4 * 0.64)
        assert groups["decoder"]["initial_lr"] == pytest.approx(1e-4 * 0.7)
        assert groups["other"]["initial_lr"] == pytest.approx(1e-4)
        # Bias parameters are exempt from weight decay.
        assert all(
            group.get("weight_decay", 1e-4) == 0.0
            for name, group in groups.items()
            if name.endswith("_no_wd")
        )
        # Cosine schedule: first optimizer step is inside the 100-step
        # linear warmup, so every group's lr doubles from the 1/100 start.
        before = [group["lr"] for group in optimizer.param_groups]
        scheduler.step()
        after = [group["lr"] for group in optimizer.param_groups]
        assert all(a == pytest.approx(2 * b) for a, b in zip(after, before))

    def test_param_buckets_layerwise(self, fake_rfdetr_transformers):
        train_model = RFDETRObjectDetectionTrain(make_train_args())
        buckets = _rfdetr_param_buckets(train_model.model, make_train_args())
        names = {bucket for bucket, _ in buckets}
        assert "encoder_0" in names and "encoder_1" in names
        assert "decoder" in names and "other" in names

    def test_ema_and_export_state(self, fake_rfdetr_transformers):
        train_model = RFDETRObjectDetectionTrain(make_train_args())
        assert train_model.ema_model is not None
        train_model.training_step(make_batch(), step=0)
        train_model.on_train_batch_end()
        export_state = train_model.get_export_state_dict()
        assert any(key.startswith("ema_model.") for key in export_state)
