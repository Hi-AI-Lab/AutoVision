# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Unit tests for the object detection components."""

import pytest
import torch
from torch import nn

from autovision._task_models.object_detection_components import (
    FlatCosineLRScheduler,
    HungarianMatcher,
    HybridEncoder,
    ModelEMA,
    RTDETRCriterionV2,
    RTDETRv2Decoder,
    box_cxcywh_norm_to_xyxy,
    box_xyxy_to_cxcywh_norm,
)


def fake_targets(batch_size: int = 2, device="cpu"):
    return [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.6], [0.2, 0.3, 0.2, 0.2]], device=device),
            "labels": torch.tensor([0, 0], device=device),
        },
        {
            "boxes": torch.tensor([[0.7, 0.7, 0.3, 0.3]], device=device),
            "labels": torch.tensor([0], device=device),
        } if batch_size > 1 else torch.empty(0),
    ][:batch_size]


class TestBoxOps:
    def test_roundtrip(self):
        boxes = torch.tensor([[0.5, 0.5, 0.4, 0.6]])
        xyxy = box_cxcywh_norm_to_xyxy(boxes, size=(100, 50))
        assert torch.allclose(
            xyxy, torch.tensor([[30.0, 10.0, 70.0, 40.0]])
        )
        back = box_xyxy_to_cxcywh_norm(xyxy, size=(100, 50))
        assert torch.allclose(back, boxes, atol=1e-6)

    def test_giou_self_zero(self):
        from autovision._task_models.object_detection_components import (
            generalized_box_iou_loss,
        )

        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        loss = generalized_box_iou_loss(boxes, boxes)
        assert loss.max() == pytest.approx(0.0, abs=1e-6)


class TestHungarianMatcher:
    def test_perfect_match(self):
        matcher = HungarianMatcher()
        targets = fake_targets()
        targets[0]["labels"] = torch.tensor([0, 1])
        targets[1]["labels"] = torch.tensor([0])
        outputs = {
            "pred_logits": torch.randn(2, 10, 2),
            "pred_boxes": torch.zeros(2, 10, 4),
        }
        # Put target boxes in the first queries so the matcher must find them.
        outputs["pred_boxes"][0, :2] = targets[0]["boxes"]
        outputs["pred_boxes"][1, :1] = targets[1]["boxes"]
        # Make the classification cost agree with the boxes, so the joint
        # minimum is unambiguous.
        for b, target in enumerate(targets):
            for q, label in zip(range(target["boxes"].shape[0]), target["labels"]):
                outputs["pred_logits"][b, q] = -5.0
                outputs["pred_logits"][b, q, label] = 5.0
        indices = matcher(outputs, targets)
        assert indices[0][0].tolist() == [0, 1]
        assert indices[1][0].tolist() == [0]

    def test_empty_target(self):
        matcher = HungarianMatcher()
        outputs = {
            "pred_logits": torch.randn(1, 5, 1),
            "pred_boxes": torch.rand(1, 5, 4),
        }
        indices = matcher(outputs, [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}])
        assert indices[0][0].numel() == 0


class TestModelEMA:
    def test_convergence_to_weights(self):
        model = nn.Linear(4, 4)
        ema = ModelEMA(model, decay=0.99, warmup_steps=0)
        with torch.no_grad():
            model.weight.fill_(2.0)
        for _ in range(2000):
            ema.update(model)
        assert ema.model.weight[0, 0].item() == pytest.approx(2.0, abs=1e-3)

    def test_buffers_copied(self):
        model = nn.BatchNorm2d(4)
        ema = ModelEMA(model, decay=0.9, warmup_steps=0)
        with torch.no_grad():
            model.running_mean.fill_(5.0)
        ema.update(model)
        assert ema.model.running_mean[0].item() > 0

    def test_state_roundtrip(self):
        model = nn.Linear(4, 4)
        ema = ModelEMA(model, decay=0.99, warmup_steps=0)
        ema.update(model)
        ema2 = ModelEMA(model, decay=0.99, warmup_steps=0)
        ema2.load_state_dict(ema.state_dict())
        assert ema2.updates == ema.updates
        assert torch.allclose(ema2.model.weight, ema.model.weight)


class TestFlatCosine:
    def _sched(self, **kwargs):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        return FlatCosineLRScheduler(optimizer, **kwargs), optimizer

    def _lr_at(self, scheduler, optimizer, step):
        for _ in range(step):
            scheduler.step()
        return optimizer.param_groups[0]["lr"]

    def test_warmup_then_flat_then_decay(self):
        scheduler, optimizer = self._sched(
            total_steps=100, warmup_steps=10, flat_steps=40, no_aug_steps=10
        )
        # _lr_at steps forward cumulatively: steps 0-8 are warmup (lr<1),
        # step 9 completes the warmup at full lr.
        assert self._lr_at(scheduler, optimizer, 9) == pytest.approx(1.0)
        # flat plateau: cumulative steps 10..49
        assert self._lr_at(scheduler, optimizer, 40) == pytest.approx(1.0)
        # cosine region decreases (cumulative step 80)
        assert self._lr_at(scheduler, optimizer, 31) < 1.0
        # tail: lr returns to full for the no-augmentation window (baseline
        # flat-cosine semantics); min_factor is the cosine floor.
        assert self._lr_at(scheduler, optimizer, 15) == pytest.approx(1.0)

    def test_validation(self):
        model = nn.Linear(2, 2)
        with pytest.raises(ValueError, match="total_steps"):
            FlatCosineLRScheduler(
                torch.optim.SGD(model.parameters(), lr=1.0), total_steps=0
            )


class TestHybridEncoder:
    def test_shapes(self):
        encoder = HybridEncoder(in_channels=[64, 128, 256], hidden_dim=32)
        features = [
            torch.randn(2, 64, 56, 56),
            torch.randn(2, 128, 28, 28),
            torch.randn(2, 256, 14, 14),
        ]
        outputs = encoder(features)
        assert len(outputs) == 3
        assert outputs[0].shape == (2, 32, 56, 56)
        assert outputs[1].shape == (2, 32, 28, 28)
        assert outputs[2].shape == (2, 32, 14, 14)
        # AIFI lives on the deepest level.
        assert any("encoder" in name for name, _ in encoder.named_modules())

    def test_level_count_validated(self):
        encoder = HybridEncoder(in_channels=[64, 128])
        with pytest.raises(AssertionError):
            encoder([torch.randn(2, 64, 8, 8)])


class TestDecoderAndCriterion:
    def _make_decoder(self):
        return RTDETRv2Decoder(
            num_classes=2, hidden_dim=32, num_queries=20,
            feat_channels=[32, 32, 32], feat_strides=[8, 16, 32],
            num_levels=3, nhead=4, num_layers=2, num_denoising=0,
        )

    def _feats(self, batch_size: int = 1):
        return [
            torch.randn(batch_size, 32, 14, 14),
            torch.randn(batch_size, 32, 7, 7),
            torch.randn(batch_size, 32, 4, 4),
        ]

    def _model_and_targets(self):
        torch.manual_seed(0)
        decoder = self._make_decoder()
        out = decoder(self._feats())
        assert "dn_meta" not in out
        assert len(out["aux_outputs"]) == 1  # num_layers - 1
        final = {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        outputs = out["aux_outputs"] + [final]
        assert len(outputs) == 2
        assert outputs[-1]["pred_logits"].shape == (1, 20, 2)
        assert outputs[-1]["pred_boxes"].shape == (1, 20, 4)
        assert outputs[-1]["pred_boxes"].dtype == torch.float32
        assert outputs[0]["pred_boxes"].min() >= 0
        return outputs

    def test_criterion_backward(self):
        torch.manual_seed(0)
        decoder = self._make_decoder()
        feats = [
            torch.randn(1, 32, 14, 14, requires_grad=True),
            torch.randn(1, 32, 7, 7),
            torch.randn(1, 32, 4, 4),
        ]
        out = decoder(feats)
        criterion = RTDETRCriterionV2(HungarianMatcher())
        targets = fake_targets(batch_size=1)
        outputs_list = out.get("aux_outputs", []) + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        losses = criterion(outputs_list, targets, world_size=1)
        assert "loss" in losses
        assert torch.isfinite(losses["loss"])
        losses["loss"].backward()
        assert feats[0].grad is not None

    def test_criterion_all_loss_names(self):
        decoder_out = {
            "pred_logits": torch.randn(2, 10, 2, requires_grad=True),
            "pred_boxes": torch.rand(2, 10, 4),
        }
        criterion = RTDETRCriterionV2(HungarianMatcher())
        losses = criterion([decoder_out], fake_targets(), world_size=1)
        for name in ("loss_vfl", "loss_bbox", "loss_giou", "loss"):
            assert name in losses
        expected = (
            losses["loss_vfl"] + 5 * losses["loss_bbox"] + 2 * losses["loss_giou"]
        ).detach()
        assert losses["loss"].detach() == pytest.approx(expected, rel=1e-5)
