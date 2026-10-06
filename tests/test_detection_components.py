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
from autovision._task_models.object_detection_components.multistep_schedule import (
    MultiStepWarmupScheduler,
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
        """Advance to the ABSOLUTE scheduler step and read the LR."""
        while scheduler.last_epoch < step:
            scheduler.step()
        return optimizer.param_groups[0]["lr"]

    def test_warmup_then_flat_then_decay(self):
        """Baseline semantics: quadratic warmup -> flat -> cosine decaying to
        min_factor (0.5) -> tail HOLDS min_factor (no restart)."""
        scheduler, optimizer = self._sched(
            total_steps=100, warmup_steps=10, flat_steps=40, no_aug_steps=10
        )
        # Quadratic warmup: at absolute step 5 lr = (5/10)^2; step 10 -> 1.0.
        assert self._lr_at(scheduler, optimizer, 5) == pytest.approx(0.25)
        assert self._lr_at(scheduler, optimizer, 10) == pytest.approx(1.0)
        # flat plateau up to (exclusive) cosine_start = 50.
        assert self._lr_at(scheduler, optimizer, 49) == pytest.approx(1.0)
        # cosine midpoint (step 70, progress 0.5) -> halfway to min_factor.
        assert self._lr_at(scheduler, optimizer, 70) == pytest.approx(0.75)
        # cosine end floors at min_factor and the tail HOLDS it.
        assert self._lr_at(scheduler, optimizer, 90) == pytest.approx(0.5)
        assert self._lr_at(scheduler, optimizer, 99) == pytest.approx(0.5)

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


class TestTrainWindow:
    def test_means_over_window(self):
        from autovision._engine.engine import _TrainWindow

        window = _TrainWindow()
        for value in (2.0, 4.0, 6.0):
            window.update({"loss": torch.tensor(value), "tag": "not-a-tensor"})
        means = window.means()
        assert means["loss"] == pytest.approx(4.0)
        assert window.count == 3
        window.reset()
        assert window.count == 0
        assert window.means() == {}

    def test_seconds_per_step_positive(self):
        import time

        from autovision._engine.engine import _TrainWindow

        window = _TrainWindow()
        window.update({"loss": torch.tensor(1.0)})
        time.sleep(0.02)
        assert window.seconds_per_step() > 0.0


class TestValidationReportTables:
    def test_detection_report_renders_overall_and_per_class(self):
        from autovision._metrics.report import format_detection_validation_report

        metrics = {
            "map": 0.0625, "map_50": 0.1030, "map_75": 0.05,
            "mar": 0.20, "f1": 0.24, "precision": 0.25, "recall": 0.27,
        }
        per_class = [
            {"name": "car", "ap": 0.29, "ar": 0.47, "f1": 0.49,
             "precision": 0.37, "recall": 0.69},
            {"name": "bus", "ap": 0.08, "ar": 0.25, "f1": 0.28,
             "precision": 0.36, "recall": 0.22},
        ]
        report = format_detection_validation_report(
            "Validation (EMA weights)", metrics, per_class, max_dets=500,
            footer="note: val_loss covers VFL/bbox/GIoU only",
        )
        assert "Validation (EMA weights)" in report
        assert "mAR @500" in report
        assert "car" in report and "bus" in report
        assert "0.0625" in report and "0.1030" in report
        assert "val_loss covers VFL/bbox/GIoU only" in report
        # Box-drawing frame lines enclose the table body.
        lines = report.splitlines()
        assert lines[1].startswith("┏") and lines[-2].startswith("└")

    def test_metric_table_first_column_left_aligned(self):
        from autovision._metrics.report import format_metric_table

        table = format_metric_table(
            "T", ("Name", "Val"), [["longer-name", "1.0"], ["x", "22"]]
        )
        name_line = next(line for line in table.splitlines() if "longer-name" in line)
        assert "longer-name┃" in name_line.replace(" ", "")


class TestMultiStepWarmup:
    def _sched(self, **kwargs):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        return MultiStepWarmupScheduler(optimizer, **kwargs), optimizer

    def _lr_at(self, scheduler, optimizer, step):
        """Advance to the ABSOLUTE scheduler step and read the LR."""
        while scheduler.last_epoch < step:
            scheduler.step()
        return optimizer.param_groups[0]["lr"]

    def test_warmup_then_constant_with_drops(self):
        scheduler, optimizer = self._sched(
            total_steps=100, warmup_steps=10, milestones=(30, 60), gamma=0.8
        )
        assert self._lr_at(scheduler, optimizer, 10) == pytest.approx(1.0)
        assert self._lr_at(scheduler, optimizer, 29) == pytest.approx(1.0)
        assert self._lr_at(scheduler, optimizer, 30) == pytest.approx(0.8)
        assert self._lr_at(scheduler, optimizer, 60) == pytest.approx(0.64)
        assert self._lr_at(scheduler, optimizer, 99) == pytest.approx(0.64)

    def test_total_steps_validated(self):
        with pytest.raises(ValueError, match="total_steps"):
            MultiStepWarmupScheduler(
                torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=1.0),
                total_steps=0,
            )
