# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Tests for the task model layer: bases, RT-DETR family, mAP evaluator."""

import pytest
import torch
from torch import nn

from autovision._metrics.map_evaluator import COCOMAPEvaluator
from autovision._task_models import (
    RTDETRObjectDetection,
    RTDETRObjectDetectionTrain,
    RTDETRObjectDetectionTrainArgs,
    TaskModel,
    TrainModel,
)
from autovision._task_models.object_detection_components import ModelEMA
from autovision._task_models.task_model_io import ModelInputSpec, TensorSpec


@pytest.fixture(scope="module")
def task_model():
    torch.manual_seed(0)
    return RTDETRObjectDetection(
        num_classes=2, backbone="torchvision/resnet18",
        image_size=(64, 64), hidden_dim=32, num_queries=10,
        dec_layers=2, num_heads=4, num_top_queries=10,
    )


class TestTaskModelBase:
    def test_class_path_and_init_args(self, task_model):
        assert task_model.class_path.endswith("RTDETRObjectDetection")
        assert "num_classes" in task_model.init_args
        assert "load_weights" not in task_model.init_args  # ignored

    def test_deploy_recursive(self):
        calls = []

        class Deployable(nn.Module):
            def convert_to_deploy(self):
                calls.append(1)

        class TinyTask(TaskModel):
            def __init__(self):
                super().__init__(init_args={"a": 1})
                self.module = Deployable()

        TinyTask().deploy()
        assert len(calls) == 1

    def test_load_train_state_dict_strips_ema_prefix(self, task_model):
        ema = ModelEMA(task_model)
        state = {f"ema_model.{k}": v for k, v in ema.model.state_dict().items()}
        clone = RTDETRObjectDetection(
            num_classes=2, backbone="torchvision/resnet18",
            image_size=(64, 64), hidden_dim=32, num_queries=10,
            dec_layers=2, num_heads=4, num_top_queries=10,
        )
        clone.load_train_state_dict(state)  # strict load, prefix stripped


class TestRTDETRTaskModel:
    def test_forward_shapes(self, task_model):
        x = torch.randn(2, 3, 64, 64)
        logits, boxes = task_model.forward_backend(task_model.preprocess_batch(x))
        assert logits.shape == (2, 10, 2)
        assert boxes.shape == (2, 10, 4)
        assert (boxes >= 0).all() and (boxes <= 1).all()

    def test_three_stage_predict(self, task_model):
        task_model.eval()
        image = torch.randint(0, 256, (3, 128, 128), dtype=torch.uint8)
        result = task_model.predict(image, threshold=0.0)
        assert set(result) == {"labels", "bboxes", "scores"}
        assert result["bboxes"].shape[1] == 4  # xyxy absolute pixels

    def test_model_input_spec(self, task_model):
        spec = task_model.model_input_spec
        inputs = spec.example_inputs(batch_size=2)
        assert inputs["images"].shape == (2, 3, 64, 64)

    def test_gradients_flow(self, task_model):
        x = torch.randn(1, 3, 64, 64)
        logits, _ = task_model.forward_backend(task_model.preprocess_batch(x))
        logits.sum().backward()
        assert task_model.encoder.lateral_convs[0].conv.weight.grad is not None


class TestRTDETRTrainModel:
    def _train_model(self):
        args = RTDETRObjectDetectionTrainArgs(
            num_classes=2, image_size=(64, 64), hidden_dim=32,
            num_queries=10, dec_layers=2, num_heads=4, backbone="torchvision/resnet18",
            num_top_queries=10,
        )
        return RTDETRObjectDetectionTrain(args), args

    def _batch(self, device="cpu"):
        torch.manual_seed(0)
        return {
            "image": torch.randn(2, 3, 64, 64),
            "bboxes": [
                torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
                torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]),
            ],
            "classes": [torch.tensor([0]), torch.tensor([1, 0])],
            "original_size": [(64, 64), (64, 64)],
        }

    def test_is_train_model(self):
        train_model, _ = self._train_model()
        assert isinstance(train_model, TrainModel)
        assert train_model.task == "object_detection"

    def test_training_step_and_backward(self):
        train_model, _ = self._train_model()
        result = train_model.training_step(self._batch(), step=0)
        assert torch.isfinite(result.loss)
        assert "loss_vfl" in result.log_dict
        result.loss.backward()

    def test_param_groups_structure(self):
        train_model, args = self._train_model()
        optimizer, scheduler = train_model.get_optimizer(total_steps=1000, global_batch_size=16)
        groups = {g["name"]: g for g in optimizer.param_groups}
        assert groups["backbone"]["lr"] == pytest.approx(args.lr * args.backbone_lr_factor)
        assert groups["detector"]["lr"] == pytest.approx(args.lr)
        # no-decay groups keep their source's LR, with wd=0
        if "backbone_no_wd" in groups:
            assert groups["backbone_no_wd"]["lr"] == pytest.approx(args.lr * args.backbone_lr_factor)
            assert groups["backbone_no_wd"]["weight_decay"] == 0.0
            backbone_ids = {id(p) for p in groups["backbone"]["params"]}
            for p in groups["backbone_no_wd"]["params"]:
                assert id(p) not in backbone_ids  # not duplicated
        if "detector_no_wd" in groups:
            assert groups["detector_no_wd"]["lr"] == pytest.approx(args.lr)
            assert groups["detector_no_wd"]["weight_decay"] == 0.0

    def test_lr_sqrt_scaling(self):
        train_model, args = self._train_model()
        optimizer, _ = train_model.get_optimizer(total_steps=1000, global_batch_size=64)
        expected = args.lr * (64 / args.default_batch_size) ** 0.5
        assert optimizer.param_groups[1]["lr"] == pytest.approx(expected)

    def test_ema_updated_on_batch_end(self):
        train_model, args = self._train_model(use_ema=True)
        assert train_model.ema_model is not None
        before = train_model.ema_model.updates
        train_model.on_train_batch_end()
        assert train_model.ema_model.updates == before + 1

    def test_export_state_dict_prefers_ema(self):
        train_model, _ = self._train_model()
        train_model.on_train_batch_end()
        export_state = train_model.get_export_state_dict()
        assert all(k.startswith("ema_model.") for k in export_state)

    def test_validation_and_map_computation(self):
        train_model, _ = self._train_model()
        train_model.set_categories({0: "cat", 1: "dog"})
        batch = self._batch()
        train_model.validation_step(batch, step=0)
        metrics = train_model.compute_metrics("val")
        assert "val_metric/map" in metrics
        assert 0.0 <= metrics["val_metric/map"] <= 1.0

    def test_gradient_clipping(self):
        train_model, _ = self._train_model()
        optimizer, _ = train_model.get_optimizer(total_steps=1000, global_batch_size=16)
        result = train_model.training_step(self._batch(), step=0)
        result.loss.backward()
        norm = train_model.clip_gradients(optimizer)
        assert norm is not None and norm >= 0


# patch the helper used in the EMA test above
def _train_model(self, use_ema: bool = True):
    args = RTDETRObjectDetectionTrainArgs(
        num_classes=2, image_size=(64, 64), hidden_dim=32,
        num_queries=10, dec_layers=2, num_heads=4, backbone="torchvision/resnet18",
        num_top_queries=10, use_ema_model=use_ema,
    )
    return RTDETRObjectDetectionTrain(args), args


TestRTDETRTrainModel._train_model = _train_model  # type: ignore[attr-defined]


class TestCOCOMAPEvaluator:
    def test_perfect_predictions_score_one(self):
        evaluator = COCOMAPEvaluator({0: "cat"})
        gt = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([0])}]
        evaluator.set_ground_truth(gt)
        evaluator.update(
            [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
              "labels": torch.tensor([0]), "scores": torch.tensor([0.9])}]
        )
        metrics = evaluator.compute()
        assert metrics["map"] == pytest.approx(1.0, abs=1e-6)

    def test_empty_predictions_score_zero(self):
        evaluator = COCOMAPEvaluator({0: "cat"})
        evaluator.set_ground_truth(
            [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([0])}]
        )
        assert evaluator.compute()["map"] == 0.0


class TestModelInputSpec:
    def test_example_inputs(self):
        spec = ModelInputSpec(
            input_specs={"images": TensorSpec(shape=(3, 64, 64), dtype=torch.float32)}
        )
        inputs = spec.example_inputs(batch_size=3)
        assert inputs["images"].shape == (3, 3, 64, 64)

    def test_dynamic_rank_validated(self):
        with pytest.raises(ValueError, match="rank"):
            ModelInputSpec(
                input_specs={"images": TensorSpec(shape=(3, 64, 64), dtype=torch.float32)},
                input_dynamic_shapes={"images": (None,)},
            )


class TestBackbonePretrained:
    def test_load_pretrained_backbone_copies_weights(self):
        import torchvision.models as tvm

        from autovision._task_models.rtdetr_object_detection import (
            RTDETRObjectDetectionTrain,
            RTDETRObjectDetectionTrainArgs,
        )
        from autovision._task_models.rtdetr_object_detection.train_model import (
            load_pretrained_backbone,
        )

        args = RTDETRObjectDetectionTrainArgs(
            num_classes=2, backbone="torchvision/resnet18", image_size=(64, 64),
            hidden_dim=32, num_queries=10, dec_layers=1, num_heads=4,
            num_top_queries=10,
        )
        train_model = RTDETRObjectDetectionTrain(args)
        reference = tvm.get_model(
            "resnet18", weights=tvm.get_model_weights("resnet18").DEFAULT
        )
        before = train_model.model.feature_extractor.conv1.weight.detach().clone()
        copied = load_pretrained_backbone(train_model.model, "torchvision/resnet18")
        after = train_model.model.feature_extractor.conv1.weight.detach()
        assert copied > 20
        assert not torch.equal(before, after)
        # ImageNet conv1 has the canonical stem layout; a random init would
        # not match the reference exactly.
        expected = reference.conv1.weight.detach()
        assert torch.allclose(after, expected)

    def test_unsupported_package_rejected(self):
        from autovision._task_models.rtdetr_object_detection import (
            RTDETRObjectDetectionTrain,
            RTDETRObjectDetectionTrainArgs,
        )
        from autovision._task_models.rtdetr_object_detection.train_model import (
            load_pretrained_backbone,
        )

        args = RTDETRObjectDetectionTrainArgs(
            num_classes=2, backbone="torchvision/resnet18", image_size=(64, 64),
            hidden_dim=32, num_queries=10, dec_layers=1, num_heads=4,
            num_top_queries=10,
        )
        train_model = RTDETRObjectDetectionTrain(args)
        with pytest.raises(ValueError, match="backbone_pretrained"):
            load_pretrained_backbone(train_model.model, "ultralytics/yolov8n")


class TestCriterionLossBoxes:
    def test_giou_uses_matched_diagonal_not_full_matrix(self):
        # Regression: generalized_box_iou_loss is pairwise (N, M); the box
        # loss must take the matched diagonal. Summing the full matrix
        # inflated loss_giou ~N× on dense datasets and destroyed box
        # learning (found via VisDrone convergence run).
        from autovision._task_models.object_detection_components.rtdetrv2_criterion import (
            RTDETRCriterionV2,
        )
        from autovision._task_models.object_detection_components.matcher import (
            HungarianMatcher,
        )

        torch.manual_seed(0)
        matcher = HungarianMatcher()
        criterion = RTDETRCriterionV2(matcher)
        # Two images with 3 and 2 boxes; predictions deliberately off.
        targets = [
            {"boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.2, 0.2, 0.05, 0.05],
                                    [0.8, 0.8, 0.2, 0.1]]),
             "labels": torch.tensor([0, 1, 2])},
            {"boxes": torch.tensor([[0.3, 0.6, 0.15, 0.2], [0.7, 0.3, 0.1, 0.1]]),
             "labels": torch.tensor([1, 2])},
        ]
        outputs_list = [
            {"pred_logits": torch.randn(2, 10, 3) * 0.1,
             "pred_boxes": torch.rand(2, 10, 4) * 0.4 + 0.3}
            for _ in range(2)
        ]
        num_boxes = sum(len(t["boxes"]) for t in targets)
        loss_dict = criterion(outputs_list, targets, world_size=1)

        # Reference: GIoU loss computed per aligned pair only.
        from autovision._task_models.object_detection_components.box_ops import (
            generalized_box_iou_loss,
        )
        from autovision._task_models.object_detection_components.rtdetrv2_criterion import (
            _cxcywh_to_xyxy_unit,
        )

        def cxcywh_to_xyxy(b):
            cx, cy, w, h = b.unbind(-1)
            return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)

        outs = outputs_list[-1]
        indices = matcher(outs, targets)
        reference = torch.tensor(0.0)
        for b, (src, dst) in enumerate(indices):
            pred = cxcywh_to_xyxy(outs["pred_boxes"][b, src])
            gt = cxcywh_to_xyxy(targets[b]["boxes"][dst])
            reference = reference + generalized_box_iou_loss(pred, gt).diagonal().sum()
        expected = float(reference) / num_boxes
        assert loss_dict["loss_giou"] == pytest.approx(expected, rel=1e-4)
        # The full-matrix sum includes every pred-vs-other-GT cross term;
        # the inflation factor grows with the box count (on VisDrone's ~700
        # boxes/image batches it measured ~92x).
        full_matrix = sum(
            float(generalized_box_iou_loss(
                cxcywh_to_xyxy(outs["pred_boxes"][b, src]),
                cxcywh_to_xyxy(targets[b]["boxes"][dst]),
            ).sum())
            for b, (src, dst) in enumerate(indices)
        ) / num_boxes
        assert full_matrix > 2 * expected
