"""Tests for the D-FINE object detection family (FGL + DDF)."""

import pytest
import torch

from autovision import load_task_model as load_model
from autovision import train_object_detection
from autovision._task_models.dfine_object_detection import (
    DFINEObjectDetection,
    DFINEObjectDetectionTrain,
    DFINEObjectDetectionTrainArgs,
)
from autovision._task_models.object_detection_components.dfine_criterion import (
    DFINECriterionV2,
)
from autovision._task_models.object_detection_components.dfine_decoder import (
    DFINEv2Decoder,
)
from autovision._task_models.object_detection_components.dfine_utils import (
    bbox2distance,
    ddf_loss,
    distance2bbox,
    fg_loss,
    integrate_corners,
    integration_weights,
    soft_bin_targets,
)
from autovision._task_models.object_detection_components.matcher import (
    HungarianMatcher,
)
from tests.conftest import create_image
from tests.test_train_object_detection import (
    data_config,
    make_coco_dataset,
    transform_args,
)

REG_MAX = 8  # small bin layout keeps tests fast


def make_decoder(num_layers: int = 2, num_queries: int = 20) -> DFINEv2Decoder:
    torch.manual_seed(0)
    return DFINEv2Decoder(
        num_classes=2,
        hidden_dim=32,
        num_queries=num_queries,
        feat_channels=[32, 32, 32],
        feat_strides=[8, 16, 32],
        num_levels=3,
        nhead=4,
        num_layers=num_layers,
        num_denoising=50,
        reg_max=REG_MAX,
    )


def make_feats(batch: int = 2):
    return [
        torch.randn(batch, 32, 8, 8),
        torch.randn(batch, 32, 4, 4),
        torch.randn(batch, 32, 2, 2),
    ]


def make_targets(batch: int = 2):
    torch.manual_seed(1)
    return [
        {"boxes": torch.rand(3, 4) * 0.3 + 0.3, "labels": torch.tensor([0, 1, 0])}
        for _ in range(batch)
    ]


def make_train_model(**overrides) -> DFINEObjectDetectionTrain:
    args = DFINEObjectDetectionTrainArgs(
        num_classes=2,
        image_size=(64, 64),
        num_queries=10,
        num_top_queries=10,
        hidden_dim=32,
        dec_layers=2,
        reg_max=REG_MAX,
        **overrides,
    )
    return DFINEObjectDetectionTrain(args)


def make_batch(size: int = 64, images: int = 2):
    targets = make_targets(images)
    return {
        "image": torch.randn(images, 3, size, size),
        "bboxes": [t["boxes"] for t in targets],
        "classes": [t["labels"] for t in targets],
        "image_path": [f"img_{i}.jpg" for i in range(images)],
        "original_size": [(size, size)] * images,
    }


class TestDFineUtils:
    def test_integration_weights_span_unit_range(self):
        weights = integration_weights(REG_MAX)
        assert weights.shape == (2 * REG_MAX + 1,)
        assert weights[0].item() == pytest.approx(-1.0)
        assert weights[-1].item() == pytest.approx(1.0)
        assert weights[REG_MAX].item() == pytest.approx(0.0)

    def test_distance_bbox_roundtrip(self):
        ref = torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.3, 0.7, 0.1, 0.1]])
        gt_xyxy = torch.tensor([[0.4, 0.3, 0.6, 0.6], [0.25, 0.6, 0.35, 0.8]])
        distances = bbox2distance(ref, gt_xyxy)
        assert distances.max() <= 1.0 and distances.min() >= -1.0
        rebuilt = distance2bbox(ref, distances)
        cx, cy, w, h = rebuilt.unbind(-1)
        rebuilt_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
        torch.testing.assert_close(rebuilt_xyxy, gt_xyxy, atol=1e-5, rtol=1e-4)

    def test_uniform_distribution_decodes_to_reference(self):
        # Uniform logits → uniform distribution → distance 0 on every side →
        # the decoded box IS the reference box.
        ref = torch.rand(5, 4) * 0.4 + 0.2
        ref[:, 2:] = ref[:, 2:].clamp(min=0.05)
        corners = torch.zeros(5, 4 * (2 * REG_MAX + 1))
        weights = integration_weights(REG_MAX)
        decoded = distance2bbox(ref, integrate_corners(corners, weights))
        torch.testing.assert_close(decoded, ref)

    def test_soft_bin_targets_interpolates_between_bins(self):
        # distance 0.5 (in ref units of 1/REG_MAX per bin → bin 0.5×...):
        # distance = 0.5 / REG_MAX maps exactly between bin REG_MAX and +1.
        distances = torch.full((1, 4), 0.5 / REG_MAX)
        left, right, wl, wr = soft_bin_targets(distances, REG_MAX)
        assert (left == REG_MAX).all() and (right == REG_MAX + 1).all()
        torch.testing.assert_close(wl, torch.full_like(wl, 0.5))
        torch.testing.assert_close(wr, torch.full_like(wr, 0.5))

    def test_soft_bin_targets_clamps_out_of_range(self):
        distances = torch.full((1, 4), -3.0)
        left, right, wl, wr = soft_bin_targets(distances, REG_MAX)
        assert (left == 0).all() and (right == 1).all()
        torch.testing.assert_close(wl, torch.ones_like(wl))
        torch.testing.assert_close(wr, torch.zeros_like(wr))


class TestDFineDecoder:
    def test_training_output_contract(self):
        model = make_decoder(num_layers=3)
        model.train()
        out = model(make_feats(), targets=make_targets())
        assert out["pred_logits"].shape == (2, 20, 2)
        assert out["pred_boxes"].shape == (2, 20, 4)
        assert len(out["aux_outputs"]) == 2
        assert len(out["pred_corners_list"]) == 3
        assert out["pred_corners_list"][0].shape == (2, 20, 4 * (2 * REG_MAX + 1))
        assert len(out["ref_boxes_list"]) == 3
        assert out["ref_boxes_list"][0].shape == (2, 20, 4)
        # dn segment split off
        dn = out["dn_meta"]
        assert out["dn_pred_corners_list"][0].shape[1] == dn["dn_num_split"][0]
        assert out["dn_ref_boxes_list"][-1].shape[1] == dn["dn_num_split"][0]

    def test_ref_boxes_are_detached(self):
        model = make_decoder()
        model.train()
        out = model(make_feats(), targets=make_targets())
        for ref in out["ref_boxes_list"] + out["dn_ref_boxes_list"]:
            assert not ref.requires_grad

    def test_corners_carry_gradients(self):
        model = make_decoder()
        model.train()
        out = model(make_feats(), targets=make_targets())
        loss = out["pred_boxes"].sum() + sum(
            c.sum() for c in out["pred_corners_list"]
        )
        loss.backward()
        for head in model.dec_bbox_head:
            assert head.layers[-1].weight.grad is not None
            assert head.layers[-1].weight.grad.abs().sum() > 0

    def test_zero_init_starts_from_reference(self):
        model = make_decoder()
        model.train()
        out = model(make_feats(), targets=make_targets())
        torch.testing.assert_close(
            out["ref_boxes_list"][0], out["pred_boxes"], atol=1e-4, rtol=1e-3
        )

    def test_eval_uses_last_layer_only(self):
        model = make_decoder(num_layers=3)
        model.eval()
        with torch.no_grad():
            out = model(make_feats())
        assert out["pred_logits"].shape == (2, 20, 2)
        assert out["pred_boxes"].shape == (2, 20, 4)
        assert "aux_outputs" not in out
        assert "pred_corners_list" not in out

    def test_empty_gt_skips_denoising(self):
        model = make_decoder()
        model.train()
        empty = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]
        out = model(make_feats(1), targets=empty * 1)
        assert "dn_meta" not in out
        assert out["pred_boxes"].shape == (1, 20, 4)

    def test_boxes_stay_finite(self):
        model = make_decoder()
        model.train()
        out = model(make_feats(), targets=make_targets())
        assert torch.isfinite(out["pred_boxes"]).all()
        assert (out["pred_boxes"][:, :, 2:] >= 0).all()


class TestFGL:
    def _layer_inputs(self, quality: float):
        ref = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]])  # (B=1, Q=1, 4)
        gt_cxcywh = torch.tensor([[0.5, 0.5, 0.4 * quality, 0.4 * quality]])
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        targets = [{"boxes": gt_cxcywh, "labels": torch.tensor([0])}]
        return ref, gt_cxcywh, indices, targets

    def _corners_for(self, ref, gt_cxcywh):
        """Corner logits whose softmax IS the FGL soft target distribution."""
        gt_xyxy = torch.stack([
            gt_cxcywh[0, 0] - gt_cxcywh[0, 2] / 2,
            gt_cxcywh[0, 1] - gt_cxcywh[0, 3] / 2,
            gt_cxcywh[0, 0] + gt_cxcywh[0, 2] / 2,
            gt_cxcywh[0, 1] + gt_cxcywh[0, 3] / 2,
        ]).unsqueeze(0)
        distances = bbox2distance(ref[0], gt_xyxy)
        left, right, wl, wr = soft_bin_targets(distances, REG_MAX)
        probs = torch.full((1, 1, 4, 2 * REG_MAX + 1), 1e-12)
        for side in range(4):
            probs[0, 0, side, left[0, side]] += wl[0, side]
            probs[0, 0, side, right[0, side]] += wr[0, side]
        return probs.log().reshape(1, 1, 4 * (2 * REG_MAX + 1))

    def test_perfect_distribution_near_zero(self):
        ref, gt, indices, targets = self._layer_inputs(quality=1.0)
        corners = self._corners_for(ref, gt)
        loss = fg_loss(
            corners, ref, gt.view(1, 1, 4), targets, indices,
            torch.tensor(1.0), REG_MAX,
        )
        assert loss.item() < 1e-3

    def test_wrong_distribution_positive(self):
        ref, gt, indices, targets = self._layer_inputs(quality=1.0)
        corners = torch.zeros(1, 1, 4 * (2 * REG_MAX + 1))
        loss = fg_loss(
            corners, ref, gt.view(1, 1, 4), targets, indices,
            torch.tensor(1.0), REG_MAX,
        )
        assert loss.item() > 0.1

    def test_iou_weight_scales_loss(self):
        # Same distribution, lower-quality target (smaller GT → smaller IoU)
        # must produce a smaller weighted loss.
        ref, gt_hi, indices, targets_hi = self._layer_inputs(quality=1.0)
        _, gt_lo, _, targets_lo = self._layer_inputs(quality=0.5)
        corners = torch.zeros(1, 1, 4 * (2 * REG_MAX + 1))
        pred_boxes = torch.zeros(1, 1, 4)  # placeholder, IoU uses pred vs gt
        pred_boxes[0, 0] = gt_hi[0]
        loss_hi = fg_loss(corners, ref, pred_boxes, targets_hi, indices,
                          torch.tensor(1.0), REG_MAX)
        loss_lo = fg_loss(corners, ref, pred_boxes, targets_lo, indices,
                          torch.tensor(1.0), REG_MAX)
        assert loss_lo.item() < loss_hi.item()

    def test_empty_indices_zero_with_graph(self):
        corners = torch.zeros(1, 1, 4 * (2 * REG_MAX + 1), requires_grad=True)
        ref = torch.zeros(1, 4)
        loss = fg_loss(
            corners, ref, corners.new_zeros(1, 1, 4),
            [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}],
            [(torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))],
            torch.tensor(1.0), REG_MAX,
        )
        loss.backward()
        assert loss.item() == 0.0
        assert corners.grad is not None


class TestDDF:
    def _stacked(self, corners_bq_side_bin):
        return corners_bq_side_bin.reshape(
            corners_bq_side_bin.shape[0], corners_bq_side_bin.shape[1],
            4 * (2 * REG_MAX + 1),
        )

    def test_identical_distributions_self_extinguish(self):
        corners = self._stacked(torch.randn(2, 3, 4, 2 * REG_MAX + 1))
        boxes = torch.rand(2, 3, 4) * 0.3 + 0.3
        indices = [
            (torch.tensor([0, 1]), torch.tensor([0, 1])) for _ in range(2)
        ]
        loss = ddf_loss(corners, corners.clone(), boxes, boxes.clone(),
                        indices, torch.tensor(6.0), REG_MAX)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_divergent_distributions_positive(self):
        torch.manual_seed(3)
        student = self._stacked(torch.randn(2, 3, 4, 2 * REG_MAX + 1) * 4)
        teacher = self._stacked(torch.randn(2, 3, 4, 2 * REG_MAX + 1) * 4)
        boxes = torch.rand(2, 3, 4) * 0.3 + 0.3
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1])) for _ in range(2)]
        loss = ddf_loss(student, teacher, boxes, boxes, indices,
                        torch.tensor(6.0), REG_MAX)
        assert loss.item() > 0.0

    def test_teacher_is_detached(self):
        student = self._stacked(torch.randn(1, 2, 4, 2 * REG_MAX + 1))
        teacher = self._stacked(torch.randn(1, 2, 4, 2 * REG_MAX + 1))
        student.requires_grad_(True)
        boxes = torch.rand(1, 2, 4) * 0.3 + 0.3
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        loss = ddf_loss(student, teacher, boxes, boxes, indices,
                        torch.tensor(2.0), REG_MAX)
        loss.backward()
        assert teacher.grad is None
        assert student.grad is not None
        assert student.grad.abs().sum() > 0


class TestDFineCriterion:
    def make_criterion(self) -> DFINECriterionV2:
        return DFINECriterionV2(
            HungarianMatcher(), reg_max=REG_MAX,
            fgl_loss_weight=0.15, ddf_loss_weight=1.5,
        )

    def test_loss_keys_and_weights(self):
        model = make_decoder(num_layers=2)
        model.train()
        out = model(make_feats(), targets=make_targets())
        outputs_list = list(out["aux_outputs"]) + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        losses = self.make_criterion()(
            outputs_list,
            make_targets(),
            dn_outputs=out["dn_aux_outputs"],
            dn_meta=out["dn_meta"],
            corners_list=out["pred_corners_list"],
            ref_boxes_list=out["ref_boxes_list"],
            dn_corners_list=out["dn_pred_corners_list"],
            dn_ref_boxes_list=out["dn_ref_boxes_list"],
        )
        for key in (
            "loss_vfl", "loss_bbox", "loss_giou", "loss_fgl",
            "loss_vfl_aux_0", "loss_fgl_aux_0", "loss_ddf_aux_0",
            "loss_vfl_dn_0", "loss_fgl_dn_0", "loss_ddf_dn_0", "loss",
        ):
            assert key in losses, key
        assert torch.isfinite(losses["loss"]).item()

    def test_matches_baseline_when_corners_absent(self):
        # Without distributional outputs the criterion must degrade to the
        # plain RT-DETRv2 loss set (validation path).
        model = make_decoder(num_layers=2)
        model.train()
        out = model(make_feats(), targets=make_targets())
        outputs_list = list(out["aux_outputs"]) + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        losses = self.make_criterion()(outputs_list, make_targets())
        assert "loss_fgl" not in losses and "loss_ddf" not in losses
        assert "loss" in losses

    def test_loss_decreases_over_steps(self):
        torch.manual_seed(0)
        model = make_decoder(num_layers=2)
        model.train()
        criterion = self.make_criterion()
        targets = make_targets()
        feats = make_feats()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        def step_loss():
            optimizer.zero_grad()
            out = model(feats, targets=targets)
            outputs_list = list(out["aux_outputs"]) + [
                {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
            ]
            losses = criterion(
                outputs_list, targets,
                dn_outputs=out["dn_aux_outputs"], dn_meta=out["dn_meta"],
                corners_list=out["pred_corners_list"],
                ref_boxes_list=out["ref_boxes_list"],
                dn_corners_list=out["dn_pred_corners_list"],
                dn_ref_boxes_list=out["dn_ref_boxes_list"],
            )
            losses["loss"].backward()
            optimizer.step()
            return float(losses["loss"].detach())

        first = sum(step_loss() for _ in range(5)) / 5
        last = sum(step_loss() for _ in range(5)) / 5
        assert last < first


class TestDFineTaskModel:
    def make_task_model(self, **overrides) -> DFINEObjectDetection:
        return DFINEObjectDetection(
            num_classes=2, image_size=(64, 64), hidden_dim=32,
            num_queries=10, dec_layers=2, num_heads=4, num_top_queries=10,
            reg_max=REG_MAX, **overrides,
        )

    def test_forward_shapes(self):
        model = self.make_task_model()
        model.eval()
        with torch.no_grad():
            logits, boxes = model(torch.randn(1, 3, 64, 64))
        assert logits.shape == (1, 10, 2)
        assert boxes.shape == (1, 10, 4)

    def test_init_args_record_reg_max(self):
        model = self.make_task_model()
        assert model.init_args["reg_max"] == REG_MAX

    def test_gradient_flow_through_backward(self):
        model = self.make_task_model()
        model.train()
        logits, boxes = model(torch.randn(1, 3, 64, 64))
        (logits.sum() + boxes.sum()).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads

    def test_postprocess_maps_to_original_pixels(self):
        model = self.make_task_model()
        model.eval()
        raw = model(torch.randn(1, 3, 64, 64))
        results = model.postprocess(raw, [{"orig_w": 128, "orig_h": 96}], threshold=0.0)
        xyxy = results[0]["bboxes"]
        assert xyxy.shape[0] > 0
        assert (xyxy[:, [0, 2]] <= 128.0 + 1e-3).all()
        assert (xyxy[:, [1, 3]] <= 96.0 + 1e-3).all()


class TestDFineTrainModel:
    def test_training_step_backward(self):
        train_model = make_train_model()
        train_model.train()
        result = train_model.training_step(make_batch(), step=0)
        assert torch.isfinite(result.loss).item()
        result.loss.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in train_model.model.decoder.dec_bbox_head.parameters()
        )

    def test_loss_log_contains_dfine_terms(self):
        train_model = make_train_model()
        train_model.train()
        result = train_model.training_step(make_batch(), step=0)
        assert "loss_fgl" in result.log_dict
        assert "loss_ddf_aux_0" in result.log_dict
        assert "loss_fgl_dn_0" in result.log_dict

    def test_is_train_model_with_watch_metric(self):
        train_model = make_train_model()
        assert train_model.watch_metric == "val_metric/map"
        assert train_model.task == "object_detection"

    def test_validation_step_plain_rtdetr_path(self):
        train_model = make_train_model()
        train_model.set_categories({0: "a", 1: "b"})
        train_model.eval()
        with torch.no_grad():
            result = train_model.validation_step(make_batch(), step=0)
        assert torch.isfinite(result.loss).item()

    def test_default_weights_follow_paper(self):
        args = DFINEObjectDetectionTrainArgs(num_classes=2)
        assert args.fgl_loss_weight == 0.15
        assert args.ddf_loss_weight == 1.5
        assert args.reg_max == 32

    def test_default_batch_and_steps_inherited(self):
        assert DFINEObjectDetectionTrainArgs.default_batch_size == 16
        assert DFINEObjectDetectionTrainArgs.default_steps == 12_000


class TestE2E:
    def test_train_object_detection_dfine(self, tmp_path):
        train_json, val_json, images_dir = make_coco_dataset(tmp_path, num_images=8)
        result = train_object_detection(
            out=tmp_path / "dfine1",
            data=data_config(train_json, val_json, images_dir),
            model="dfine/resnet18",
            steps=4,
            batch_size=2,
            num_workers=0,
            model_args={
                "image_size": (64, 64),
                "num_queries": 10,
                "num_top_queries": 10,
                "hidden_dim": 32,
                "dec_layers": 2,
                "reg_max": REG_MAX,
            },
            transform_args=transform_args(),
            trainer_args={"val_every_num_steps": 2},
        )
        assert result.steps_run == 4
        assert result.checkpoint_path and result.checkpoint_path.exists()
        assert result.best_checkpoint_path and result.best_checkpoint_path.exists()
        assert "val_metric/map" in result.best_metrics

    def test_exported_model_loads_and_predicts(self, tmp_path):
        train_json, val_json, images_dir = make_coco_dataset(tmp_path, num_images=4)
        result = train_object_detection(
            out=tmp_path / "dfine2",
            data=data_config(train_json, val_json, images_dir),
            model="dfine/resnet18",
            steps=2,
            batch_size=2,
            num_workers=0,
            model_args={
                "image_size": (64, 64),
                "num_queries": 10,
                "num_top_queries": 10,
                "hidden_dim": 32,
                "dec_layers": 2,
                "reg_max": REG_MAX,
            },
            transform_args=transform_args(),
            trainer_args={"val_every_num_steps": 2},
        )
        model = load_model(result.exported_model_path, device="cpu")
        image = create_image(tmp_path / "probe.jpg", size=(64, 64), seed=0)
        prediction = model.predict(image, threshold=0.05)
        assert set(prediction) == {"labels", "bboxes", "scores"}
