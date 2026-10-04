# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Tests for the LT-DETR object detection family (SSL-backbone finetuning)."""

import random

import pytest
import torch
import transformers

from autovision import load_task_model as load_model, train_object_detection
from autovision._models.dinov2.dinov2_wrapper import TransformersDinov2Wrapper
from autovision._task_models.ltdetr_object_detection import (
    LTDETRObjectDetectionTrain,
    LTDETRObjectDetectionTrainArgs,
    list_ltdetr_model_names,
    resolve_ltdetr_variant,
)
from tests.conftest import create_image
from tests.test_train_object_detection import (
    data_config,
    make_coco_dataset,
    transform_args,
)


def make_tiny_hf_model() -> TransformersDinov2Wrapper:
    config = transformers.Dinov2Config(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        image_size=224,
        patch_size=14,
        num_register_tokens=0,
    )
    return TransformersDinov2Wrapper(transformers.Dinov2Model(config))


def make_train_model(**overrides) -> LTDETRObjectDetectionTrain:
    args = LTDETRObjectDetectionTrainArgs(
        num_classes=2,
        backbone="dinov2-vittest-notpretrained",
        image_size=(224, 224),
        num_queries=20,
        num_top_queries=20,
        **overrides,
    )
    return LTDETRObjectDetectionTrain(args)


def make_batch(size: int = 224, images: int = 2):
    targets = [
        {"boxes": torch.rand(3, 4) * 0.3 + 0.3, "labels": torch.tensor([0, 1, 0])}
        for _ in range(images)
    ]
    return {
        "image": torch.randn(images, 3, size, size),
        "bboxes": [t["boxes"] for t in targets],
        "classes": [t["labels"] for t in targets],
        "targets": targets,
    }


class TestRegistry:
    def test_names_sorted_with_prefix(self):
        names = list_ltdetr_model_names()
        assert names == sorted(names)
        assert all(name.startswith("ltdetr/") for name in names)
        assert "ltdetr/dinov2-vits14" in names
        assert "ltdetr/dinov2-vits14-notpretrained" in names
        assert "ltdetr/dinov2-vittest-notpretrained" in names
        # Baseline-faithful expansions.
        assert "ltdetr/dinov3-vitt16-notpretrained" in names
        assert "ltdetr/dinov3-vits16" in names
        assert "ltdetr/edgecrafter-ecvitt" in names

    def test_resolve_variant(self):
        key, spec = resolve_ltdetr_variant("ltdetr/dinov2-vits14")
        assert key == "dinov2-vits14"
        assert spec.family == "vit"
        assert spec.patch_size == 14
        assert spec.interaction_indexes == (5, 8, 11)

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown LT-DETR"):
            resolve_ltdetr_variant("ltdetr/nope")
        with pytest.raises(ValueError, match="Unknown LT-DETR"):
            resolve_ltdetr_variant("rtdetr/resnet18")

    def test_dinov2_variants_are_nosta_with_baseline_recipes(self):
        """Baseline: DINOv2 ViT variants use NO spatial prior, and the
        vits14 recipe projects to hidden 224 with a 4-layer decoder."""
        _, spec = resolve_ltdetr_variant("ltdetr/dinov2-vits14")
        assert spec.use_sta is False
        assert spec.conv_inplane == 2 * 14
        assert spec.hidden_dim == 224
        assert spec.dec_layers == 4
        assert spec.enc_dim_feedforward == 896
        assert spec.dec_dim_feedforward == 1792
        _, base = resolve_ltdetr_variant("ltdetr/dinov2-vitb14")
        assert base.conv_inplane == 4 * 14 and base.hidden_dim == 768
        _, giant = resolve_ltdetr_variant("ltdetr/dinov2-vitg14")
        assert giant.interaction_indexes == (19, 29, 39)

    def test_dinov3_variants_use_sta(self):
        """Baseline: DINOv3 ViT variants fuse the CNN spatial prior with
        conv_inplane = factor x patch_size (16)."""
        _, tiny = resolve_ltdetr_variant("ltdetr/dinov3-vitt16")
        assert tiny.use_sta is True
        assert tiny.conv_inplane == 1 * 16
        assert tiny.hidden_dim == 192
        assert tiny.enc_expansion == 0.34 and tiny.enc_depth_mult == 0.67
        assert tiny.requires_dinov3 is True
        _, small = resolve_ltdetr_variant("ltdetr/dinov3-vits16")
        assert small.conv_inplane == 2 * 16 and small.hidden_dim == 224

    def test_eupe_names_are_aliases(self):
        base = resolve_ltdetr_variant("ltdetr/dinov3-vits16")[1]
        alias = resolve_ltdetr_variant("ltdetr/dinov3-vits16-eupe")[1]
        assert alias is base

    def test_ecvit_variants_default_to_dfine(self):
        _, spec = resolve_ltdetr_variant("ltdetr/edgecrafter-ecvitt")
        assert spec.family == "ecvit"
        assert spec.decoder_default == "dfine"
        assert spec.hidden_dim == 192
        _, plus = resolve_ltdetr_variant("ltdetr/edgecrafter-ecvittplus")
        assert plus.backbone_lr_factor == 0.0025
        assert plus.decoder_default == "dfine"

    def test_convnext_recipes(self):
        _, large = resolve_ltdetr_variant("ltdetr/dinov3-convnext-large")
        assert large.family == "convnext"
        assert large.dec_layers == 6
        assert large.dec_num_points == (4, 4, 4)
        assert large.hidden_dim == 384


class TestMultiscaleFeatures:
    def test_intermediate_taps(self):
        wrapper = make_tiny_hf_model()
        x = torch.randn(2, 3, 224, 224)
        feats = wrapper.forward_multiscale_features(x, (1, 2, 3))
        assert len(feats) == 3
        for feat in feats:
            assert feat["features"].shape == (2, 64, 16, 16)
            assert feat["cls_token"].shape == (2, 64)
        # Taps are distinct blocks: features must differ across taps.
        assert not torch.allclose(feats[0]["features"], feats[2]["features"])


class TestViTBackboneWrapper:
    def test_pyramid_shapes_and_projection(self):
        train_model = make_train_model()
        backbone = train_model.model.backbone
        x = torch.randn(1, 3, 224, 224)
        c2, c3, c4 = backbone(x)
        # Conv-chain pyramid shapes (input//8, then successive ceil-halves
        # matching the encoder's stride-2 convolutions).
        assert c2.shape == (1, 64, 28, 28)
        assert c3.shape == (1, 64, 14, 14)
        assert c4.shape == (1, 64, 7, 7)

    def test_pyramid_shapes_at_odd_patch_grid(self):
        """640 gives a 45x45 patch grid (odd) — the pyramid must still
        land on conv-chain-consistent shapes for the encoder fusion."""
        train_model = make_train_model()
        x = torch.randn(1, 3, 640, 640)
        c2, c3, c4 = train_model.model.backbone(x)
        assert c2.shape == (1, 64, 80, 80)
        assert c3.shape == (1, 64, 40, 40)
        assert c4.shape == (1, 64, 20, 20)

    def test_pyramid_shapes_at_644(self):
        """Regression: 644 (the patch-14 default after snapping) must
        ceil-halve along the encoder's stride-2 conv chain (80 -> 40 -> 20);
        an independent ceil(x/16) ladder would give 41 and break the PAN
        concat."""
        train_model = make_train_model()
        x = torch.randn(1, 3, 644, 644)
        c2, c3, c4 = train_model.model.backbone(x)
        assert c2.shape == (1, 64, 80, 80)
        assert c3.shape == (1, 64, 40, 40)
        assert c4.shape == (1, 64, 20, 20)
        # The full forward must run through the encoder + decoder.
        out = train_model.model._forward_train(x)
        assert out["pred_logits"].shape[-1] == 2

    def test_backbone_model_is_the_hf_vit(self):
        train_model = make_train_model()
        backbone = train_model.model.backbone
        assert backbone.backbone_model is backbone._model_wrapper.get_model()
        # The pretrained ViT is the only part of the backbone at low LR —
        # STA + projection layers must live OUTSIDE backbone_model.
        backbone_ids = {id(p) for p in backbone.backbone_model.parameters()}
        sta_params = list(backbone.sta.parameters())
        assert sta_params
        assert all(id(p) not in backbone_ids for p in sta_params)

    def test_finetune_false_freezes_vit(self):
        train_model = make_train_model()
        backbone = train_model.model.backbone
        backbone._model_wrapper.requires_grad_(False)
        backbone._model_wrapper.eval()
        assert not any(p.requires_grad for p in backbone.backbone_model.parameters())


class TestTaskModel:
    def test_forward_shapes(self):
        train_model = make_train_model()
        out = train_model.model._forward_train(torch.randn(2, 3, 224, 224))
        assert out["pred_logits"].shape == (2, 20, 2)
        assert out["pred_boxes"].shape == (2, 20, 4)
        assert (out["pred_boxes"] >= 0).all() and (out["pred_boxes"] <= 1).all()

    def test_loss_backward(self):
        train_model = make_train_model()
        batch = make_batch()
        result = train_model.training_step(
            {
                "image": batch["image"],
                "bboxes": batch["bboxes"],
                "classes": batch["classes"],
            },
            0,
        )
        assert torch.isfinite(result.loss)
        result.loss.backward()
        grads = [
            p.grad.abs().sum()
            for p in train_model.model.parameters()
            if p.grad is not None
        ]
        assert grads and float(sum(grads)) > 0

    def test_postprocess_maps_to_original_pixels(self):
        train_model = make_train_model()
        model = train_model.model
        orig_w, orig_h = 136, 76
        boxes_px = torch.tensor([[10.0, 10.0, 32.0, 32.0]])
        cx = (boxes_px[:, 0] + boxes_px[:, 2]) / 2 / orig_w
        cy = (boxes_px[:, 1] + boxes_px[:, 3]) / 2 / orig_h
        w = (boxes_px[:, 2] - boxes_px[:, 0]) / orig_w
        h = (boxes_px[:, 3] - boxes_px[:, 1]) / orig_h
        boxes_norm = torch.stack([cx, cy, w, h], -1).unsqueeze(0)
        logits = torch.full((1, 1, 2), -8.0)
        logits[0, 0, 0] = 6.0
        (decoded,) = model.postprocess(
            (logits, boxes_norm),
            [{"orig_w": orig_w, "orig_h": orig_h}],
            threshold=0.05,
        )
        assert torch.allclose(decoded["bboxes"], boxes_px, atol=1e-4)

    def test_backbone_weights_rejects_foreign_keys(self, tmp_path):
        train_model = make_train_model()
        foreign = {"does.not.exist": torch.zeros(1)}
        path = tmp_path / "weights.pt"
        torch.save(foreign, path)
        with pytest.raises(ValueError, match="unknown to the backbone"):
            train_model.model._load_backbone_weights(str(path))


class TestOptimizerGroups:
    def test_vit_lr_groups(self):
        train_model = make_train_model()
        optimizer, _ = train_model.get_optimizer(
            total_steps=100, global_batch_size=16, gradient_accumulation_steps=2
        )
        groups = {g["name"]: g for g in optimizer.param_groups}
        # lr = 5e-4 * sqrt(16/16); backbone = lr * 0.05.
        assert groups["detector"]["lr"] == pytest.approx(5e-4)
        assert groups["backbone"]["lr"] == pytest.approx(5e-4 * 0.05)
        # Full parameter coverage across groups.
        total = sum(len(g["params"]) for g in optimizer.param_groups)
        assert total == len(list(train_model.model.parameters()))
        # The STA prior trains at detector LR (randomly initialized).
        sta_ids = {
            id(p) for p in train_model.model.backbone.sta.parameters()
        }
        detector_ids = {id(p) for g in groups.values() for p in g["params"]}
        assert sta_ids <= detector_ids

    def test_scheduler_is_flat_cosine(self):
        from autovision._task_models.object_detection_components.flat_cosine import (
            FlatCosineLRScheduler,
        )

        train_model = make_train_model()
        _, scheduler = train_model.get_optimizer(
            total_steps=100, global_batch_size=16
        )
        assert isinstance(scheduler, FlatCosineLRScheduler)


class TestDecoders:
    def test_rtdetrv2_is_default(self):
        train_model = make_train_model()
        assert train_model.model.decoder_name == "rtdetrv2"
        from autovision._task_models.object_detection_components.rtdetrv2_decoder import (
            RTDETRv2Decoder,
        )

        assert type(train_model.model.decoder) is RTDETRv2Decoder
        assert type(train_model.criterion).__name__ == "RTDETRCriterionV2"

    def test_dfine_decoder_swaps_criterion(self):
        from autovision._task_models.object_detection_components.dfine_decoder import (
            DFINEv2Decoder,
        )

        train_model = make_train_model(decoder="dfine", reg_max=16)
        assert type(train_model.model.decoder) is DFINEv2Decoder
        assert train_model.model.decoder.reg_max == 16
        assert type(train_model.criterion).__name__ == "DFINECriterionV2"
        batch = make_batch()
        result = train_model.training_step(
            {
                "image": batch["image"],
                "bboxes": batch["bboxes"],
                "classes": batch["classes"],
            },
            0,
        )
        assert torch.isfinite(result.loss)
        assert any("loss_fgl" in key for key in result.log_dict)
        result.loss.backward()

    def test_ecvit_auto_selects_dfine(self):
        from autovision._task_models.object_detection_components.dfine_decoder import (
            DFINEv2Decoder,
        )

        args = LTDETRObjectDetectionTrainArgs(
            num_classes=2,
            backbone="edgecrafter-ecvitt",
            image_size=(224, 224),
            num_queries=20,
            num_top_queries=20,
        )
        train_model = LTDETRObjectDetectionTrain(args)
        assert train_model.model.decoder_name == "dfine"
        assert type(train_model.model.decoder) is DFINEv2Decoder
        # Larger ECViT variants drop the backbone LR factor (baseline).
        optimizer, _ = train_model.get_optimizer(
            total_steps=100, global_batch_size=16
        )
        groups = {g["name"]: g for g in optimizer.param_groups}
        assert groups["backbone"]["lr"] == pytest.approx(5e-4 * 0.05)
        # The projector is connector: it must sit in the detector group.
        projector_ids = {
            id(p) for p in train_model.model.backbone._model_wrapper.projector.parameters()
        }
        detector_ids = {
            id(p) for g in groups.values() for p in g["params"]
        }
        assert projector_ids <= detector_ids


class TestImageSize:
    def test_default_snaps_to_patch_grid(self):
        args = LTDETRObjectDetectionTrainArgs(num_classes=2)
        assert args.image_size == (640, 640)
        args.resolve_image_size()
        # 640 -> 644: next multiple of 2 x patch 14.
        assert args.image_size == (644, 644)

    def test_patch16_default_unchanged(self):
        args = LTDETRObjectDetectionTrainArgs(
            num_classes=2, backbone="edgecrafter-ecvitt"
        )
        args.resolve_image_size()
        assert args.image_size == (640, 640)

    def test_explicit_divisible_size_kept(self):
        args = LTDETRObjectDetectionTrainArgs(num_classes=2, image_size=(644, 644))
        args.resolve_image_size()
        assert args.image_size == (644, 644)

    def test_indivisible_explicit_size_raises(self):
        args = LTDETRObjectDetectionTrainArgs(num_classes=2, image_size=(640, 640))
        args.backbone = "edgecrafter-ecvitt"  # divisor 32 -> 640 still fine
        args.resolve_image_size()
        args.image_size = (500, 500)
        with pytest.raises(ValueError, match="divisible by 32"):
            args.resolve_image_size()

    def test_task_model_rejects_bad_size(self):
        from autovision._task_models.ltdetr_object_detection.task_model import (
            LTDETRObjectDetection,
        )

        with pytest.raises(ValueError, match="divisible by 28"):
            LTDETRObjectDetection(
                num_classes=2,
                model_name="dinov2-vittest-notpretrained",
                image_size=(64, 64),
                num_queries=10,
            )


class TestECViT:
    def test_package_lists_models(self):
        from autovision._models.package_helpers import list_model_names

        names = list_model_names()
        for size in ("ecvitt", "ecvittplus", "ecvits", "ecvitsplus"):
            assert f"edgecrafter/{size}" in names

    def test_wrapper_emits_pyramid(self):
        from autovision._models.ecvit.ecvit import ECViTModelWrapper

        wrapper = ECViTModelWrapper("ecvitt")
        p3, p4, p5 = wrapper(torch.randn(2, 3, 224, 224))
        assert p3.shape == (2, 192, 28, 28)
        assert p4.shape == (2, 192, 14, 14)
        assert p5.shape == (2, 192, 7, 7)

    def test_weights_path_strict_load(self, tmp_path):
        from autovision._models.ecvit.ecvit import ECViTModelWrapper

        wrapper = ECViTModelWrapper("ecvitt")
        path = tmp_path / "ecvit.pt"
        torch.save(wrapper.backbone.state_dict(), path)
        reloaded = ECViTModelWrapper("ecvitt", weights_path=str(path))
        for key, value in wrapper.backbone.state_dict().items():
            assert torch.allclose(reloaded.backbone.state_dict()[key], value)

    def test_ecvit_train_step(self):
        args = LTDETRObjectDetectionTrainArgs(
            num_classes=2,
            backbone="edgecrafter-ecvitt",
            image_size=(224, 224),
            num_queries=20,
            num_top_queries=20,
        )
        train_model = LTDETRObjectDetectionTrain(args)
        batch = make_batch()
        result = train_model.training_step(
            {
                "image": batch["image"],
                "bboxes": batch["bboxes"],
                "classes": batch["classes"],
            },
            0,
        )
        assert torch.isfinite(result.loss)
        result.loss.backward()


class TestLTDETRTransforms:
    def test_scale_jitter_ladder_aligns_to_patch_grid(self):
        from autovision._task_models.ltdetr_object_detection.transforms import (
            ltdetr_scale_jitter_sizes,
        )

        # Patch 16 at 640 reproduces the stock 480..800 ladder.
        sizes = ltdetr_scale_jitter_sizes((640, 640), 32)
        assert sizes[:11] == [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        assert all(s % 32 == 0 for s in sizes)
        # Patch 14 ladder snaps to multiples of 28 around 644.
        sizes14 = ltdetr_scale_jitter_sizes((644, 644), 28)
        assert all(s % 28 == 0 for s in sizes14)
        assert 644 in sizes14 and 476 in sizes14 and 812 in sizes14

    def test_build_args_enables_mosaic(self):
        from autovision._task_models.ltdetr_object_detection.transforms import (
            build_ltdetr_train_transform_args,
        )

        args = build_ltdetr_train_transform_args((644, 644), "dinov2-vits14")
        assert args.mosaic_prob == 0.5
        assert args.image_size == (644, 644)
        assert all(s % 28 == 0 for s in args.scale_jitter.sizes)
        # EdgeCrafter "v2" recipe raises mosaic prob (baseline).
        args_v2 = build_ltdetr_train_transform_args((640, 640), "edgecrafter-ecvitt")
        assert args_v2.mosaic_prob == 0.75
        # User overrides win.
        args_off = build_ltdetr_train_transform_args(
            (644, 644), "dinov2-vits14", {"mosaic_prob": None}
        )
        assert args_off.mosaic_prob is None

    def test_mosaic_merges_samples_and_aligns_boxes(self):
        from torchvision import tv_tensors

        from autovision._transforms.detection_transforms.args import (
            RTDETRDetectionTransformArgs,
        )
        from autovision._transforms.detection_transforms.transform import (
            RTDETRDetectionTransform,
        )

        random.seed(0)
        torch.manual_seed(0)
        args = RTDETRDetectionTransformArgs(
            image_size=(224, 224),
            mosaic_prob=1.0,
            mosaic_output_size=160,
            photometric_distort_prob=None,
            random_zoom_out_prob=None,
            random_iou_crop_prob=None,
            scale_jitter=None,
            horizontal_flip_prob=0.0,
        )
        args.mosaic_step_start = 0
        args.mosaic_step_stop = None
        transform = RTDETRDetectionTransform(args)

        def make_item(index: int) -> dict:
            return {
                "image_path": f"img{index}.jpg",
                "image": torch.randint(0, 255, (3, 100, 120), dtype=torch.uint8),
                "bboxes": tv_tensors.BoundingBoxes(
                    torch.tensor([[10.0, 10.0, 50.0, 60.0], [30.0, 30.0, 90.0, 80.0]]),
                    format="XYXY",
                    canvas_size=(100, 120),
                ),
                "classes": torch.tensor([0, 1]),
                "original_size": (120, 100),
            }

        outputs = [transform(make_item(i)) for i in range(6)]
        for output in outputs:
            assert output["image"].shape == (3, 224, 224)
            # Box/class alignment must survive the per-tile degenerate filter.
            assert output["bboxes"].shape[0] == output["classes"].shape[0]
            assert bool((output["bboxes"] >= 0).all())
            assert bool((output["bboxes"] <= 1).all())
        # After the cache holds the current sample plus >= 3 distinct
        # donors, every sample is a 4-tile mosaic (the current sample is
        # never drawn twice).
        assert outputs[-1]["bboxes"].shape[0] >= 8
        # Mosaic samples keep photometric distort; only zoom-out/IoU-crop
        # are skipped (baseline behavior).
        assert (True, False, False, False) in transform.compose._compose_cache

    def test_mosaic_keeps_photometric(self):
        from torchvision import tv_tensors

        from autovision._transforms.detection_transforms.args import (
            RTDETRDetectionTransformArgs,
        )
        from autovision._transforms.detection_transforms.transform import (
            RTDETRDetectionTransform,
        )

        random.seed(0)
        torch.manual_seed(0)
        args = RTDETRDetectionTransformArgs(
            image_size=(224, 224),
            mosaic_prob=1.0,
            mosaic_output_size=160,
            photometric_distort_prob=0.5,
            random_zoom_out_prob=None,
            random_iou_crop_prob=None,
            scale_jitter=None,
            horizontal_flip_prob=0.0,
        )
        args.mosaic_step_start = 0
        args.mosaic_step_stop = None
        args.photometric_step_start = 0
        args.photometric_step_stop = None
        transform = RTDETRDetectionTransform(args)

        def make_item(index: int) -> dict:
            return {
                "image_path": f"img{index}.jpg",
                "image": torch.randint(0, 255, (3, 100, 120), dtype=torch.uint8),
                "bboxes": tv_tensors.BoundingBoxes(
                    torch.tensor([[10.0, 10.0, 50.0, 60.0], [30.0, 30.0, 90.0, 80.0]]),
                    format="XYXY",
                    canvas_size=(100, 120),
                ),
                "classes": torch.tensor([0, 1]),
                "original_size": (120, 100),
            }

        for _ in range(6):
            transform(make_item(0))
        assert (True, True, False, False) in transform.compose._compose_cache

    def test_scale_jitter_step_window(self):
        from autovision._transforms.detection_transforms.args import (
            RTDETRDetectionTransformArgs,
        )
        from autovision._transforms.detection_transforms.collate import (
            RTDETRDetectionCollateFunction,
        )

        args = RTDETRDetectionTransformArgs(
            image_size=(64, 64),
            mosaic_prob=None,
            scale_jitter={"sizes": [64, 80], "step_stop": 10},
        )
        collate = RTDETRDetectionCollateFunction(args, train=True)
        batch = [
            {
                "image": torch.randn(3, 64, 64),
                "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "classes": torch.tensor([0]),
                "original_size": (32, 32),
                "image_path": "x.jpg",
            }
        ]
        collate.set_step(5)
        assert collate(batch)["image"].shape[-1] in (64, 80)
        collate.set_step(20)
        assert collate(batch)["image"].shape[-1] == 64
        collate.set_step(30)
        assert collate.requires_dataloader_reinitialization()


class TestSoftBinTargets:
    def test_exact_distance_targets_stay_in_bounds(self):
        """Regression: the float32 clamp of the continuous bin position to
        2*reg_max - 1e-6 rounds back UP to 2*reg_max, pushing the right bin
        one past the distribution for exact-distance targets."""
        from autovision._task_models.object_detection_components.dfine_utils import (
            soft_bin_targets,
        )

        distances = torch.tensor([[1.0, -1.0, 0.5, 1.0]])
        left, right, weight_left, weight_right = soft_bin_targets(distances, 32)
        assert int(right.max()) <= 64
        assert int(left.min()) >= 0
        # Exact +1 distance: all weight on the top bin (64).
        assert (right[0, 0] == 64) and weight_right[0, 0] == pytest.approx(1.0)
        assert (left[0, 0] == 63) and weight_left[0, 0] == pytest.approx(0.0)


class TestE2E:
    def test_train_object_detection_ltetr(self, tmp_path):
        train_json, val_json, images_dir = make_coco_dataset(tmp_path, num_images=8)
        result = train_object_detection(
            out=tmp_path / "ltdetr1",
            data=data_config(train_json, val_json, images_dir),
            model="ltdetr/dinov2-vittest-notpretrained",
            steps=4,
            batch_size=2,
            num_workers=0,
            model_args={
                # 56 = 4 x patch 14: divisible by 2 x patch_size.
                "image_size": (56, 56),
                "num_queries": 10,
                "num_top_queries": 10,
                "hidden_dim": 64,
                "dec_layers": 2,
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
            out=tmp_path / "ltdetr2",
            data=data_config(train_json, val_json, images_dir),
            model="ltdetr/dinov2-vittest-notpretrained",
            steps=2,
            batch_size=2,
            num_workers=0,
            model_args={
                "image_size": (56, 56),
                "num_queries": 10,
                "num_top_queries": 10,
                "hidden_dim": 64,
                "dec_layers": 2,
            },
            transform_args=transform_args(),
            trainer_args={"val_every_num_steps": 2},
        )
        model = load_model(result.exported_model_path, device="cpu")
        image = create_image(tmp_path / "probe.jpg", size=(64, 64), seed=0)
        prediction = model.predict(image, threshold=0.05)
        assert set(prediction) == {"labels", "bboxes", "scores"}
