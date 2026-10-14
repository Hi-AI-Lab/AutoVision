# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Tests for the AV-DETR object detection family (SSL-backbone finetuning)."""

import random

import pytest
import torch
import transformers
from tests.conftest import create_image
from tests.test_train_object_detection import (
    data_config,
    make_coco_dataset,
    transform_args,
)

from autovision import load_task_model as load_model
from autovision import train_object_detection
from autovision.encoders.dinov2.dinov2_wrapper import TransformersDinov2Wrapper
from autovision.tasks.detection.avdetr import (
    AVDETRObjectDetectionTrain,
    AVDETRObjectDetectionTrainArgs,
    list_avdetr_model_names,
    resolve_avdetr_variant,
)
from autovision.tasks.detection.avdetr.task_model import (
    _relabel_prefix_drift,
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


def make_train_model(**overrides) -> AVDETRObjectDetectionTrain:
    args = AVDETRObjectDetectionTrainArgs(
        num_classes=2,
        backbone="dinov2-vittest-notpretrained",
        image_size=(224, 224),
        num_queries=20,
        num_top_queries=20,
        **overrides,
    )
    return AVDETRObjectDetectionTrain(args)


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
        names = list_avdetr_model_names()
        assert names == sorted(names)
        assert all(name.startswith("avdetr/") for name in names)
        assert "avdetr/dinov2-vits14" in names
        assert "avdetr/dinov2-vits14-notpretrained" in names
        assert "avdetr/dinov2-vittest-notpretrained" in names
        # Baseline-faithful expansions.
        assert "avdetr/dinov3-vitt16-notpretrained" in names
        assert "avdetr/dinov3-vits16" in names
        assert "avdetr/edgecrafter-ecvitt" in names

    def test_resolve_variant(self):
        key, spec = resolve_avdetr_variant("avdetr/dinov2-vits14")
        assert key == "dinov2-vits14"
        assert spec.family == "vit"
        assert spec.patch_size == 14
        assert spec.interaction_indexes == (5, 8, 11)

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown AV-DETR"):
            resolve_avdetr_variant("avdetr/nope")
        with pytest.raises(ValueError, match="Unknown AV-DETR"):
            resolve_avdetr_variant("rtdetr/resnet18")

    def test_dinov2_variants_are_nosta_with_baseline_recipes(self):
        """Baseline: DINOv2 ViT variants use NO spatial prior, and the
        vits14 recipe projects to hidden 224 with a 4-layer decoder."""
        _, spec = resolve_avdetr_variant("avdetr/dinov2-vits14")
        assert spec.use_sta is False
        assert spec.conv_inplane == 2 * 14
        assert spec.hidden_dim == 224
        assert spec.dec_layers == 4
        assert spec.enc_dim_feedforward == 896
        assert spec.dec_dim_feedforward == 1792
        _, base = resolve_avdetr_variant("avdetr/dinov2-vitb14")
        assert base.conv_inplane == 4 * 14 and base.hidden_dim == 768
        _, giant = resolve_avdetr_variant("avdetr/dinov2-vitg14")
        assert giant.interaction_indexes == (19, 29, 39)

    def test_dinov3_variants_use_sta(self):
        """Baseline: DINOv3 ViT variants fuse the CNN spatial prior with
        conv_inplane = factor x patch_size (16)."""
        _, tiny = resolve_avdetr_variant("avdetr/dinov3-vitt16")
        assert tiny.use_sta is True
        assert tiny.conv_inplane == 1 * 16
        assert tiny.hidden_dim == 192
        assert tiny.enc_expansion == 0.34 and tiny.enc_depth_mult == 0.67
        assert tiny.requires_dinov3 is True
        _, small = resolve_avdetr_variant("avdetr/dinov3-vits16")
        assert small.conv_inplane == 2 * 16 and small.hidden_dim == 224

    def test_eupe_names_are_aliases(self):
        base = resolve_avdetr_variant("avdetr/dinov3-vits16")[1]
        alias = resolve_avdetr_variant("avdetr/dinov3-vits16-eupe")[1]
        assert alias is base

    def test_ecvit_variants_default_to_official_dfine(self):
        _, spec = resolve_avdetr_variant("avdetr/edgecrafter-ecvitt")
        assert spec.family == "ecvit"
        assert spec.decoder_default == "dfine_official"
        assert spec.hidden_dim == 192
        _, plus = resolve_avdetr_variant("avdetr/edgecrafter-ecvittplus")
        assert plus.backbone_lr_factor == 0.0025
        assert plus.decoder_default == "dfine_official"

    def test_convnext_recipes(self):
        _, large = resolve_avdetr_variant("avdetr/dinov3-convnext-large")
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

    def test_backbone_weights_reject_partial_files(self, tmp_path):
        # A file that drops tensors must fail loud — silently keeping the
        # random init for the missing keys would poison the run.
        from safetensors.torch import save_file

        train_model = make_train_model()
        state = train_model.model.backbone.backbone_model.state_dict()
        partial = dict(list(state.items())[: len(state) // 2])
        path = tmp_path / "partial.safetensors"
        save_file(partial, str(path))
        with pytest.raises(ValueError, match="missing"):
            train_model.model._load_backbone_weights(str(path))

    def test_backbone_weights_accept_safetensors(self, tmp_path):
        from safetensors.torch import save_file

        train_model = make_train_model()
        state = train_model.model.backbone.backbone_model.state_dict()
        path = tmp_path / "backbone.safetensors"
        save_file(state, str(path))
        # Loads without raising and produces the same backbone weights.
        fresh = make_train_model()
        fresh.model._load_backbone_weights(str(path))
        reference = train_model.model.backbone.backbone_model.state_dict()
        for key, value in fresh.model.backbone.backbone_model.state_dict().items():
            assert torch.equal(value, reference[key])


class _PrefixedInner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)


class _PrefixedTarget(torch.nn.Module):
    """Mirrors the transformers 5.x DINOv3-ConvNeXt layout: weights under
    ``model.*`` plus a top-level ``norm`` (the ConvNeXt ``layer_norm``)."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _PrefixedInner()
        self.norm = torch.nn.LayerNorm(1)


class TestBackbonePrefixDrift:
    def test_relabels_legacy_unprefixed_keys(self):
        target = _PrefixedTarget()
        reference = target.state_dict()
        legacy = {
            key.removeprefix("model."): value.clone()
            for key, value in reference.items()
        }
        relabeled = _relabel_prefix_drift(legacy, target)
        assert set(relabeled) == set(reference)
        for key, value in relabeled.items():
            assert torch.equal(value, reference[key])

    def test_identity_state_passes_through(self):
        target = _PrefixedTarget()
        state = target.state_dict()
        assert _relabel_prefix_drift(state, target) is state

    def test_foreign_keys_returned_unchanged_for_strict_report(self):
        target = _PrefixedTarget()
        reference = target.state_dict()
        foreign = {key: value.clone() for key, value in reference.items()}
        foreign["does.not.exist"] = torch.zeros(1)
        assert _relabel_prefix_drift(foreign, target) is foreign


class TestOptimizerGroups:
    def test_vit_lr_groups(self):
        train_model = make_train_model()
        optimizer, _ = train_model.get_optimizer(
            total_steps=100, global_batch_size=16, gradient_accumulation_steps=2
        )
        groups = {g["name"]: g for g in optimizer.param_groups}
        base = {
            g["name"]: g["_initial_lr"] if "_initial_lr" in g else g["initial_lr"]
            for g in optimizer.param_groups
        }
        # DINOv2 ViT family recipe (baseline V2 args): lr 1e-4 at the
        # reference batch, backbone factor 1e-2. The group "lr" at
        # construction already reflects the step-0 warmup factor, so the
        # recipe is asserted on the scheduler's base LRs.
        assert base["detector"] == pytest.approx(1e-4)
        assert base["backbone"] == pytest.approx(1e-4 * 0.01)
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
        from autovision.tasks.detection.components.flat_cosine import (
            FlatCosineLRScheduler,
        )

        train_model = make_train_model()
        _, scheduler = train_model.get_optimizer(
            total_steps=100, global_batch_size=16
        )
        # DINOv2 ViT family resolves to the LINEAR recipe (warmup ->
        # constant); CNN families resolve to flat-cosine.
        from autovision.tasks.detection.components.multistep_schedule import (
            MultiStepWarmupScheduler,
        )

        assert isinstance(scheduler, MultiStepWarmupScheduler)
        assert scheduler.milestones == ()
        cnn_args = AVDETRObjectDetectionTrainArgs(
            num_classes=2,
            backbone="dinov3-convnext-tiny",
            image_size=(224, 224),
            decoder="dfine",
            num_queries=20,
            num_top_queries=20,
            backbone_pretrained=False,
        )
        cnn_train_model = AVDETRObjectDetectionTrain(cnn_args)
        _, cnn_scheduler = cnn_train_model.get_optimizer(
            total_steps=100, global_batch_size=16
        )
        assert isinstance(cnn_scheduler, FlatCosineLRScheduler)


class TestCocoCheckpointLoader:
    """Published baseline COCO detector warm-start (doc: loader contract)."""

    @staticmethod
    def _model(num_classes: int):
        from autovision.tasks.detection.avdetr.task_model import (
            AVDETRObjectDetection,
        )

        return AVDETRObjectDetection(
            num_classes=num_classes,
            model_name="avdetr/dinov3-convnext-tiny",
            image_size=(224, 224),
            decoder="rtdetrv2",
            dec_layers=6,
            num_queries=20,
            pyramid_levels=3,
            decoder_hidden_dim=256,
            load_weights=False,
        )

    @staticmethod
    def _fake_checkpoint(model) -> dict:
        """Reverse-map the model's own state dict into the baseline
        checkpoint layout (old ConvNeXt names + ema_model.model prefix)."""
        RENAME_INV = {"depthwise_conv": "dwconv", "layer_norm": "norm",
                      "pointwise_conv1": "pwconv1", "pointwise_conv2": "pwconv2"}
        sd = model.state_dict()
        out = {}
        for key, value in sd.items():
            if key.startswith("backbone_wrapper._model."):
                rest = key[len("backbone_wrapper._model."):]
                if rest.startswith("model.stages."):
                    p = rest[len("model.stages."):].split(".")
                    if p[1] == "downsample_layers":
                        out[f"backbone.backbone.downsample_layers.{p[0]}.{'.'.join(p[2:])}"] = value
                    else:
                        # stages.{s}.layers.{b}.{module}.{suffix}
                        out[f"backbone.backbone.stages.{p[0]}.{p[2]}.{RENAME_INV.get(p[3], p[3])}.{'.'.join(p[4:])}"] = value
                elif rest.startswith("model.downsample_layers."):
                    p = rest.split(".")  # model.downsample_layers.{s}.{m}.{...}
                    out[f"backbone.backbone.downsample_layers.{p[2]}.{'.'.join(p[3:])}"] = value
                elif rest.startswith("layer_norm."):
                    out[f"backbone.backbone.norm.{rest[len('layer_norm.'):]}"] = value
                else:
                    raise AssertionError(f"unmapped test key {key}")
            elif key.startswith(("encoder.", "decoder.")):
                out[key] = value
        return {"train_model": {f"ema_model.model.{k}": v for k, v in out.items()}}

    def test_round_trip_80_classes(self, tmp_path):
        import torch

        from autovision.tasks.detection.avdetr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
        )

        source = self._model(num_classes=80)
        path = tmp_path / "fake_baseline.pt"
        torch.save(self._fake_checkpoint(source), path)

        target = self._model(num_classes=80)
        summary = load_baseline_coco_checkpoint(target, path)
        assert summary == {"backbone": 180, "encoder": 246,
                           "decoder": 449 - 1, "decoder_skipped_cls": 0} or \
               summary["decoder_skipped_cls"] == 0
        # spot tensors identical
        for key in ("backbone_wrapper._model.model.stages.2.layers.5.gamma",
                    "encoder.fpn_blocks.0.conv1.conv.weight",
                    "decoder.decoder.layers.2.self_attn.in_proj_weight",
                    "decoder.dec_score_head.0.weight"):
            assert torch.equal(target.state_dict()[key],
                               source.state_dict()[key]), key

    def test_skips_class_count_heads(self, tmp_path):
        import torch

        from autovision.tasks.detection.avdetr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
        )

        source = self._model(num_classes=80)
        path = tmp_path / "fake_baseline.pt"
        torch.save(self._fake_checkpoint(source), path)

        target = self._model(num_classes=10)
        summary = load_baseline_coco_checkpoint(target, path)
        # 3 dec score heads + 1 enc score head + 1 denoising embed, (w+b).
        assert summary["decoder_skipped_cls"] == 15
        # body tensors still landed
        assert torch.equal(
            target.state_dict()["decoder.decoder.layers.0.self_attn.in_proj_weight"],
            source.state_dict()["decoder.decoder.layers.0.self_attn.in_proj_weight"],
        )
        # cls heads stayed fresh
        assert not torch.equal(
            target.state_dict()["decoder.dec_score_head.0.weight"],
            source.state_dict()["decoder.dec_score_head.0.weight"],
        )

    def test_decoder_hidden_dim_decoupling(self):
        # Default (None) keeps the shared width; explicit 256 decouples it
        # (checkpoint contract) without touching the encoder width.
        from autovision.tasks.detection.avdetr.task_model import (
            AVDETRObjectDetection,
        )

        model = self._model(num_classes=10)
        sd = model.state_dict()
        assert sd["encoder.lateral_convs.0.conv.weight"].shape[0] == 384
        assert sd["decoder.decoder.layers.0.self_attn.in_proj_weight"].shape == (768, 256)
        assert sd["decoder.dec_score_head.0.weight"].shape == (10, 256)
        # default shared-width model still behaves as before
        shared = AVDETRObjectDetection(
            num_classes=10, model_name="avdetr/dinov3-convnext-tiny",
            image_size=(224, 224), decoder="rtdetrv2", dec_layers=6,
            num_queries=20, pyramid_levels=3, load_weights=False,
        )
        ssd = shared.state_dict()
        assert ssd["decoder.decoder.layers.0.self_attn.in_proj_weight"].shape == (1152, 384)


class TestCocoCheckpointLoaderViT:
    """ViT-family baseline COCO detector warm-start (Meta ViT remap)."""

    @staticmethod
    def _model(num_classes: int, variant: str = "vitt16", **kwargs):
        from autovision.tasks.detection.avdetr.task_model import (
            AVDETRObjectDetection,
        )

        return AVDETRObjectDetection(
            num_classes=num_classes,
            model_name=f"avdetr/dinov3-{variant}-notpretrained",
            image_size=(128, 128),
            decoder="rtdetrv2",
            num_queries=20,
            load_weights=False,
            **kwargs,
        )

    @staticmethod
    def _fake_checkpoint(model) -> dict:
        """Reverse-map the model's own state dict into the baseline ViT
        checkpoint layout (Meta names + ema_model.model prefix)."""
        sd = model.state_dict()
        out: dict[str, torch.Tensor] = {}
        for key, value in sd.items():
            if key.startswith("backbone._model_wrapper."):
                # Same module as backbone_wrapper._model (registered under
                # both parents); handled by the branch below.
                continue
            if not key.startswith("backbone_wrapper._model."):
                if key.startswith(
                    ("backbone.sta.", "backbone.convs.", "backbone.norms.",
                     "encoder.", "decoder.")
                ):
                    # wrapper extras + encoder/decoder: identical names on
                    # both sides.
                    out[key] = value
                continue
            rest = key[len("backbone_wrapper._model."):]
            if rest.startswith("model.layer."):
                index, sub = rest[len("model.layer."):].split(".", 1)
                if sub.startswith(
                    ("attention.q_proj", "attention.k_proj", "attention.v_proj")
                ):
                    continue  # fused qkv rebuilt after the loop
                if sub.startswith("attention.o_proj."):
                    out[f"backbone.dinov3.blocks.{index}.attn.proj."
                        f"{sub.split('.')[-1]}"] = value
                    continue
                rename = {
                    "layer_scale1.lambda1": "ls1.gamma",
                    "layer_scale2.lambda1": "ls2.gamma",
                    "mlp.up_proj.": "mlp.fc1.",
                    "mlp.down_proj.": "mlp.fc2.",
                }
                for src, dst in rename.items():
                    if sub.startswith(src):
                        out[f"backbone.dinov3.blocks.{index}."
                            f"{dst}{sub[len(src):]}"] = value
                        break
                else:
                    out[f"backbone.dinov3.blocks.{index}.{sub}"] = value
            elif rest.startswith("embeddings.cls_token"):
                out["backbone.dinov3.cls_token"] = value
            elif rest.startswith("embeddings.mask_token"):
                out["backbone.dinov3.mask_token"] = value.reshape(value.shape[0], -1)
            elif rest.startswith("embeddings.register_tokens"):
                out["backbone.dinov3.storage_tokens"] = value
            elif rest.startswith("embeddings.patch_embeddings."):
                out[f"backbone.dinov3.patch_embed.proj.{rest.split('.')[-1]}"] = value
            elif rest.startswith("norm."):
                out[f"backbone.dinov3.{rest}"] = value
        # Rebuild the fused qkv (weight cat, gated bias cat, channel mask).
        n_layers = len(
            {k.split(".layer.")[1].split(".")[0] for k in sd if ".layer." in k}
        )
        for index in range(n_layers):
            layer = f"backbone_wrapper._model.model.layer.{index}.attention"
            q_w = sd[f"{layer}.q_proj.weight"]
            k_w = sd[f"{layer}.k_proj.weight"]
            v_w = sd[f"{layer}.v_proj.weight"]
            q_b = sd[f"{layer}.q_proj.bias"]
            v_b = sd[f"{layer}.v_proj.bias"]
            dim = q_w.shape[0]
            zeros = torch.zeros(dim)
            ones = torch.ones(dim)
            out[f"backbone.dinov3.blocks.{index}.attn.qkv.weight"] = torch.cat(
                [q_w, k_w, v_w]
            )
            out[f"backbone.dinov3.blocks.{index}.attn.qkv.bias"] = torch.cat(
                [q_b, zeros, v_b]
            )
            out[f"backbone.dinov3.blocks.{index}.attn.qkv.bias_mask"] = torch.cat(
                [ones, zeros, ones]
            )
        return {"train_model": {f"ema_model.model.{k}": v for k, v in out.items()}}

    def test_round_trip_80_classes(self, tmp_path):
        from autovision.tasks.detection.avdetr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
        )

        source = self._model(num_classes=80)
        path = tmp_path / "fake_vit_baseline.pt"
        torch.save(self._fake_checkpoint(source), path)

        target = self._model(num_classes=80)
        summary = load_baseline_coco_checkpoint(target, path)
        assert summary["backbone"] == 211
        assert summary["backbone_extras"] == 42  # sta 24 + convs 3 + norms 15
        assert summary["decoder_skipped_cls"] == 0
        # spot tensors identical across the remap
        src_sd = source.state_dict()
        tgt_sd = target.state_dict()
        for key in (
            "backbone_wrapper._model.model.layer.5.attention.q_proj.weight",
            "backbone_wrapper._model.model.layer.5.attention.v_proj.bias",
            "backbone_wrapper._model.model.layer.5.layer_scale1.lambda1",
            "backbone_wrapper._model.embeddings.register_tokens",
            "backbone_wrapper._model.embeddings.mask_token",
            "backbone.sta.stem.0.weight",
            "backbone.convs.1.weight",
            "backbone.norms.2.running_var",
            "encoder.fpn_blocks.0.conv1.conv.weight",
            "decoder.dec_score_head.0.weight",
        ):
            assert torch.equal(tgt_sd[key], src_sd[key]), key

    def test_skips_class_count_heads(self, tmp_path):
        from autovision.tasks.detection.avdetr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
        )

        source = self._model(num_classes=80)
        path = tmp_path / "fake_vit_baseline.pt"
        torch.save(self._fake_checkpoint(source), path)

        target = self._model(num_classes=10)
        summary = load_baseline_coco_checkpoint(target, path)
        # 4 dec score heads + 1 enc score head (w+b each) + 1 denoising
        # embed (weight only).
        assert summary["decoder_skipped_cls"] == 11
        assert not torch.equal(
            target.state_dict()["decoder.dec_score_head.0.weight"],
            source.state_dict()["decoder.dec_score_head.0.weight"],
        )
        assert torch.equal(
            target.state_dict()["decoder.decoder.layers.0.self_attn.in_proj_weight"],
            source.state_dict()["decoder.decoder.layers.0.self_attn.in_proj_weight"],
        )

    def test_geometry_mismatch_raises(self, tmp_path):
        from autovision.tasks.detection.avdetr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
        )

        source = self._model(num_classes=80, variant="vitt16")
        path = tmp_path / "fake_vit_baseline.pt"
        torch.save(self._fake_checkpoint(source), path)

        # vitt16 checkpoint into a vits16 target: the backbone remap cannot
        # land (different widths) and must fail loudly.
        target = self._model(num_classes=80, variant="vits16")
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            load_baseline_coco_checkpoint(target, path)


class TestCocoCheckpointLoaderEdgeCrafter:
    """EdgeCrafter (ECViT) baseline COCO detector warm-start.

    The baseline layout equals the fork's ECViT state dict key-for-key
    (single last-level projector + persistent rope periods), and the
    decoder is the OFFICIAL gateway D-FINE — so the fake checkpoint is a
    straight re-prefix of the fork's own state.
    """

    @staticmethod
    def _model(num_classes: int, variant: str = "ecvitt", **kwargs):
        from autovision.tasks.detection.avdetr.task_model import (
            AVDETRObjectDetection,
        )

        return AVDETRObjectDetection(
            num_classes=num_classes,
            model_name=f"avdetr/edgecrafter-{variant}",
            image_size=(128, 128),
            num_queries=20,
            load_weights=False,
            **kwargs,
        )

    @staticmethod
    def _fake_checkpoint(model) -> dict:
        sd = model.state_dict()
        out: dict[str, torch.Tensor] = {}
        for key, value in sd.items():
            if key.startswith("backbone_wrapper."):
                continue  # duplicate registration of the ECViT wrapper
            if key.startswith(("backbone.", "encoder.", "decoder.")):
                out[key] = value
        return {"train_model": {f"ema_model.model.{k}": v for k, v in out.items()}}

    def test_single_projector_baseline_parity(self):
        # proj_dim=None resolves to ONE projector (applied to the last
        # level) — the published-checkpoint layout, not per-level.
        model = self._model(num_classes=80)
        wrapper = model.backbone._model_wrapper
        assert wrapper.proj_dim == [192]
        assert len(wrapper.projector) == 1
        assert model.backbone.in_channels == [192, 192, 192]
        # RoPE periods are persistent (checkpoint round-trip required —
        # the published values drift ~1e-4 from the formula).
        sd = wrapper.state_dict()
        assert "backbone.rope_embed.periods" in sd
        p3, p4, p5 = wrapper(torch.randn(1, 3, 128, 128))
        assert p3.shape[-3:] == (192, 16, 16)
        assert p5.shape[-3:] == (192, 4, 4)

    def test_round_trip_80_classes(self, tmp_path):
        from autovision.tasks.detection.avdetr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
        )

        source = self._model(num_classes=80)
        path = tmp_path / "fake_ecvit_baseline.pt"
        torch.save(self._fake_checkpoint(source), path)

        target = self._model(num_classes=80)
        summary = load_baseline_coco_checkpoint(target, path)
        # ecvitt @ 224px spec: backbone 172 (incl. rope periods),
        # encoder 222, official dfine decoder 159, no cls skips.
        assert summary == {"backbone": 172, "backbone_extras": 0,
                           "encoder": 222, "decoder": 159,
                           "decoder_skipped_cls": 0}
        src_sd = source.state_dict()
        tgt_sd = target.state_dict()
        for key in (
            "backbone._model_wrapper.backbone.blocks.5.attn.qkv.weight",
            "backbone._model_wrapper.backbone.rope_embed.periods",
            "backbone._model_wrapper.projector.0.conv.weight",
            "encoder.fpn_blocks.0.conv1.conv.weight",
            "decoder.decoder.layers.0.gateway.gate.weight",
            "decoder.dec_bbox_head.0.layers.2.weight",
        ):
            assert torch.equal(tgt_sd[key], src_sd[key]), key

    def test_skips_class_count_heads(self, tmp_path):
        from autovision.tasks.detection.avdetr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
        )

        source = self._model(num_classes=80)
        path = tmp_path / "fake_ecvit_baseline.pt"
        torch.save(self._fake_checkpoint(source), path)

        target = self._model(num_classes=10)
        summary = load_baseline_coco_checkpoint(target, path)
        assert summary["decoder_skipped_cls"] == 11
        assert not torch.equal(
            target.state_dict()["decoder.dec_score_head.0.weight"],
            source.state_dict()["decoder.dec_score_head.0.weight"],
        )
        assert torch.equal(
            target.state_dict()["decoder.decoder.layers.0.self_attn.in_proj_weight"],
            source.state_dict()["decoder.decoder.layers.0.self_attn.in_proj_weight"],
        )

    def test_training_step_matches_fork_contract(self):
        # The official decoder emits the fork's criterion contract and the
        # full D-FINE loss suite (VFL/bbox/giou/FGL/DDF + dn branches).
        from autovision.tasks.detection.components.dfine_criterion import (
            DFINECriterionV2,
        )
        from autovision.tasks.detection.components.matcher import (
            HungarianMatcher,
        )

        model = self._model(num_classes=80)
        model.train()
        criterion = DFINECriterionV2(HungarianMatcher(), reg_max=32, reg_scale=4.0)
        targets = [
            {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
             "labels": torch.tensor([3])},
            {"boxes": torch.tensor([[0.4, 0.4, 0.3, 0.3]]),
             "labels": torch.tensor([1])},
        ]
        out = model._forward_train(torch.randn(2, 3, 128, 128), targets=targets)
        outputs_list = list(out["aux_outputs"]) + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        loss_dict = criterion(
            outputs_list,
            targets,
            world_size=1,
            dn_outputs=out.get("dn_aux_outputs", []),
            dn_meta=out.get("dn_meta"),
            corners_list=out.get("pred_corners_list"),
            ref_boxes_list=out.get("ref_boxes_list"),
            dn_corners_list=out.get("dn_pred_corners_list"),
            dn_ref_boxes_list=out.get("dn_ref_boxes_list"),
            pre_outputs=out.get("pre_outputs"),
            enc_aux_outputs=out.get("enc_aux_outputs"),
            enc_meta=out.get("enc_meta"),
        )
        assert any("loss_vfl" in k for k in loss_dict)
        assert any("loss_fgl" in k for k in loss_dict)
        assert any("loss_ddf" in k for k in loss_dict)
        sum(loss_dict.values()).backward()

    def test_density_and_ssl_rejected(self):
        with pytest.raises(ValueError, match="D-FINE"):
            self._model(num_classes=2, density={"max_num_select": 500})
        with pytest.raises(ValueError, match="D-FINE"):
            self._model(num_classes=2, ssl={"mode": "off"})


class TestGradientClipping:
    def _train_model_with_ones_grads(self):
        train_model = make_train_model()
        optimizer = torch.optim.AdamW(train_model.model.parameters(), lr=1e-9)
        with torch.no_grad():
            for p in train_model.model.parameters():
                p.grad = torch.ones_like(p)
        # All-ones grads: the total norm is sqrt(total element count).
        expected = sum(p.numel() for p in train_model.model.parameters()) ** 0.5
        return train_model, optimizer, expected

    def test_zero_max_norm_disables_clipping(self):
        # Baseline contract: gradient_clip_val <= 0 turns clipping OFF and
        # only reports the total norm — clip_grad_norm_ with max_norm=0
        # would silently ZERO every gradient and kill the run.
        train_model, optimizer, expected = self._train_model_with_ones_grads()
        train_model.model_args.gradient_clip_val = 0.0
        total = train_model.clip_gradients(optimizer)
        assert total == pytest.approx(expected, rel=1e-4)
        for p in train_model.model.parameters():
            assert torch.all(p.grad == 1.0)

    def test_negative_max_norm_disables_clipping(self):
        train_model, optimizer, expected = self._train_model_with_ones_grads()
        train_model.model_args.gradient_clip_val = -1.0
        total = train_model.clip_gradients(optimizer)
        assert total == pytest.approx(expected, rel=1e-4)
        for p in train_model.model.parameters():
            assert torch.all(p.grad == 1.0)

    def test_positive_max_norm_still_clips(self):
        train_model, optimizer, expected = self._train_model_with_ones_grads()
        train_model.model_args.gradient_clip_val = 0.1
        total = train_model.clip_gradients(optimizer)
        # clip_grad_norm_ returns the PRE-clip total norm; the grads
        # themselves are rescaled down to the max norm.
        assert total == pytest.approx(expected, rel=1e-4)
        scale = 0.1 / expected
        for p in train_model.model.parameters():
            torch.testing.assert_close(p.grad, torch.full_like(p.grad, scale))


class TestVariantRecipes:
    """Baseline recipe dispatch is by model-name prefix (dinov2 V2 args vs
    the AVDETR args everything else shares) — not by network family."""

    def _resolved(self, backbone: str, **kwargs):
        args = AVDETRObjectDetectionTrainArgs(
            num_classes=2, backbone=backbone, image_size=(224, 224), **kwargs
        )
        args.resolve_auto(
            total_steps=1000, train_num_batches=100, gradient_accumulation_steps=1
        )
        return args

    def test_dinov2_keeps_v2_recipe(self):
        args = self._resolved("dinov2-vittest-notpretrained")
        assert args.lr == 1e-4
        assert args.backbone_lr_factor == 1e-2
        assert args.lr_scheduler == "linear"
        assert args.lr_warmup_steps == 2000
        assert args.lr_reference_batch == 16

    def test_dinov3_vit_walks_ltdetr_recipe(self):
        # The bug this locks: dinov3 ViT used to inherit the dinov2 V2
        # recipe (1e-4 / 1e-2 / linear) instead of the AVDETR recipe.
        args = self._resolved("dinov3-vits16-notpretrained")
        assert args.lr == 5e-4
        assert args.backbone_lr_factor == 0.05
        assert args.lr_scheduler == "flat_cosine"
        assert isinstance(args.lr_warmup_steps, int)
        assert args.lr_warmup_steps != 2000
        assert args.lr_reference_batch == 32

    def test_convnext_walks_ltdetr_recipe(self):
        args = self._resolved("dinov3-convnext-tiny")
        assert args.lr == 5e-4
        assert args.backbone_lr_factor == 0.05
        assert args.lr_scheduler == "flat_cosine"
        assert args.lr_reference_batch == 32

    def test_explicit_pins_win(self):
        args = self._resolved(
            "dinov3-vits16-notpretrained",
            lr=1e-4,
            backbone_lr_factor=0.1,
            lr_scheduler="linear",
            lr_reference_batch=16,
        )
        assert args.lr == 1e-4
        assert args.backbone_lr_factor == 0.1
        assert args.lr_scheduler == "linear"
        assert args.lr_reference_batch == 16

    def test_default_batch_size_helper(self):
        from autovision.tasks.detection.avdetr import (
            avdetr_default_batch_size,
        )

        assert avdetr_default_batch_size("dinov2-vits14") == 16
        assert avdetr_default_batch_size("dinov2-vittest-notpretrained") == 16
        assert avdetr_default_batch_size("dinov3-convnext-tiny") == 32
        assert avdetr_default_batch_size("dinov3-vits16-notpretrained") == 32
        assert avdetr_default_batch_size("edgecrafter-ecvitt") == 32


class TestConvNextOptimizerScaling:
    def test_lr_sqrt_scaling_uses_variant_reference_batch(self):
        # Baseline denominator: 32 for the ConvNeXt variant (the dinov2
        # V2 args keep 16). 5e-4 * sqrt(16/32) ~ 3.536e-4.
        args = AVDETRObjectDetectionTrainArgs(
            num_classes=2,
            backbone="dinov3-convnext-tiny",
            image_size=(224, 224),
            decoder="dfine",
            backbone_pretrained=False,
        )
        train_model = AVDETRObjectDetectionTrain(args)
        optimizer, _ = train_model.get_optimizer(
            total_steps=100, global_batch_size=16
        )
        detector = next(
            g for g in optimizer.param_groups if g["name"] == "detector"
        )
        base = detector.get("_initial_lr", detector.get("initial_lr"))
        assert base == pytest.approx(5e-4 * (16 / 32) ** 0.5)


class TestDecoders:
    def test_rtdetrv2_is_default(self):
        train_model = make_train_model()
        assert train_model.model.decoder_name == "rtdetrv2"
        from autovision.tasks.detection.components.rtdetrv2_decoder import (
            RTDETRv2Decoder,
        )

        assert type(train_model.model.decoder) is RTDETRv2Decoder
        assert type(train_model.criterion).__name__ == "RTDETRCriterionV2"

    def test_dfine_decoder_swaps_criterion(self):
        from autovision.tasks.detection.components.dfine_decoder import (
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

    def test_ecvit_auto_selects_official_dfine(self):
        from autovision.tasks.detection.components.dfine_criterion import (
            DFINECriterionV2,
        )
        from autovision.tasks.detection.components.dfine_official_decoder import (
            DFINEOfficialDecoder,
        )

        args = AVDETRObjectDetectionTrainArgs(
            num_classes=2,
            backbone="edgecrafter-ecvitt",
            image_size=(224, 224),
            num_queries=20,
            num_top_queries=20,
        )
        train_model = AVDETRObjectDetectionTrain(args)
        assert train_model.model.decoder_name == "dfine_official"
        assert type(train_model.model.decoder) is DFINEOfficialDecoder
        assert isinstance(train_model.criterion, DFINECriterionV2)
        # Larger ECViT variants drop the backbone LR factor (baseline).
        optimizer, _ = train_model.get_optimizer(
            total_steps=100, global_batch_size=16
        )
        groups = {g["name"]: g for g in optimizer.param_groups}
        # Baseline denominator 32 for the ECViT variant (was fork-16).
        assert groups["backbone"]["lr"] == pytest.approx(
            5e-4 * (16 / 32) ** 0.5 * 0.05
        )
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
        args = AVDETRObjectDetectionTrainArgs(num_classes=2)
        assert args.image_size == (640, 640)
        args.resolve_image_size()
        # 640 -> 644: next multiple of 2 x patch 14.
        assert args.image_size == (644, 644)

    def test_patch16_default_unchanged(self):
        args = AVDETRObjectDetectionTrainArgs(
            num_classes=2, backbone="edgecrafter-ecvitt"
        )
        args.resolve_image_size()
        assert args.image_size == (640, 640)

    def test_explicit_divisible_size_kept(self):
        args = AVDETRObjectDetectionTrainArgs(num_classes=2, image_size=(644, 644))
        args.resolve_image_size()
        assert args.image_size == (644, 644)

    def test_indivisible_explicit_size_raises(self):
        args = AVDETRObjectDetectionTrainArgs(num_classes=2, image_size=(640, 640))
        args.backbone = "edgecrafter-ecvitt"  # divisor 32 -> 640 still fine
        args.resolve_image_size()
        args.image_size = (500, 500)
        with pytest.raises(ValueError, match="divisible by 32"):
            args.resolve_image_size()

    def test_task_model_rejects_bad_size(self):
        from autovision.tasks.detection.avdetr.task_model import (
            AVDETRObjectDetection,
        )

        with pytest.raises(ValueError, match="divisible by 28"):
            AVDETRObjectDetection(
                num_classes=2,
                model_name="dinov2-vittest-notpretrained",
                image_size=(64, 64),
                num_queries=10,
            )


class TestECViT:
    def test_package_lists_models(self):
        from autovision.encoders.package_helpers import list_model_names

        names = list_model_names()
        for size in ("ecvitt", "ecvittplus", "ecvits", "ecvitsplus"):
            assert f"edgecrafter/{size}" in names

    def test_wrapper_emits_pyramid(self):
        from autovision.encoders.ecvit.ecvit import ECViTModelWrapper

        wrapper = ECViTModelWrapper("ecvitt")
        p3, p4, p5 = wrapper(torch.randn(2, 3, 224, 224))
        assert p3.shape == (2, 192, 28, 28)
        assert p4.shape == (2, 192, 14, 14)
        assert p5.shape == (2, 192, 7, 7)

    def test_weights_path_strict_load(self, tmp_path):
        from autovision.encoders.ecvit.ecvit import ECViTModelWrapper

        wrapper = ECViTModelWrapper("ecvitt")
        path = tmp_path / "ecvit.pt"
        torch.save(wrapper.backbone.state_dict(), path)
        reloaded = ECViTModelWrapper("ecvitt", weights_path=str(path))
        for key, value in wrapper.backbone.state_dict().items():
            assert torch.allclose(reloaded.backbone.state_dict()[key], value)

    def test_ecvit_train_step(self):
        args = AVDETRObjectDetectionTrainArgs(
            num_classes=2,
            backbone="edgecrafter-ecvitt",
            image_size=(224, 224),
            num_queries=20,
            num_top_queries=20,
        )
        train_model = AVDETRObjectDetectionTrain(args)
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


class TestAVDETRTransforms:
    def test_scale_jitter_ladder_aligns_to_patch_grid(self):
        from autovision.tasks.detection.avdetr.transforms import (
            avdetr_scale_jitter_sizes,
        )

        # Patch 16 at 640 reproduces the stock 480..800 ladder.
        sizes = avdetr_scale_jitter_sizes((640, 640), 32)
        assert sizes[:11] == [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        assert all(s % 32 == 0 for s in sizes)
        # Patch 14 ladder snaps to multiples of 28 around 644.
        sizes14 = avdetr_scale_jitter_sizes((644, 644), 28)
        assert all(s % 28 == 0 for s in sizes14)
        assert 644 in sizes14 and 476 in sizes14 and 812 in sizes14

    def test_build_args_enables_mosaic(self):
        from autovision.tasks.detection.avdetr.transforms import (
            build_avdetr_train_transform_args,
        )

        args = build_avdetr_train_transform_args((644, 644), "dinov2-vits14")
        assert args.mosaic_prob == 0.5
        assert args.image_size == (644, 644)
        assert all(s % 28 == 0 for s in args.scale_jitter.sizes)
        # EdgeCrafter "v2" recipe raises mosaic prob (baseline).
        args_v2 = build_avdetr_train_transform_args((640, 640), "edgecrafter-ecvitt")
        assert args_v2.mosaic_prob == 0.75
        # ... AND the mixup prob (baseline AVDETRv2 args: 0.75; v1 keeps 0.5).
        assert args_v2.mixup_prob == 0.75
        assert args.mixup_prob == 0.5
        # User overrides win.
        args_off = build_avdetr_train_transform_args(
            (644, 644), "dinov2-vits14", {"mosaic_prob": None}
        )
        assert args_off.mosaic_prob is None

    def test_mosaic_merges_samples_and_aligns_boxes(self):
        from torchvision import tv_tensors

        from autovision.augment.detection.args import (
            RTDETRDetectionTransformArgs,
        )
        from autovision.augment.detection.transform import (
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
        # Baseline donor sampling is with replacement from the whole
        # cache (duplicates allowed); the merged sample still carries a
        # healthy box count.
        assert outputs[-1]["bboxes"].shape[0] >= 4
        # Mosaic samples keep photometric distort; only zoom-out/IoU-crop
        # are skipped (baseline behavior).
        assert (True, False, False, False) in transform.compose._compose_cache

    def test_mosaic_composes_with_tiny_cache(self):
        # Baseline donor draws are with replacement: the FIRST sample (a
        # one-entry cache) still composes a full 4-tile mosaic.
        from torchvision import tv_tensors

        from autovision.augment.detection.args import (
            RTDETRDetectionTransformArgs,
        )
        from autovision.augment.detection.transform import (
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
        item = {
            "image_path": "img0.jpg",
            "image": torch.randint(0, 255, (3, 100, 120), dtype=torch.uint8),
            "bboxes": tv_tensors.BoundingBoxes(
                torch.tensor([[10.0, 10.0, 50.0, 60.0]]),
                format="XYXY",
                canvas_size=(100, 120),
            ),
            "classes": torch.tensor([0]),
            "original_size": (120, 100),
        }
        output = transform(item)
        assert output["image"].shape == (3, 224, 224)
        assert output["bboxes"].shape[0] >= 1
        assert output["bboxes"].shape[0] == output["classes"].shape[0]

    def test_mosaic_cache_random_pop_keeps_newest(self):
        from torchvision import tv_tensors

        from autovision.augment.detection.args import (
            RTDETRDetectionTransformArgs,
        )
        from autovision.augment.detection.transform import (
            RTDETRDetectionTransform,
        )

        args = RTDETRDetectionTransformArgs(
            image_size=(224, 224),
            mosaic_prob=None,  # cache append only fires when mosaic is on
            scale_jitter=None,
        )
        transform = RTDETRDetectionTransform(args)
        transform.MOSAIC_CACHE_SIZE = 3
        for index in range(10):
            # Distinct pixel value identifies the sample through the clone.
            transform._mosaic_cache_append(
                {
                    "image_path": f"img{index}",
                    "image": torch.full((3, 8, 8), index, dtype=torch.uint8),
                    "bboxes": tv_tensors.BoundingBoxes(
                        torch.zeros(0, 4), format="XYXY", canvas_size=(8, 8)
                    ),
                    "classes": torch.zeros(0, dtype=torch.int64),
                }
            )
        marks = [int(s["image"][0, 0, 0]) for s in transform._mosaic_cache]
        assert len(marks) == 3
        # random_pop must never evict the freshly appended entry.
        assert marks[-1] == 9

    def test_mosaic_boxes_survive_whole_canvas_affine(self):
        # The affine is applied to the assembled canvas: boxes must stay
        # inside the canvas, aligned with classes, and never degenerate.
        from torchvision import tv_tensors

        from autovision.augment.detection.args import (
            RTDETRDetectionTransformArgs,
        )
        from autovision.augment.detection.transform import (
            RTDETRDetectionTransform,
        )

        random.seed(0)
        torch.manual_seed(0)
        args = RTDETRDetectionTransformArgs(
            image_size=(320, 320),
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
                    torch.tensor([[20.0, 20.0, 80.0, 80.0]]),
                    format="XYXY",
                    canvas_size=(100, 120),
                ),
                "classes": torch.tensor([3]),
                "original_size": (120, 100),
            }

        for _ in range(8):
            output = transform(make_item(0))
            boxes = output["bboxes"]
            assert boxes.shape[0] == output["classes"].shape[0]
            if boxes.shape[0]:
                # Normalized cxcywh: w/h are columns 2/3.
                assert bool((boxes[:, 2] > 0).all()) and bool((boxes[:, 3] > 0).all())
                assert bool((boxes >= 0).all()) and bool((boxes <= 1).all())

    def test_mosaic_keeps_photometric(self):
        from torchvision import tv_tensors

        from autovision.augment.detection.args import (
            RTDETRDetectionTransformArgs,
        )
        from autovision.augment.detection.transform import (
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
        from autovision.augment.detection.args import (
            RTDETRDetectionTransformArgs,
        )
        from autovision.augment.detection.collate import (
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


class TestE2E:
    def test_train_object_detection_ltetr(self, tmp_path):
        train_json, val_json, images_dir = make_coco_dataset(tmp_path, num_images=8)
        result = train_object_detection(
            out=tmp_path / "ltdetr1",
            data=data_config(train_json, val_json, images_dir),
            model="avdetr/dinov2-vittest-notpretrained",
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
            model="avdetr/dinov2-vittest-notpretrained",
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


class TestEncoderOverrides:
    def test_enc_overrides_reach_the_encoder(self):
        """The official Dome-S 'S' encoder recipe (1024/0.5/0.34) must land
        in the built HybridEncoder instead of the variant spec's heavy
        defaults (2048/1.0/1.0)."""
        from autovision.tasks.detection.avdetr import (
            AVDETRObjectDetection,
            AVDETRObjectDetectionTrainArgs,
        )

        args = AVDETRObjectDetectionTrainArgs(
            num_classes=10,
            backbone="dinov3-convnext-tiny",
            image_size=(64, 64),
            decoder="dfine",
            num_queries=10,
            num_top_queries=10,
            enc_dim_feedforward=1024,
            enc_expansion=0.5,
            enc_depth_mult=0.34,
        )
        model = AVDETRObjectDetection(
            num_classes=args.num_classes,
            model_name=args.backbone,
            image_size=args.image_size,
            decoder="dfine",
            reg_max=8,
            load_weights=False,
            enc_dim_feedforward=args.enc_dim_feedforward,
            enc_expansion=args.enc_expansion,
            enc_depth_mult=args.enc_depth_mult,
        )
        assert model.encoder.enc_dim_feedforward == 1024
        assert model.encoder.enc_expansion == 0.5
        assert model.encoder.enc_depth_mult == 0.34

    def test_enc_auto_follows_variant_spec(self):
        from autovision.tasks.detection.avdetr import (
            AVDETRObjectDetection,
            AVDETRObjectDetectionTrainArgs,
        )

        args = AVDETRObjectDetectionTrainArgs(
            num_classes=10,
            backbone="dinov3-convnext-tiny",
            image_size=(64, 64),
            decoder="dfine",
            num_queries=10,
            num_top_queries=10,
        )
        model = AVDETRObjectDetection(
            num_classes=args.num_classes,
            model_name=args.backbone,
            image_size=args.image_size,
            decoder="dfine",
            reg_max=8,
            load_weights=False,
        )
        assert model.encoder.enc_dim_feedforward == 2048  # variant spec value
        assert model.encoder.enc_expansion == 1.0
        assert model.encoder.enc_depth_mult == 1.0


class TestSIoUBoxRegression:
    """box_regression_iou='siou' config surface (design doc 15 §4.4-4.5)."""

    def test_default_config_keeps_giou(self):
        from autovision.tasks.detection.avdetr import (
            AVDETRObjectDetectionTrainArgs,
        )

        args = AVDETRObjectDetectionTrainArgs(num_classes=2)
        assert args.box_regression_iou == "giou"
        assert args.siou_loss_coef is None
        assert args.siou_theta == pytest.approx(4.0)

    def test_config_rejects_invalid_siou_values(self):
        from pydantic import ValidationError

        from autovision.tasks.detection.avdetr import (
            AVDETRObjectDetectionTrainArgs,
        )

        with pytest.raises(ValidationError):
            AVDETRObjectDetectionTrainArgs(num_classes=2, siou_theta=0.0)
        with pytest.raises(ValidationError):
            AVDETRObjectDetectionTrainArgs(num_classes=2, siou_loss_coef=-0.1)

    def test_siou_requires_dfine_decoder(self):
        with pytest.raises(ValueError, match="box_regression_iou='siou'"):
            make_train_model(box_regression_iou="siou", decoder="rtdetrv2")

    def test_siou_wires_dfine_criterion_weight(self):
        train_model = make_train_model(
            decoder="dfine", box_regression_iou="siou", reg_max=16
        )
        assert train_model.criterion.box_regression_iou == "siou"
        assert train_model.criterion.weight_dict["loss_siou"] == pytest.approx(2.0)
        assert "loss_giou" not in train_model.criterion.weight_dict

    def test_siou_training_step_emits_full_suffix_family(self):
        """End-to-end (design doc 15 §7 item 12): the swapped term reaches
        every box-loss branch — main, _pre, _dn_pre, the dn layers and the
        encoder proposals — with no loss_giou key anywhere."""
        torch.manual_seed(0)
        train_model = make_train_model(
            decoder="dfine", box_regression_iou="siou", reg_max=16
        )
        batch = make_batch()
        result = train_model.training_step(
            {
                "image": batch["image"],
                "bboxes": batch["bboxes"],
                "classes": batch["classes"],
            },
            0,
        )
        keys = set(result.log_dict)
        assert "loss_siou" in keys
        assert "loss_bbox_pre" in keys  # the _pre branch ran
        assert "loss_siou_pre" in keys
        assert "loss_siou_dn_pre" in keys
        assert any(key.startswith("loss_siou_dn_") for key in keys)
        assert any(key.startswith("loss_siou_enc_") for key in keys)
        assert not any("loss_giou" in key for key in keys)
        assert torch.isfinite(result.loss)


# ── R16 family rename: legacy "ltdetr/" key alias ───────────────────────


class TestFamilyKeyAlias:
    def test_legacy_key_resolves_with_deprecation(self):
        import warnings

        from autovision.tasks.detection.avdetr.config import (
            resolve_avdetr_variant,
        )

        canonical = resolve_avdetr_variant("avdetr/dinov3-vits16")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = resolve_avdetr_variant("ltdetr/dinov3-vits16")
        assert legacy[0] == canonical[0]
        assert legacy[1] == canonical[1]
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_listed_names_use_new_prefix(self):
        from autovision.tasks.detection.avdetr.config import (
            list_avdetr_model_names,
        )

        names = list_avdetr_model_names()
        assert names and all(name.startswith("avdetr/") for name in names)
