# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""User-held DINOv3 detector assets: registry ↔ store consistency (D1-a).

The store lives outside the git tree; store-dependent tests auto-skip
(trees without it stay green). Registry shape and build-contract tests run
everywhere. Container sha verification: one 40 MB spot check by default,
the full ~4 GB sweep is opt-in via ``AUTOVISION_USER_HELD_TESTS=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from autovision.registry.user_held_assets import (
    USER_HELD_ASSETS,
    USER_HELD_ROOT,
    UserHeldAssetError,
    resolve_avwarm_start_source,
    resolve_user_held_asset,
    user_held_model_args,
    verify_user_held_asset,
)

_MANIFEST = USER_HELD_ROOT / "manifest.json"
_store_present = pytest.mark.skipif(
    not _MANIFEST.is_file(), reason="user-held store not present"
)

_CONVNEXT_NAMES = (
    "dinov3-convnext-tiny",
    "dinov3-convnext-small",
    "dinov3-convnext-base",
    "dinov3-convnext-large",
)
_VIT_NAMES = ("dinov3-vitt16", "dinov3-vitt16plus", "dinov3-vits16")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and set(value) <= set("0123456789abcdef")


class TestRegistryShape:
    def test_seven_assets_with_pinned_identity(self):
        assert sorted(USER_HELD_ASSETS) == sorted(_VIT_NAMES + _CONVNEXT_NAMES)
        for name, asset in USER_HELD_ASSETS.items():
            assert asset.name == name
            assert asset.model_name.startswith("detr/")
            for digest in (
                asset.source_sha256,
                asset.weights_sha256,
                asset.warm_start_sha256,
            ):
                assert _is_sha256(digest), (name, digest)
            assert asset.num_classes == 80
            assert asset.image_size == (640, 640)
            assert asset.license_id == "DINOv3-License+NOASSERTION"
            assert asset.redistributable is False
            assert asset.origin.endswith(".pt")

    def test_model_names_resolve_in_detr_registry(self):
        from autovision.tasks.detection.detr.variants import resolve_detr_variant

        for name, asset in USER_HELD_ASSETS.items():
            key, _spec = resolve_detr_variant(asset.model_name)
            # ConvNeXt store names double as the registry key; the ViT
            # entries load into the ``-notpretrained`` (offline) variants.
            assert key in (name, f"{name}-notpretrained")

    def test_convnext_requires_six_layer_256_decoder(self):
        """The published ConvNeXt checkpoints carry dec_layers=6 at the
        fixed 256 decoder width — NOT the variant spec defaults (3/4
        layers, 384 width), so the overrides are part of the contract."""
        for name in _CONVNEXT_NAMES:
            asset = USER_HELD_ASSETS[name]
            assert asset.dec_layers == 6, name
            assert asset.decoder_hidden_dim == 256, name
        for name in _VIT_NAMES:
            asset = USER_HELD_ASSETS[name]
            assert asset.dec_layers is None, name
            assert asset.decoder_hidden_dim is None, name

    def test_required_model_args_shape(self):
        args = user_held_model_args("dinov3-convnext-base")
        assert args["dec_layers"] == 6
        assert args["decoder_hidden_dim"] == 256
        assert args["backbone_pretrained"] is False
        assert args["image_size"] == (640, 640)
        # The registered name, not a path: resolution happens at build
        # time against the (overridable) store root.
        assert args["detector_checkpoint"] == "dinov3-convnext-base"

        vit = user_held_model_args("dinov3-vitt16")
        assert "dec_layers" not in vit
        assert "decoder_hidden_dim" not in vit

    def test_unknown_name_error_lists_registered(self):
        with pytest.raises(UserHeldAssetError) as excinfo:
            resolve_user_held_asset("dinov3-nope")
        message = str(excinfo.value)
        assert "dinov3-vitt16" in message
        assert "D1-a" in message


class TestConfigAcceptsNames:
    def test_registered_names_pass_strict_validation(self):
        """``detector_checkpoint`` / ``warm_start`` accept registered names
        (strings) next to Path — same convention as the GTR families'
        ``checkpoint`` field. Resolution happens at build time."""
        from autovision.tasks.detection.detr import DETRObjectDetectionTrainArgs

        args = DETRObjectDetectionTrainArgs(
            backbone="dinov3-vitt16-notpretrained",
            num_classes=80,
            detector_checkpoint="dinov3-vitt16",
            warm_start="dinov3-vitt16",
        )
        assert args.detector_checkpoint == "dinov3-vitt16"
        assert args.warm_start == "dinov3-vitt16"


class TestStoreConsistency:
    """Store-dependent: run on the dev box, skip where the store is absent."""

    @_store_present
    def test_manifest_matches_registry(self):
        manifest = json.loads(_MANIFEST.read_text())
        by_name = {a["variant"]: a for a in manifest["assets"]}
        assert set(by_name) == set(USER_HELD_ASSETS)
        for name, entry in by_name.items():
            asset = USER_HELD_ASSETS[name]
            assert entry["dir"] == asset.name
            assert entry["model_name"] == asset.model_name
            assert entry["files"]["weights.pt"] == asset.weights_sha256
            assert entry["files"]["warm_start.pt"] == asset.warm_start_sha256
            assert entry["license"]["redistributable"] is False

    @_store_present
    def test_vitt16_weights_sha256_spot_check(self):
        verify_user_held_asset(
            resolve_user_held_asset("dinov3-vitt16"), files=("weights.pt",)
        )

    @_store_present
    def test_resolve_detector_checkpoint_accepts_registered_name(self):
        from autovision.tasks.detection.detr.coco_checkpoint import (
            resolve_detector_checkpoint,
        )

        resolved = resolve_detector_checkpoint("dinov3-vitt16")
        assert resolved == USER_HELD_ASSETS["dinov3-vitt16"].weights_path
        assert resolved.is_file()

    @_store_present
    def test_warm_start_name_resolution(self):
        resolved = resolve_avwarm_start_source("dinov3-vitt16")
        assert resolved == USER_HELD_ASSETS["dinov3-vitt16"].warm_start_path
        assert resolved.is_file()

    def test_warm_start_unknown_name_passes_through(self):
        passthrough = resolve_avwarm_start_source("some/other/path.pt")
        assert passthrough == Path("some/other/path.pt")

    @_store_present
    def test_vitt16_real_load_matches_recorded_summary(self):
        """End-to-end: build through the train model (name resolution +
        load) and match the summary the packing session recorded."""
        from autovision.tasks.detection.detr import (
            DETRObjectDetectionTrain,
            DETRObjectDetectionTrainArgs,
        )
        from autovision.tasks.detection.detr.coco_checkpoint import (
            load_baseline_coco_checkpoint,
            resolve_detector_checkpoint,
        )

        args = DETRObjectDetectionTrainArgs(
            backbone="dinov3-vitt16-notpretrained",
            num_classes=80,
            **user_held_model_args("dinov3-vitt16"),
        )
        model = DETRObjectDetectionTrain(args)
        recorded = json.loads(
            (USER_HELD_ROOT / "dinov3-vitt16" / "ASSET.json").read_text()
        )["loader_summary"]
        path = resolve_detector_checkpoint(args.detector_checkpoint)
        actual = load_baseline_coco_checkpoint(model.model, path)
        for key, value in recorded.items():
            assert actual[key] == value, key

    @_store_present
    def test_convnext_without_overrides_is_rejected(self):
        """The store's ConvNeXt checkpoints carry dec_layers=6; building at
        the variant defaults must fail loudly rather than mis-load."""
        from autovision.tasks.detection.detr import (
            DETRObjectDetectionTrain,
            DETRObjectDetectionTrainArgs,
        )

        args = DETRObjectDetectionTrainArgs(
            backbone="dinov3-convnext-tiny",
            num_classes=80,
            detector_checkpoint="dinov3-convnext-tiny",
            backbone_pretrained=False,
            image_size=(640, 640),
        )
        with pytest.raises(ValueError):
            DETRObjectDetectionTrain(args)

    @_store_present
    def test_warm_start_name_loads_into_network(self):
        from autovision.tasks.detection.detr import (
            DETRObjectDetectionTrain,
            DETRObjectDetectionTrainArgs,
        )
        from autovision.tasks.detection.detr.task_model import (
            load_warm_start_into_network,
        )

        args = DETRObjectDetectionTrainArgs(
            backbone="dinov3-vitt16-notpretrained", num_classes=80
        )
        model = DETRObjectDetectionTrain(args)
        summary = load_warm_start_into_network(model.model, "dinov3-vitt16")
        assert summary

    @pytest.mark.skipif(
        os.environ.get("AUTOVISION_USER_HELD_TESTS") != "1",
        reason="set AUTOVISION_USER_HELD_TESTS=1 to hash all store containers",
    )
    def test_full_sha256_verification_all_assets(self):
        for asset in USER_HELD_ASSETS.values():
            verify_user_held_asset(asset)
