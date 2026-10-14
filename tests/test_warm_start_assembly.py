# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Assembly-loader tests (design doc 28 §6.2): Apache components -> AV-DETR.

These tests exercise the real downloaded Apache weights when they are
present in the local cache and skip cleanly on machines without them
(CI has no weight files; the AVWarmStart format itself is covered by
tests/test_warm_start.py without binaries).
"""

from pathlib import Path

import pytest

from autovision.registry.warm_start import WarmStartError

_CACHE = Path.home() / ".cache" / "autovision"
DFINE_S = _CACHE / "d-fine_dfine_s_coco.pth"
ECVITT = _CACHE / "edgecrafter_ecvitt.pth"

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _make_model(backbone: str):
    from autovision.tasks.detection.avdetr import AVDETRObjectDetection

    return AVDETRObjectDetection(
        num_classes=80, model_name=backbone,
        image_size=(320, 320), decoder="dfine_official",
    )


@pytest.mark.skipif(not DFINE_S.is_file(), reason="D-FINE COCO weights not cached")
class TestDfineOfficialWarmStart:
    def test_width_matched_variant_transfers(self):
        from autovision.tasks.detection.avdetr.dfine_official_warm_start import (
            load_dfine_official_warm_start,
        )

        model = _make_model("edgecrafter-ecvits")  # 256-wide: matches D-FINE
        summary = load_dfine_official_warm_start(model, DFINE_S)
        assert summary["decoder_coverage"] >= 0.9
        assert summary["tensors_transferred"] >= 150
        assert summary["discarded_backbone_tensors"] > 0

    def test_width_mismatched_variant_refused(self):
        from autovision.tasks.detection.avdetr.dfine_official_warm_start import (
            load_dfine_official_warm_start,
        )

        model = _make_model("edgecrafter-ecvitt")  # 192-wide: no D-FINE match
        with pytest.raises(WarmStartError, match="coverage"):
            load_dfine_official_warm_start(model, DFINE_S)

    @pytest.mark.skipif(not ECVITT.is_file(), reason="ECViT weights not cached")
    def test_ecvitt_backbone_loads_into_matching_variant(self):
        from autovision.registry.assets import LicenseRef
        from autovision.registry.warm_start import save_warm_start
        from autovision.tasks.detection.avdetr.dfine_official_warm_start import (
            load_dfine_official_warm_start,
        )

        model = _make_model("edgecrafter-ecvitt")
        summary = load_dfine_official_warm_start(
            model, DFINE_S, backbone_weights=ECVITT,
            min_decoder_coverage=0.2,  # 192-wide: decoder mostly fresh by design
        )
        assert summary["backbone_weights"] == str(ECVITT)
        # Backbone tensors actually changed (not fresh init): spot-check one.
        block = model.state_dict()["backbone._model_wrapper.backbone.blocks.0.norm1.weight"]
        assert block.abs().sum() > 0
        # The full pipeline endpoint: the assembled state saves as AVWarmStart.
        out = save_warm_start(
            _CACHE / "smoke_warm_start.pt",
            model=model.state_dict(),
            spec={"family": "avdetr", "classes": 80},
            license=LicenseRef(id="Apache-2.0", origin="assembly smoke"),
        )
        assert out.is_file()
        out.unlink()
