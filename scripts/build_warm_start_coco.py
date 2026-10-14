#!/usr/bin/env python3
"""Build the self-produced COCO warm-start checkpoint (design doc 28, §6.2).

Replaces the reference-system S3 warm-start assets with a licensed,
self-trained equivalent (ledger P3-1). Pipeline:

    1. init : Apache-2.0 official component weights (EdgeCrafter ECViT /
       DINOv2 backbone, D-FINE decoder) — downloaded via the asset
       manifest with license metadata;
    2. assemble : the target detection network (family per --family);
    3. finetune : short standard COCO recipe through the regular
       detection training pipeline;
    4. export : AVWarmStart v1 checkpoint (registry/warm_start.py) with
       mandatory license metadata.

STATUS: skeleton. Steps 1-2 need the Apache upstream weights, which are
BLOCKED on network access (ledger P1-2..P1-6); the COCO finetune call
reuses ``autovision.cli.train_object_detection`` and needs D1 (backbone
route) decided. The save/export path below is final and already tested.

Usage (once unblocked):
    python scripts/build_warm_start_coco.py \
        --family ltdetr --backbone ecvitt --classes 80 \
        --out weights/avdetr_ecvitt_coco_warmstart.pt \
        --spec-note "built from Apache upstream init"
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autovision.registry.assets import LicenseRef  # noqa: E402
from autovision.registry.warm_start import (  # noqa: E402
    WarmStartError,
    save_warm_start,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default="ltdetr", help="detector family")
    parser.add_argument(
        "--backbone",
        default="ecvitt",
        help="backbone variant; Apache lines (ecvitt, dinov2-*) need no "
        "extra approval, DINOv3 variants additionally require "
        "AUTOVISION_DINOV3_WEIGHTS=1 (decision D1-a)",
    )
    parser.add_argument("--classes", type=int, default=80, help="COCO class count")
    parser.add_argument(
        "--out",
        default="weights/av_warm_start_coco.pt",
        help="output AVWarmStart v1 path",
    )
    parser.add_argument("--spec-note", default="", help="free-form provenance note")
    return parser.parse_args()


def build(family: str, backbone: str, classes: int, out: Path, note: str) -> Path:
    # TODO(Phase 4, ledger P3-1): the three steps below are blocked on
    # (a) Apache upstream weights (no network in the remediation env) and
    # (b) decision D1 (DINOv3 route). The export path is final.
    raise NotImplementedError(
        "COCO warm-start build needs the Apache upstream weights "
        "(ledger P1-2..P1-6) and decision D1; see docs/design/"
        "28-clean-provenance-refactor.md section 6.2."
    )
    # state_dict = assemble_target_network(family=family, backbone=backbone)  # noqa: E800
    # state_dict = finetune_on_coco(state_dict, classes=classes)  # noqa: E800
    return save_warm_start(
        out,
        model={},  # replaced by the finetuned state dict
        spec={
            "family": family,
            "backbone": backbone,
            "classes": classes,
            "created": _dt.date.today().isoformat(),
            "note": note,
        },
        license=LicenseRef(id="Apache-2.0", origin="self-trained (this run)"),
    )


def main() -> int:
    args = parse_args()
    try:
        out_path = build(
            family=args.family,
            backbone=args.backbone,
            classes=args.classes,
            out=Path(args.out),
            note=args.spec_note,
        )
    except NotImplementedError as error:
        print(f"BLOCKED: {error}")
        return 2
    except WarmStartError as error:
        print(f"ERROR: {error}")
        return 1
    print(f"warm-start written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
