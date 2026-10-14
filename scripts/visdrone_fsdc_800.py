# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone FSDC-DETR at 800x800 (design doc 22 §8).

Twin of ``scripts/visdrone_deimv2_800.py`` — identical data paths,
10-epoch pilot, global batch 16 = per-device 4 x accum 4, LR auto-scaled
by sqrt(16/32), scale jitter locked at 800, no mosaic/mixup, pilot EMA
0.993 — with the FSDC family: the DEIMv2 trunk (official HF criterion,
dense O2O) carries the native dual-branch encoder (DBFSAF fusion) and
the FSC hybrid encoder (SFS-FF + embedded FSD-Down at PAN). The ONLY
model-side difference vs the DEIMv2 baseline is the ``fsdc`` field, so
every metric delta is attributable to the frequency modules.

Variant notes (design doc 22 PR-0):
- ``fsdc-l`` — paper anchor (DEIMv2-L architecture, hidden 224). No
  transformers-format COCO conversion exists on the Hub: trains with a
  fresh decoder (``load_weights=False`` is enforced by the registry).
- ``fsdc-s`` — warm-start arm: the decoder initializes from the
  in-repo DEIMv2-S COCO conversion (``load_weights=True``), the swapped
  backbone/encoder modules train fresh.

Run:

    .venv/bin/python scripts/visdrone_fsdc_800.py [--smoke]
    .venv/bin/python scripts/visdrone_fsdc_800.py --model fsdc-s \
        --no-dbfsaf          # encoder-only ablation arm (doc 22 §8 c/b)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from autovision import train_object_detection

VISDRONE = Path("/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone")

DATA_ARGS = {
    "format": "coco",
    "train": {
        "annotations": VISDRONE / "annotations" / "visdrone_train.json",
        "images": VISDRONE / "VisDrone2019-DET-train" / "images",
    },
    "val": {
        "annotations": VISDRONE / "annotations" / "visdrone_val.json",
        "images": VISDRONE / "VisDrone2019-DET-val" / "images",
    },
}

# Twin semantics (must match visdrone_deimv2_800.py exactly): batch_size
# is the PER-DEVICE load batch (8 at 800px), accum 2 -> effective global
# 16 for the sqrt(16/32) LR auto-scale; steps counts loader steps
# (6471 images / 8 = 808.9 per epoch; epochs via --epochs, default 10 =
# the 8088-step pilot).
TRAIN_IMAGES = 6471
LOAD_BATCH = 8
GRAD_ACCUM = 2  # effective global batch 16

MODEL_ARGS = {
    "image_size": (800, 800),
    "num_select": 500,  # dense scenes: widen eval decode (official 300)
    "gradient_clip_val": 0.1,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 500,
    "use_ema_model": True,
    # Pilot-scale EMA — same rationale as the DEIMv2 baseline twin.
    "ema_momentum": 0.993,
    "ema_warmup_steps": 200,
    "eval_max_dets": 500,
}

# Full-paper form (doc 22 §4.5 defaults): DBFSAF backbone fusion +
# SFS-FF with embedded FSD-Down, learnable wavelets, DEIMv2-faithful
# 3-level pyramid. Ablation arms override individual switches.
#
# Backbone pretraining (doc 22 v1.2 erratum): the official DEIMv2-L
# finetunes the LVD-1689M pretrained DINOv3 ViT-S/16 (384-wide — the
# yml's weights_path; the serialized config's "embed_dim: 224" is the
# STA wrapper's PROJECTED width). fsdc-l arms pass the local facebook
# conversion so the native backbone starts pretrained; without it the
# 0.025x backbone LR bucket starves a random ViT (the v1 pilot's flaw).
FSDC_ARGS = {
    "sta_levels": 3,
    "use_dbfsaf": True,
    "use_sfs_ff": True,
    "use_fsd_down": True,
    "freeze_wavelet": False,
    "fdconv_use_ksm": False,
    "fdconv_lite": False,
}

BACKBONE_WEIGHTS = (
    "/home/hdjs/Research/Project/rf-detr-drone/weights/"
    "dinov3-vits16-pretrain-lvd1689m/model.safetensors"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="fsdc-l",
        help="FSDC variant (fsdc-l | fsdc-s)",
    )
    parser.add_argument(
        "--no-dbfsaf",
        action="store_true",
        help="ablation: keep the HF blind fusion, SFS-FF encoder only",
    )
    parser.add_argument("--smoke", action="store_true", help="tiny validation run")
    parser.add_argument(
        "--tag",
        default=None,
        help="output-dir tag override (default: variant [-enc])",
    )
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="skip the DINOv3 ViT pretrained load (random backbone)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="stock DEIMv2-L architecture twin: every FSDC module off "
        "(the pretrained ViT-S load is KEPT — this is the ablation "
        "baseline that isolates the FSDC modules' net contribution)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="training epochs (default 10 = the 8088-step pilot; "
        "30 = the ~24k-step long-budget arms, design doc 22 §8 "
        "SFS-FF sign-flip probe)",
    )
    namespace = parser.parse_args()

    epochs = namespace.epochs
    steps = TRAIN_IMAGES * epochs // LOAD_BATCH  # 6471/8 = 808.9 per epoch
    aug_stop_step = round(steps * 108 / 160)

    if namespace.smoke:
        steps, val_every, save_every, out_name = 60, 30, 60, "smoke"
    else:
        val_every, save_every = 2000, 4000
        out_name = "full" if epochs == 10 else f"{epochs}ep"

    fsdc_args = dict(FSDC_ARGS)
    if namespace.no_dbfsaf:
        # Encoder-only ablation arm (doc 22 §8): DBFSAF off, SFS-FF +
        # embedded FSD-Down stay on — isolates the backbone fusion's
        # contribution from the encoder's.
        fsdc_args["use_dbfsaf"] = False
    if namespace.baseline:
        # Stock-architecture twin (doc 22 §8 fsdc-none arm): the HF 1x1
        # fusion and the original encoder, same pretrained ViT-S — the
        # swap-face-equivalent configuration.
        fsdc_args.update(
            use_dbfsaf=False, use_sfs_ff=False, use_fsd_down=False
        )
    if not namespace.no_pretrained_backbone and namespace.model.startswith(
        "fsdc-l"
    ):
        # Only the fsdc-l family (384-wide ViT-S) consumes the external
        # facebook conversion; fsdc-s's 192-wide ViT comes pretrained
        # INSIDE its DEIMv2-S COCO checkpoint (warm-start path) — passing
        # the 384 file there is a hard shape mismatch.
        fsdc_args["backbone_weights"] = BACKBONE_WEIGHTS

    # fsdc-l has no COCO conversion (PR-0): a warm-start request would be
    # rejected downstream — make the default explicit here.
    load_weights = namespace.model == "fsdc-s"

    variant_tag = namespace.tag or namespace.model.removeprefix("fsdc-")
    if namespace.no_dbfsaf and not namespace.tag:
        variant_tag += "-enc"
    if namespace.baseline and not namespace.tag:
        variant_tag += "-baseline"
    result = train_object_detection(
        out=f"runs/visdrone_fsdc_{variant_tag}_800_{out_name}",
        data=DATA_ARGS,
        model=f"fsdc/{namespace.model}",
        steps=steps,
        batch_size=LOAD_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        model_args={
            **MODEL_ARGS,
            "load_weights": load_weights,
            "fsdc": fsdc_args,
        },
        transform_args={
            "mosaic_prob": None,
            "mixup_prob": None,
            "scale_jitter": {"sizes": [800], "step_stop": aug_stop_step},
        },
        trainer_args={"val_every_num_steps": val_every},
        save_checkpoint_args={"save_every_num_steps": save_every, "save_last": True},
    )
    print("best metrics:", result.best_metrics)


if __name__ == "__main__":
    main()
