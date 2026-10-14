# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone RF-DETR baseline at 800x800.

Mirrors ``scripts/visdrone_ltdetr_800.py`` (same data paths, 10-epoch
pilot, global batch 8, scale jitter locked at 800, no mosaic/mixup) with
the RF-DETR family: HF ``RfDetrForObjectDetection`` fine-tuned from the
verified Roboflow checkpoint with the official recipe (AdamW lr 1e-4 /
encoder 1.5e-4 layer-decayed 0.8, cosine schedule, EMA 0.993).

Requires transformers >= 5.9 (RfDetr classes). Run with the overlay env:

    PYTHONPATH=/tmp/torchshim:src AUTOVISION_WEIGHT_SOURCE=auto \
        /tmp/rf590env/bin/python scripts/visdrone_rfdetr_800.py [--smoke]
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

TRAIN_IMAGES = 6471
GLOBAL_BATCH = 8
MICRO_BATCH = 4  # 800px + group queries peak ~20GB at batch 8; batch 4 x accum 2
GRAD_ACCUM = GLOBAL_BATCH // MICRO_BATCH
#: Pilot aug-stop cadence (Dome 108/160). Fine for 10-epoch pilots; long
#: runs should raise it (--aug-stop-fraction 0.9) — a 50-epoch run with the
#: pilot fraction trains ~16 final epochs augmentation-free and measurably
#: overfits (peak mAP lands BEFORE the stop, then decays ~0.7 mAP).
PILOT_AUG_STOP_FRACTION = 108 / 160
#: VisDrone multiscale ladder (official rf-detr-style): all multiples of
#: the nano input divisor 32. Single-scale 800 undertrains small objects —
#: the diagnosed AP_small = 0.16 vs AP_medium 0.39 vs AP_large 0.62.
MULTISCALE_SIZES = [640, 704, 768, 832, 896, 960]

MODEL_ARGS = {
    "image_size": (800, 800),
    "num_queries": 300,
    "num_select": 500,  # dense scenes: widen eval decode (official 300)
    "lr": 1e-4,
    "lr_encoder": 1.5e-4,
    "weight_decay": 1e-4,
    "gradient_clip_val": 0.1,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 500,
    "use_ema_model": True,
    "ema_momentum": 0.993,
    "ema_warmup_steps": 200,
    "eval_max_dets": 500,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny validation run")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="training epochs (default 10 pilot; the cosine schedule and "
        "the aug-stop step both scale with this)",
    )
    parser.add_argument(
        "--aug-stop-fraction",
        type=float,
        default=PILOT_AUG_STOP_FRACTION,
        help="fraction of the run after which heavy augs + scale jitter "
        "turn off (default 108/160 = pilot parity; use 0.9 for long runs)",
    )
    parser.add_argument(
        "--multiscale",
        action="store_true",
        help=f"train on the VisDrone multiscale ladder {MULTISCALE_SIZES} "
        "instead of the locked 800 canvas (val stays 800 either way)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=800,
        help="square train/val canvas (default 800; must be divisible by "
        "32; 960 costs ~1.4x step time)",
    )
    parser.add_argument(
        "--erase",
        action="store_true",
        help="box-aware background erasure (V2 constrained mode: every "
        "object keeps >= 2 patches; rf-detr-ssl-v2 design doc 8) — the "
        "SSL-line patch erasure, train views only",
    )
    parser.add_argument(
        "--erase-val",
        action="store_true",
        help="secondary eval channel on erased val views (doc 16 P1.6): "
        "every val image is box-aware erased (V2 recipe, path-seeded "
        "deterministic masks) and scored against the unchanged dense GT — "
        "logs val_erased_metric/* next to the untouched dense "
        "val_metric/*; doubles validation cost, independent of --erase",
    )
    parser.add_argument(
        "--siou",
        action="store_true",
        help="swap the box loss IoU term GIoU -> SIoU (design doc 15 "
        "semantics: emission-only, matcher costs stay GIoU; loss weight "
        "inherits loss_giou)",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        help="per-object contrastive SSL branch, upstream M4 recommended "
        "start: unsup / two_crop / loss_coef=0.1 (design doc 14, rfdetr "
        "port — doc 18); two_crop doubles the per-forward images to 2B",
    )
    parser.add_argument(
        "--sigreg-coef",
        type=float,
        default=0.0,
        help="SIGReg λ_sr (requires --ssl). 0 (default) = read-only "
        "calibration: the unweighted Epps–Pulley statistic still logs as "
        "train_loss/loss_sigreg_stat — pick λ_sr so λ_sr·stat is "
        "comparable to λ_ssl·loss_ssl, then re-run with this flag",
    )
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=MICRO_BATCH,
        help="per-forward batch (grad-accum scales back to the global "
        "batch). Measured on a 4090D @800px: ~1GB/image baseline, but "
        "--ssl two_crop adds contrastive + SIGReg + dense-batch criterion "
        "spikes — under --ssl keep micro 4 (8 imgs/forward, ~8GB); "
        "micro 8 OOMed after the first validation cycle",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="dataloader workers; two_crop transforms every image twice, "
        "so underfed workers show up as low GPU utilization (<50%)",
    )
    namespace = parser.parse_args()
    # Keep the global batch at 8 whatever the per-forward batch is.
    grad_accum = max(1, GLOBAL_BATCH // namespace.micro_batch)

    if namespace.resolution % 32:
        raise SystemExit(
            f"--resolution {namespace.resolution} must be divisible by 32 "
            "(rf-detr-nano patch 16 x 2 windows)."
        )
    MODEL_ARGS["image_size"] = (namespace.resolution, namespace.resolution)

    aug_stop_step = round(
        TRAIN_IMAGES * namespace.epochs // GLOBAL_BATCH * namespace.aug_stop_fraction
    )

    scale_jitter_sizes = (
        list(MULTISCALE_SIZES) if namespace.multiscale else [namespace.resolution]
    )
    transform_args = {
        "mosaic_prob": None,
        "mixup_prob": None,
        "scale_jitter": {"sizes": scale_jitter_sizes, "step_stop": aug_stop_step},
    }

    if namespace.smoke:
        steps, val_every, save_every = 60, 30, 60
        out_name = "smoke"
    else:
        steps = TRAIN_IMAGES * namespace.epochs // GLOBAL_BATCH
        val_every, save_every = 1000, 2000
        out_name = "full"
    flags = [
        name
        for name, on in (
            ("erase", namespace.erase),
            ("siou", namespace.siou),
            ("multiscale", namespace.multiscale),
            ("ssl", namespace.ssl),
            ("eraseval", namespace.erase_val),
        )
        if on
    ]
    if namespace.sigreg_coef > 0:
        flags.append(f"sr{namespace.sigreg_coef:g}")
    if namespace.aug_stop_fraction != PILOT_AUG_STOP_FRACTION:
        flags.append(f"as{round(namespace.aug_stop_fraction * 100)}")
    if flags:
        out_name += "-" + "-".join(flags)
    if not namespace.smoke and namespace.epochs != 10:
        out_name += f"-{namespace.epochs}ep"
    if namespace.resolution != 800:
        out_name += f"-{namespace.resolution}px"

    model_args = dict(MODEL_ARGS)
    if namespace.siou:
        model_args["box_regression_iou"] = "siou"
    if namespace.erase_val:
        # V2 recipe mirrors --erase (padding 0.15 / cell 16 stay at the
        # config defaults); deterministic path-seeded masks.
        model_args["eval_erase"] = {
            "ratio": 0.4,
            "interior": True,
            "min_keep": 2,
        }
    if namespace.ssl:
        model_args["ssl"] = {
            "mode": "unsup",
            "view": "two_crop",
            "loss_coef": 0.1,
        }
        if namespace.sigreg_coef > 0:
            model_args["ssl"]["sigreg_coef"] = namespace.sigreg_coef

    if namespace.erase:
        transform_args.update(
            {
                "random_erase_prob": 0.5,
                "random_erase_ratio": 0.4,
                "random_erase_interior": True,
                "random_erase_min_keep": 2,
                # Erasure follows the same aug-stop cadence (doc 16 P1.5).
                "random_erase_step_stop": aug_stop_step,
            }
        )

    result = train_object_detection(
        out=f"runs/visdrone_rfdetr_nano_800_{out_name}",
        data=DATA_ARGS,
        model="rfdetr/rf-detr-nano",
        steps=steps,
        batch_size=namespace.micro_batch,
        gradient_accumulation_steps=grad_accum,
        num_workers=namespace.num_workers,
        model_args=model_args,
        transform_args=transform_args,
        overwrite=True,
        trainer_args={
            "val_every_num_steps": val_every,
            "log_every_n_steps": max(1, val_every // 2),
        },
        save_checkpoint_args={"save_every_num_steps": save_every},
    )
    print("=== run summary ===")
    print("steps_run:", result.steps_run)
    print("best checkpoint:", result.best_checkpoint_path)
    print("best export:", result.best_exported_model_path)
    if result.best_metrics:
        print("best metrics:", result.best_metrics)


if __name__ == "__main__":
    main()
