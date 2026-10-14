# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone LT-DETR at 800x800 — EdgeCrafter (ECViT) COCO-detector warm start.

Loads the published EdgeCrafter COCO detector checkpoint into backbone +
encoder + decoder. The ECViT checkpoints are the registry's only published
D-FINE-decoder family: the target builds with the variant-spec default
``decoder="dfine_official"`` (gateway architecture, 4 layers, decoder width
bound to the spec — 192 for ecvitt), and ``coco_checkpoint`` loads the ECViT
wrapper state 1:1 (``backbone._model_wrapper.*``, single last-level
projector, persistent rope periods).

Checkpoint (SHA256-verified against the registry pin):
    edgecrafter_ecvitt_ltdetr_coco_260624_f8aefe49.pt  (~40 MB)
Also available: edgecrafter-ecvittplus / -ecvits (-ltdetr-coco) — swap
ECVIT_SIZE below; ecvitt is the EdgeCrafter edge model (ltdetrv2-s).

Training recipe is byte-identical to the convnext-tiny COCO-full line
(``visdrone_ltdetr_800_baseline_coco.py``, best mAP 0.3465 @ 36k) — lr
1.4e-4 @ reference batch 16, EMA 0.9999/1000, augs from step 0, aug-stop
108/160, batch 8, 50 epochs — so the WARM-START FAMILY (ECViT+D-FINE vs
DINOv3-ConvNeXt+rtdetrv2) is the variable under test at equal data/steps.
Backbone LR follows the variant spec (0.05 for ecvitt, not the convnext
0.1) per the upstream EdgeCrafter recipe.

Launch:

    cd /home/hdjs/Research/Project/AutoVision && \
    setsid nohup .venv/bin/python scripts/visdrone_ltdetr_800_edgecrafter_coco.py \
        > runs/visdrone_ltdetr_edgecrafter_ecvitt_800_cocofull_console.log 2>&1 &
"""

import time
from pathlib import Path

VISDRONE = Path("/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone")
DATA_ARGS = {
    "format": "coco",
    "train": {"annotations": str(VISDRONE / "annotations/visdrone_train.json"),
              "images": str(VISDRONE / "VisDrone2019-DET-train/images")},
    "val": {"annotations": str(VISDRONE / "annotations/visdrone_val.json"),
            "images": str(VISDRONE / "VisDrone2019-DET-val/images")},
}

#: Which published EdgeCrafter COCO checkpoint to warm-start from. The
#: resolver accepts either a local file or a registered published name
#: (download + sha256 verify): see
#: ``detr.coco_checkpoint.resolve_detector_checkpoint``.
ECVIT_SIZE = "ecvitt"
DETECTOR_CHECKPOINT = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/weight/edgecrafter"
    f"/edgecrafter_{ECVIT_SIZE}_ltdetr_coco_260624_f8aefe49.pt"
)

# 50-epoch horizon, batch 8 — same as the convnext-tiny COCO-full run.
EPOCHS = 50
TRAIN_IMAGES = 6471
GLOBAL_BATCH = 8
STEPS = (TRAIN_IMAGES * EPOCHS) // GLOBAL_BATCH  # 40,443
AUG_STOP_FRACTION = 108 / 160
AUG_STOP_STEP = round(STEPS * AUG_STOP_FRACTION)  # 27,299

MODEL_ARGS = {
    "backbone": f"edgecrafter-{ECVIT_SIZE}",
    "backbone_pretrained": False,       # backbone comes from the checkpoint
    "detector_checkpoint": DETECTOR_CHECKPOINT,
    "image_size": (800, 800),
    "decoder": "dfine_official",        # checkpoint's gateway decoder family
    "pyramid_levels": 3,
    "num_queries": 300,
    "num_top_queries": 300,
    "num_denoising": 100,
    "label_noise_ratio": 0.5,
    "box_noise_scale": 1.0,
    "reg_max": 32,
    "reg_scale": 4.0,
    # Official eval protocol for the dense benchmark.
    "eval_max_dets": 500,
    # Pilot LR recipe, identical to the convnext-tiny COCO-full run
    # (1.4e-4 -> 9.9e-5 effective at global batch 8). Backbone LR factor
    # stays at the ecvitt spec default (0.05) — the upstream EdgeCrafter
    # recipe, deliberately NOT the convnext 0.1.
    "lr": 1.4e-4,
    "lr_reference_batch": 16,
    "weight_decay": 6.5e-5,
    "gradient_clip_val": 0.1,
    "lr_warmup_steps": 500,
    "lr_scheduler": "multistep",
    "lr_milestones": (0.5, 0.75),
    "lr_gamma": 0.8,
    "ema_momentum": 0.9999,
    "ema_warmup_steps": 1000,
}

TRANSFORM_ARGS = {
    "mosaic_prob": None,
    "mixup_prob": None,
    "photometric_step_start": 0,
    "zoom_out_step_start": 0,
    "iou_crop_step_start": 0,
    "photometric_step_stop": AUG_STOP_STEP,
    "zoom_out_step_stop": AUG_STOP_STEP,
    "iou_crop_step_stop": AUG_STOP_STEP,
    "random_erase_step_stop": AUG_STOP_STEP,
    "scale_jitter": {"sizes": [800], "step_stop": AUG_STOP_STEP},
}


def main() -> None:
    from autovision import train_object_detection

    start = time.perf_counter()
    result = train_object_detection(
        out=f"runs/visdrone_ltdetr_edgecrafter_{ECVIT_SIZE}_800_cocofull",
        data=DATA_ARGS,
        model=f"detr/edgecrafter-{ECVIT_SIZE}",
        steps=STEPS,
        batch_size=GLOBAL_BATCH,
        gradient_accumulation_steps=1,
        num_workers=8,
        model_args=MODEL_ARGS,
        transform_args=TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": 1000, "log_every_n_steps": 500},
        save_checkpoint_args={"save_every_num_steps": 2000},
    )
    print("\n" + "=" * 60)
    print(f"VisDrone LT-DETR 800px {EPOCHS} epochs (EdgeCrafter {ECVIT_SIZE} COCO warm start)")
    print("=" * 60)
    print(f"steps:        {result.steps_run}")
    print(f"best ckpt:    {result.best_checkpoint_path}")
    print(f"best export:  {result.best_exported_model_path}")
    print(f"best metrics: {result.best_metrics}")
    print(f"wall time:    {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
