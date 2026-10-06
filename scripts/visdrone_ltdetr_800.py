# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone LT-DETR baseline at 800x800 — density-OFF twin of the Dome run.

Exact ablation twin of ``scripts/visdrone_ltdetr_dome_800.py``: identical
backbone (COCO-detection ConvNeXt-tiny, safetensors), decoder (dfine),
Dome-S optimizer/EMA/warmup recipe, 64,710-step schedule, and augmented
windows. The ONLY difference is the absence of ``density`` — fixed
top-``num_queries`` proposals, no DeFE / MWAS / dynamic NMS (design doc
13 §8 ablation arm a).

10-epoch pilot recipe v3 (symmetric in both twins; aligned with the
official Dome-S-VisDrone run where noted):
- Distributional bin space aligned with official D-FINE v2: 33 non-uniform
  bins (``reg_max + 1``) over the weighting function ``W`` with
  ``reg_scale: 4.0`` — the previous linear ±1w/65-bin port clamped FGL
  targets on VisDrone's tiny boxes (fgl_dn stuck ~8, never converged);
- LR schedule ``multistep`` (official Dome: ×0.8 at epochs 80/120 of 160 →
  here ×0.8 at 50 %/75 % of the run; LR never drops below 64 % of peak).
  Deep cosine + restart tail measurably damaged end-of-run mAP (pilot2:
  0.1033 peak → 0.0842 final, online 0.0535);
- ``gradient_accumulation_steps: 1`` (official trains batch 8 with 809
  optimizer steps/epoch; batch 16 × accum 4 gave only 404);
- ``hidden_dim: 256`` (official Dome-S encoder/decoder width);
- EMA 0.995 kept for the pilot (0.9999/warmup 1000 on the 160-epoch run,
  official values).

Launch:

    cd /home/hdjs/Research/Project/AutoVision && \
    .venv/bin/python scripts/visdrone_ltdetr_800.py
"""

import json
import time
from pathlib import Path

from autovision import train_object_detection

VISDRONE = Path("/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone")
# Module-level data config (shared with diag_ema_vs_online.py).
DATA_ARGS = {
    "format": "coco",
    "train": {"annotations": str(VISDRONE / "annotations/visdrone_train.json"),
              "images": str(VISDRONE / "VisDrone2019-DET-train/images")},
    "val": {"annotations": str(VISDRONE / "annotations/visdrone_val.json"),
            "images": str(VISDRONE / "VisDrone2019-DET-val/images")},
}
# COCO-detection-finetuned ConvNeXt-tiny backbone (LT-DETR/COCO extract),
# loaded directly from the safetensors source of truth.
BACKBONE_WEIGHTS = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/weight/dinov3"
    "/convnext-tiny-ltdetr-coco/model.safetensors"
)

# Same schedule as the Dome run: 160 epochs x 6471 images at global
# batch 16; heavy augs stop at Dome's epoch 108/160 boundary.
EPOCHS = 10  # 10-epoch pilot (full Dome recipe: 160)
TRAIN_IMAGES = 6471
# Pilot v4 trains at the OFFICIAL Dome-S global batch (8 = 809 optimizer
# steps/epoch); bf16 + hidden 256 fits batch 8 at 800px on the 24GB card.
GLOBAL_BATCH = 8
STEPS = (TRAIN_IMAGES * EPOCHS) // GLOBAL_BATCH  # 8,088 @ 10 epochs
AUG_STOP_FRACTION = 108 / 160
AUG_STOP_STEP = round(STEPS * AUG_STOP_FRACTION)  # 5,459 @ 10 epochs

# Official Dome-S LR recipe scaled to the pilot: ×0.8 at 50 %/75 % of the
# run (Dome's epochs 80/120 of 160). The LR never drops below 64 % of
# peak — deep cosine annealing plus the flat-cosine restart tail
# measurably damaged end-of-run mAP in the pilots.
LR_MILESTONES = (0.5, 0.75)
LR_GAMMA = 0.8

# Module-level so the twin-consistency test can diff the two scripts
# programmatically (the ONLY allowed difference vs the Dome script is
# the absence of "density").
MODEL_ARGS = {
    "backbone": "dinov3-convnext-tiny",
    "backbone_weights": BACKBONE_WEIGHTS,
    "image_size": (800, 800),
    # Dome twin: dfine decoder, 250 fixed queries (= Dome
    # min_num_select), same denoising/reg_max recipe.
    "decoder": "dfine",
    # Dome-faithful 4-level pyramid (strides 4/8/16/32) — design doc 13 D1.
    "pyramid_levels": 4,
    "num_queries": 250,
    "num_top_queries": 500,
    "reg_max": 32,
    "num_denoising": 100,
    # Official D-FINE v2 weighting-function curvature (33 non-uniform bins
    # per side in w/4 units — aligned decode + FGL target spaces).
    "reg_scale": 4.0,
    # Official Dome-S encoder/decoder width.
    "hidden_dim": 256,
    # Official Dome-S HybridEncoder "S" recipe (the ConvNeXt variant spec
    # defaults are 2048/1.0/1.0 — a 4-6x heavier RANDOM-initialized
    # encoder that slows early convergence).
    "enc_dim_feedforward": 1024,
    "enc_expansion": 0.5,
    "enc_depth_mult": 0.34,
    # NOTE: num_denoising is inert on dense batches — num_group floors to
    # 1 once max_gt_num > num_denoising, so the DN query count is
    # 2 × max_gt_num regardless (VisDrone regularly exceeds 100 GT/image).
    "label_noise_ratio": 0.5,
    # Pilot: halved CDN noise. Noise scales with box size, and the
    # negative queries land 0.5-1 box-size away — their FGL residuals
    # saturate the reg_max bins on VisDrone's tiny boxes (fgl_dn ~7 and
    # flat across the whole pilot).
    # Official CDN noise (restored: the W-space FGL targets no longer
    # saturate, so the old pilot overrides are obsolete).
    "box_noise_scale": 1.0,
    # NO "density" key — the ablation variable.
    # Dome optimizer recipe (identical to the Dome run).
    # LR halved from the official 2e-4 equivalent: pilot4's peak-LR phase
    # (1.98e-4) was measurably unstable (mAP dipped 0.0723 -> 0.0439 at
    # epochs 2.5-3.7 and only recovered once the schedule decayed to
    # 1.58e-4). 1.4e-4 sqrt-scales to 9.9e-5 — the stable regime.
    "lr": 1.4e-4,
    "backbone_lr_factor": 0.1,
    "weight_decay": 6.5e-5,
    "gradient_clip_val": 0.1,
    "lr_warmup_steps": 500,
    # Official Dome-S MultiStepLR (×0.8 at 80/120 of 160 epochs).
    "lr_scheduler": "multistep",
    "lr_milestones": LR_MILESTONES,
    "lr_gamma": LR_GAMMA,
    # Pilot: 0.995 — 0.999's ~1k-step EMA window averages in early
    # weights, a measured +9.5% mAP drag at 4k steps. Restore 0.999
    # (official: 0.9999, warmup 1000) for the 160-epoch run.
    "ema_momentum": 0.995,
    "ema_warmup_steps": 200,
    "eval_max_dets": 500,
}
# Identical augmentations to the Dome run: no mosaic / no mixup,
# photometric 0.5 / zoom-out 0.5 / IoU-crop 0.8 / hflip 0.5,
# scale locked at the val resolution (pilot), heavy augs stop at
# epoch 108/160.
TRANSFORM_ARGS = {
    "mosaic_prob": None,
    "mixup_prob": None,
    "photometric_step_stop": AUG_STOP_STEP,
    "zoom_out_step_stop": AUG_STOP_STEP,
    "iou_crop_step_stop": AUG_STOP_STEP,
    "scale_jitter": {
        # Pilot: locked at 800 — the 576-960 ladder sends ~43% of batches
        # below the val resolution and undersamples VisDrone's small
        # objects. Restore the ladder (cap 960: the 4-level pyramid's
        # 1024px tail peaks ~20GB on the 24GB card) for the 160-epoch run:
        # [s for s in ltdetr_scale_jitter_sizes((800, 800), 32) if s <= 960]
        "sizes": [800],
        "step_stop": AUG_STOP_STEP,
    },
}


def main() -> None:
    start = time.perf_counter()
    result = train_object_detection(
        out="runs/visdrone_ltdetr_convnextt_base_800",
        data=DATA_ARGS,
        model="ltdetr/dinov3-convnext-tiny",
        steps=STEPS,
        batch_size=8,
        gradient_accumulation_steps=1,
        num_workers=8,
        model_args=MODEL_ARGS,
        transform_args=TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": 1000, "log_every_n_steps": 500},
        save_checkpoint_args={"save_every_num_steps": 2000},
    )
    print("\n" + "=" * 60)
    print("VisDrone LT-DETR base 800px (dinov3-convnext-tiny, density OFF)")
    print("=" * 60)
    print(f"steps:        {result.steps_run}")
    print(f"best ckpt:    {result.best_checkpoint_path}")
    print(f"best export:  {result.best_exported_model_path}")
    print(f"best metrics: {json.dumps(result.best_metrics, indent=2)}")
    print(f"wall time:    {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
