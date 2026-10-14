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
# The always-on ssl two_crop doubles the per-forward images, so main()
# runs micro 4 x accum 2 — same global batch, same optimizer-step count,
# same 8-images/forward load the card already fits (rfdetr 4090D
# measurement: micro 8 under ssl OOMed after the first validation cycle).
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
    # Twin parity with visdrone_ltdetr_dome_800.py: crash guard for
    # VisDrone's 500+ GT images (DN slots capped at 300/image; typical
    # batches bit-identical). Must match the dome script so density stays
    # the only ablation variable.
    "dn_max_gt_cap": 300,
    # Trigger-guarded fix for the documented dense-batch failure mode
    # (config dn_fgl_loss_weight docstring: dn FGL can dominate without
    # converging). None = inherit the main FGL weight 0.15 (official).
    # Monitored by the metrics automation:
    #   - flip to 0.05 if loss_fgl_dn_0 * 0.15 / loss > 0.20, or
    #   - flip to 0.05 if val mAP < 0.01 at step 4000.
    # Flipping requires a fresh run (loss semantics change; no resume).
    "dn_fgl_loss_weight": None,

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
    # LR sqrt-scaling reference batch. The recipe's documented effective
    # LR (1.4e-4 -> 9.9e-5 at global batch 8) assumes the fork-historical
    # denominator 16; pin it so the family-wide baseline-parity default
    # (32) can never silently rescale a resumed run.
    "lr_reference_batch": 16,
        # Official Dome-S EMA (include/optimizer.yml), restored with the Dome
    # twin for the 100-epoch run: 0.995's ~200-step window was a
    # pilot-only setting — at 80k steps the official 10k-step window
    # tracks the converged weights instead of averaging in the early
    # ones. The exponential ramp (decay × (1 − e^−k/warmup)) keeps the
    # early EMA tracking recent weights, so the old pilot concern does
    # not apply to the official pair.
    "ema_momentum": 0.9999,
    "ema_warmup_steps": 1000,
    "eval_max_dets": 500,
    # Per-object contrastive SSL + SIGReg (design docs 14/16/18), part of
    # the shared twin recipe (identical key in the Dome script — the twin
    # test only tolerates the "density" difference). Upstream M4 start:
    # unsup / two_crop / λ_ssl=0.1. SIGReg (doc 16 R1-A1) at the doc-18
    # reference λ_sr=0.1 forces projector='mlp2' (the simcse head's raw
    # tap is post-Tanh, bounded — N(0,I) unreachable) and view/matched_gt
    # constraints; λ_sr itself is the pre-calibration reference (doc 18
    # workflow: read λ_sr·loss_sigreg_stat vs λ_ssl·loss_ssl from the
    # logs, then adjust). two_crop doubles the per-forward images to 2B —
    # main() halves the micro-batch to compensate. The command layer
    # auto-disables IoU-crop/zoom-out under two_crop (doc 18 §2:
    # independent geometric crops wipe cross-view pairing on VisDrone
    # tiny boxes), so the InfoNCE view difference comes from the
    # box-aware erasure + photometric/flip.
    "ssl": {
        "mode": "unsup",
        "view": "two_crop",
        "loss_coef": 0.1,
        "projector": "mlp2",
        "sigreg_coef": 0.1,
    },
}
# Identical augmentations to the Dome run: no mosaic / no mixup,
# photometric 0.5 / zoom-out 0.5 / IoU-crop 0.8 / hflip 0.5,
# scale locked at the val resolution (pilot), heavy augs stop at
# epoch 108/160.
TRANSFORM_ARGS = {
    "mosaic_prob": None,
    "mixup_prob": None,
    # Heavy augs run from step 0 (official Dome/D-FINE: transforms are
    # always on; the generic "auto" start holds them out for the first
    # ~30% of a short run).
    "photometric_step_start": 0,
    "zoom_out_step_start": 0,
    "iou_crop_step_start": 0,
    "photometric_step_stop": AUG_STOP_STEP,
    "zoom_out_step_stop": AUG_STOP_STEP,
    "iou_crop_step_stop": AUG_STOP_STEP,
    # Box-aware background erasure, part of the shared twin recipe
    # (doc 16): prob 0.5 / ratio 0.4, constrained-interior mode (every
    # object keeps >= 2 patches; rf-detr-ssl-v2 design doc 8). Val stays
    # erase-free (the loader clears random_erase_prob). The stop below
    # follows the Dome aug-stop cadence (doc 16 P1.5) instead of the
    # generic "auto" resolution (epoch 88).
    "random_erase_prob": 0.5,
    "random_erase_ratio": 0.4,
    "random_erase_interior": True,
    "random_erase_min_keep": 2,
    "random_erase_step_stop": AUG_STOP_STEP,
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
        # Recipe suffix per the Dome script's flag convention: this is the
        # ssl+sigreg+erase recipe (fresh dir — the two_crop pipeline and
        # micro-batching are not compatible with the old run's layout).
        out="runs/visdrone_ltdetr_convnextt_base_800_ssl_erase_sr0.1",
        data=DATA_ARGS,
        model="detr/dinov3-convnext-tiny",
        steps=STEPS,
        # micro 4 x accum 2 keeps the Dome global batch 8 under the
        # two_crop doubling (see the GLOBAL_BATCH note above).
        batch_size=4,
        gradient_accumulation_steps=2,
        num_workers=8,
        model_args=MODEL_ARGS,
        transform_args=TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": 1000, "log_every_n_steps": 500},
        save_checkpoint_args={"save_every_num_steps": 2000},
    )
    print("\n" + "=" * 60)
    print("VisDrone LT-DETR base 800px (dinov3-convnext-tiny, density OFF, ssl+sigreg+erase)")
    print("=" * 60)
    print(f"steps:        {result.steps_run}")
    print(f"best ckpt:    {result.best_checkpoint_path}")
    print(f"best export:  {result.best_exported_model_path}")
    print(f"best metrics: {json.dumps(result.best_metrics, indent=2)}")
    print(f"wall time:    {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
