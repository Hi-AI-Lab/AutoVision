# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""SSL-off control short run: quantify two_crop's early gradient tax.

3000-step twin of the Dome run with ONLY ``--ssl`` removed (erase + siou
kept), validated at 1000/2000/3000. At those steps the EMA window is ~2-5
updates (decay 0.9999 ramping under warmup 1000), so ``val_metric/map``
is effectively an ONLINE-weights measurement — a clean read of early
convergence, uncontaminated by EMA lag.

Decision rule — compare three noise-robust quantities against the main
run (the early DETR phase transition shifts mAP by orders of magnitude
at any FIXED step, so a single-step ±20% band is meaningless):
1. transition step: first validation with mAP > 0.05 (main: 2000);
2. mAP at step 3000 (both arms safely past transition);
3. loss_bbox / loss_vfl at matched steps (continuous, no phase jump).
Verdict: transition within ~1.5× AND mAP@3000 within factor 2 AND matched
losses comparable → no material SSL tax. Otherwise two_crop's gradient
dilution is real: consider every_n_steps interleave or delayed SSL
warmup (design doc 14 has the stride knob) in the next run.

Run ONLY when the GPU is free (the main training holds ~20 GiB):

    cd /home/hdjs/Research/Project/AutoVision && \\
    AUTOVISION_MIRROR_URL=https://hf-mirror.com \\
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    .venv/bin/python scripts/visdrone_ltdetr_dome_ssl_control.py
"""

import argparse
import json
import time
from pathlib import Path

from autovision import train_object_detection

VISDRONE = Path("/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone")
BACKBONE_WEIGHTS = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/weight/dinov3"
    "/convnext-tiny-ltdetr-coco/model.safetensors"
)

STEPS = 3000

MODEL_ARGS = {
    "backbone": "dinov3-convnext-tiny",
    "backbone_weights": BACKBONE_WEIGHTS,
    "image_size": (800, 800),
    "decoder": "dfine",
    "pyramid_levels": 4,
    "num_queries": 250,
    "num_top_queries": 500,
    "reg_max": 32,
    "num_denoising": 100,
    "reg_scale": 4.0,
    "hidden_dim": 256,
    "enc_dim_feedforward": 1024,
    "enc_expansion": 0.5,
    "enc_depth_mult": 0.34,
    "label_noise_ratio": 0.5,
    "box_noise_scale": 1.0,
    "density": {"max_num_select": 500},
    "lr": 1.4e-4,
    "backbone_lr_factor": 0.1,
    "weight_decay": 6.5e-5,
    "gradient_clip_val": 0.1,
    "lr_warmup_steps": 500,
    "lr_scheduler": "multistep",
    "lr_milestones": (0.5, 0.75),
    "lr_gamma": 0.8,
    "ema_momentum": 0.9999,
    "ema_warmup_steps": 1000,
    "eval_max_dets": 500,
    "dn_max_gt_cap": 300,
    "dn_fgl_loss_weight": None,
    # NO "ssl" key — this is the control arm.
    "box_regression_iou": "siou",
}

TRANSFORM_ARGS = {
    "mosaic_prob": None,
    "mixup_prob": None,
    # step_stop=None keeps every scheduled aug active for the whole 2000
    # steps — the main run's AUG_STOP_STEP (43,679) is equivalent over this
    # window, so the arms differ ONLY by the two_crop views.
    "photometric_step_stop": None,
    "zoom_out_step_stop": None,
    "iou_crop_step_stop": None,
    "random_erase_step_stop": None,
    "random_erase_prob": 0.5,
    "random_erase_ratio": 0.4,
    "random_erase_interior": True,
    "random_erase_min_keep": 2,
    "scale_jitter": {"sizes": [800], "step_stop": None},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--micro-batch", type=int, default=4, help="keep 4×accum 2 = global 8"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=2, help="keep 4×accum 2 = global 8"
    )
    namespace = parser.parse_args()

    start = time.perf_counter()
    result = train_object_detection(
        out="runs/probe_ssl_off_control",
        overwrite=True,
        data={
            "format": "coco",
            "train": {
                "annotations": str(VISDRONE / "annotations/visdrone_train.json"),
                "images": str(VISDRONE / "VisDrone2019-DET-train/images"),
            },
            "val": {
                "annotations": str(VISDRONE / "annotations/visdrone_val.json"),
                "images": str(VISDRONE / "VisDrone2019-DET-val/images"),
            },
        },
        model="avdetr/dinov3-convnext-tiny",
        steps=STEPS,
        batch_size=namespace.micro_batch,
        gradient_accumulation_steps=namespace.grad_accum,
        num_workers=8,
        model_args=MODEL_ARGS,
        transform_args=TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": 1000, "log_every_n_steps": 500},
    )
    print("\n" + "=" * 60)
    print("SSL-off control (erase + siou kept, two_crop removed) finished")
    print("=" * 60)
    print(f"steps:       {result.steps_run}")
    print(f"best ckpt:   {result.best_checkpoint_path}")
    print(f"best mAP:    {json.dumps(result.best_metrics.get('val_metric/map'))}")
    print(f"wall time:   {time.perf_counter() - start:.0f}s")
    print("compare: transition step / mAP@3000 / matched loss_bbox vs the main run.")


if __name__ == "__main__":
    main()
