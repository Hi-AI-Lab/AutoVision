# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone LT-DETR baseline at 800x800 — 50-epoch horizon of the pilot.

Every hyperparameter is inherited verbatim from
``scripts/visdrone_ltdetr_800.py`` (importlib, the diag-script pattern):
model/optimizer/EMA recipe, twin transform windows, backbone weights.
Only the horizon changes — 10 -> 50 epochs — plus the derived absolute
step counts:

- ``STEPS = 6471 x 50 / 8 = 40 443`` (global batch 8, ~0.62 s/step on the
  24 GB card -> ~7 h);
- heavy-aug stop ``round(STEPS * 108/160) = 27 299`` (Dome cadence, the
  fraction is horizon-agnostic);
- ``lr_milestones (0.5, 0.75)`` and ``lr_gamma 0.8`` are fractions of the
  run (official Dome epochs 80/120 of 160) — no re-derivation needed.

This is the first horizon launched with the baseline-parity loss fix
(dfine aux-layer L1+GIoU deep supervision) and augs running from step 0
(`*_step_start = 0`). Reference point: the 10-epoch pilot of the same
recipe peaked at 0.2588 val mAP (runs/..._pilot5_v2).

Launch (survives the shell; log to the run dir):

    cd /home/hdjs/Research/Project/AutoVision && \
    setsid nohup .venv/bin/python scripts/visdrone_ltdetr_800_50ep.py \
        > runs/visdrone_ltdetr_convnextt_base_800_50ep_console.log 2>&1 &
"""

import importlib.util
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_pilot():
    spec = importlib.util.spec_from_file_location(
        "visdrone_ltdetr_800", REPO / "scripts" / "visdrone_ltdetr_800.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = _load_pilot()

EPOCHS = 50
TRAIN_IMAGES = pilot.TRAIN_IMAGES
GLOBAL_BATCH = pilot.GLOBAL_BATCH
STEPS = (TRAIN_IMAGES * EPOCHS) // GLOBAL_BATCH  # 40,443 @ 50 epochs
AUG_STOP_FRACTION = pilot.AUG_STOP_FRACTION
AUG_STOP_STEP = round(STEPS * AUG_STOP_FRACTION)  # 27,299 @ 50 epochs

OUT = "runs/visdrone_ltdetr_convnextt_base_800_50ep"

MODEL_ARGS = dict(pilot.MODEL_ARGS)
TRANSFORM_ARGS = {
    **pilot.TRANSFORM_ARGS,
    # Re-derive the absolute stop steps for the 50-epoch horizon (the
    # fraction stays Dome's 108/160; the starts stay pinned at 0).
    "photometric_step_stop": AUG_STOP_STEP,
    "zoom_out_step_stop": AUG_STOP_STEP,
    "iou_crop_step_stop": AUG_STOP_STEP,
    "random_erase_step_stop": AUG_STOP_STEP,
    "scale_jitter": {
        **pilot.TRANSFORM_ARGS["scale_jitter"],
        "step_stop": AUG_STOP_STEP,
    },
}


def main() -> None:
    start = time.perf_counter()
    result = pilot.train_object_detection(
        out=OUT,
        data=pilot.DATA_ARGS,
        model="avdetr/dinov3-convnext-tiny",
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
    print(f"VisDrone LT-DETR base 800px {EPOCHS} epochs (density OFF)")
    print("=" * 60)
    print(f"steps:        {result.steps_run}")
    print(f"best ckpt:    {result.best_checkpoint_path}")
    print(f"best export:  {result.best_exported_model_path}")
    print(f"best metrics: {result.best_metrics}")
    print(f"wall time:    {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
