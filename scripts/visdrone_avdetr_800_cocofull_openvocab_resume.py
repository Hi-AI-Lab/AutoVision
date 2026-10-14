# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Resume helper for the G3 OV arm (design doc 27 §13.4 G3).

Reuses ``visdrone_ltdetr_800_cocofull_openvocab.py`` verbatim and resumes
from the run's ``last.ckpt`` into the SAME output directory
(``resume_interrupted=True`` keeps the previous artifacts — metrics.jsonl
continuity preserved). The engine restores optimizer/scheduler/EMA/step.

The caller (the bash watcher in the launch block below) only fires this
once the GPU has enough free memory again — the original run died from
GPU CONTENTION (a co-tenant process took ~8 GiB while our solo peak is
~23.6 GiB on the 24 GiB card), not from the recipe.

Launch (waits for >= 20 GiB free, then resumes):

    cd /home/hdjs/Research/Project/AutoVision && \
    setsid nohup bash -c 'while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits);
        [ "$free" -ge 20000 ] && break; sleep 60;
      done;
      AUTOVISION_MIRROR_URL=https://hf-mirror.com \
      AUTOVISION_CACHE_DIR=/home/hdjs/Research/Project/AutoVision/.autovision-cache \
      .venv/bin/python scripts/visdrone_ltdetr_800_cocofull_openvocab_resume.py' \
      > runs/visdrone_ltdetr_convnextt_800_cocofull_ov_resume_console.log 2>&1 &
"""

import importlib.util
import time
from pathlib import Path

RUN_DIR = "runs/visdrone_ltdetr_convnextt_800_cocofull_ov"
LAST_CKPT = f"{RUN_DIR}/checkpoints/last.ckpt"


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "ov_arm", Path(__file__).parent / "visdrone_ltdetr_800_cocofull_openvocab.py"
    )
    arm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arm)

    from autovision import train_object_detection

    start = time.perf_counter()
    result = train_object_detection(
        out=RUN_DIR,
        data=arm.DATA_ARGS,
        model="avdetr/dinov3-convnext-tiny",
        steps=arm.STEPS,
        batch_size=arm.GLOBAL_BATCH,
        gradient_accumulation_steps=1,
        num_workers=8,
        model_args=arm.MODEL_ARGS,
        transform_args=arm.TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": 1000, "log_every_n_steps": 500},
        save_checkpoint_args={"save_every_num_steps": 2000},
        checkpoint=LAST_CKPT,
        resume_interrupted=True,
    )
    print("\n" + "=" * 60)
    print("VisDrone LT-DETR 800px (COCO FULL warm start + open-vocab head) — RESUMED")
    print("=" * 60)
    print(f"steps:        {result.steps_run}")
    print(f"best ckpt:    {result.best_checkpoint_path}")
    print(f"best export:  {result.best_exported_model_path}")
    print(f"best metrics: {result.best_metrics}")
    print(f"wall time:    {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
