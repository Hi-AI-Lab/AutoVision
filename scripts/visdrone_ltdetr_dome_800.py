# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""VisDrone LT-DETR + density-guided queries (Dome-DETR port) at 800x800.

Hyperparameters follow the Dome-DETR Dome-S VisDrone recipe
(``Dome-DETR-master/configs/dome/Dome-S-VisDrone.yml`` + its includes) as
closely as the VTrain engine allows:

=============================================  ====================================  ====================
Dome-DETR (effective config)                   本脚本                                 说明
=============================================  ====================================  ====================
min/max_num_select 250 / 500                   num_queries=250 / density.max=500     相同
mwas_window_size 10                            density 默认                           相同
defe_density_map_weight 1                      density 默认                           相同
NMS IoU = 0.4 + 0.5*density                    density 默认                           相同
num_denoising 100 / noise 0.5 / 1.0            显式传入                               相同
reg_max 32                                     显式传入                               相同
dec_layers 3, num_points (4,4,4)               convnext-tiny spec 默认                 相同(3 层金字塔)
AdamW lr 2e-4, backbone 2e-5 (x0.1)            lr=2e-4, backbone_lr_factor=0.1       相同(全局 batch 16 vs Dome 8)
weight_decay 6.5e-5, betas (0.9, 0.999)        显式传入                               相同
clip_max_norm 0.1                              默认                                   相同
EMA 0.9999, warmup 1000                        显式传入                               相同
LinearWarmup 500                               lr_warmup_steps=500                    相同
160 epochs @ batch 8 (=6471*160 images)        steps=64_710 @ batch 16                同样本量、一半更新次数
增广停用 @ epoch 108/160 (67.5%)               *_step_stop=43_679                     相同
multiscale base 800 ±25%                       家族阶梯(800 base, 30 槽)              等价
photometric 0.5 / zoomout 0.5 / ioucrop 0.8    家族默认                               相同
无 Mosaic / 无 MixUp(生效配置)                 显式关闭(mosaic_prob/mixup_prob=None)  相同
seed 0                                         默认                                   相同
=============================================  ====================================  ====================

有意偏离(引擎契约,设计文档 13 §5 D9):MultiStepLR(80/120, γ0.8)→
flat-cosine(平坦 50% 后余弦衰减到 1% 尾部);Dome 的 EMA restart-on-aug-stop
无对应机制;HGNetv2-B0 → DINOv3 ConvNeXt-tiny;4 层金字塔 → 3 层。

注意:与 ``visdrone_ltdetr_800.py`` 基线脚本相比,本脚本同时改了优化器
与调度(Dome 配方)。若做 density 开/关消融,基线必须用同一套优化器参数
重跑,只有 ``density`` 是唯一变量。
Launch:

    cd /home/hdjs/Research/Project/AutoVision && \
    .venv/bin/python scripts/visdrone_ltdetr_dome_800.py
"""

import json
import time
from pathlib import Path

from autovision import train_object_detection
from autovision._task_models.ltdetr_object_detection.transforms import (
    ltdetr_scale_jitter_sizes,
)

VISDRONE = Path("/home/hdjs/Research/Project/Dino-Dome-Research/dataset/visdrone")
# COCO-detection-finetuned ConvNeXt-tiny backbone, loaded directly from
# the safetensors source of truth — shared with the baseline twin
# (visdrone_ltdetr_800.py) so density on/off is the only ablation variable.
BACKBONE_WEIGHTS = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/weight/dinov3"
    "/convnext-tiny-ltdetr-coco/model.safetensors"
)

# Dome-S-VisDrone: 160 epochs x 6471 train images. At global batch 16 the
# same number of sample presentations is 6471*160/16 optimizer steps.
EPOCHS = 10  # 10-epoch pilot (full Dome recipe: 160)
TRAIN_IMAGES = 6471
GLOBAL_BATCH = 16
STEPS = (TRAIN_IMAGES * EPOCHS) // GLOBAL_BATCH  # 4,044 @ 10 epochs
# Dome stops photometric/zoom-out/IoU-crop and multiscale at epoch 108/160.
AUG_STOP_FRACTION = 108 / 160
AUG_STOP_STEP = round(STEPS * AUG_STOP_FRACTION)  # 43,679

# Module-level so the twin-consistency test can diff the two scripts
# programmatically (vs visdrone_ltdetr_800.py: identical except density).
MODEL_ARGS = {
    "backbone": "dinov3-convnext-tiny",
    "backbone_weights": BACKBONE_WEIGHTS,
    "image_size": (800, 800),
    "decoder": "dfine",
    # Dome-faithful 4-level pyramid (strides 4/8/16/32) — design doc 13 D1.
    "pyramid_levels": 4,
    # --- Dome-S-VisDrone decoder/criterion params ---
    "num_queries": 250,            # = Dome min_num_select
    "num_top_queries": 500,
    "reg_max": 32,
    "num_denoising": 100,
    "label_noise_ratio": 0.5,
    "box_noise_scale": 1.0,
    # --- Density branch (Dome-S-VisDrone: min 250 / max 500 / win 10) ---
    "density": {"max_num_select": 500},
    # --- Dome optimizer recipe (include/optimizer.yml + S overrides) ---
    "lr": 2e-4,
    "backbone_lr_factor": 0.1,     # Dome backbone lr 2e-5 / 2e-4
    "weight_decay": 6.5e-5,
    "gradient_clip_val": 0.1,
    "lr_warmup_steps": 500,        # Dome LinearWarmup warmup_duration 500
    # Dome ModelEMA decay 0.9999, warmups 1000 (include/optimizer.yml)
    "ema_momentum": 0.999,
    "ema_warmup_steps": 200,
    # VisDrone official eval protocol (Dome maxDets=500).
    "eval_max_dets": 500,
}
# Dome-S-VisDrone effective augmentations: photometric 0.5 / zoom-out
# 0.5 / IoU-crop 0.8 / hflip 0.5 (family defaults already match), NO
# mosaic / NO mixup (the include/dataloader.yml overrides drop them),
# multiscale on base 800, all heavy augs stop at epoch 108/160.
TRANSFORM_ARGS = {
    "mosaic_prob": None,
    "mixup_prob": None,
    "photometric_step_stop": AUG_STOP_STEP,
    "zoom_out_step_stop": AUG_STOP_STEP,
    "iou_crop_step_stop": AUG_STOP_STEP,
    "scale_jitter": {
        # Capped at 960: the 4-level pyramid's 1024px ladder tail peaks
        # ~20GB and brushes the allocator limit on the 24GB card.
        "sizes": [s for s in ltdetr_scale_jitter_sizes((800, 800), 32) if s <= 960],
        "step_stop": AUG_STOP_STEP,
    },
}


def main() -> None:
    start = time.perf_counter()
    result = train_object_detection(
        out="runs/visdrone_ltdetr_convnextt_dome_800",
        data={
            "format": "coco",
            "train": {"annotations": str(VISDRONE / "annotations/visdrone_train.json"),
                      "images": str(VISDRONE / "VisDrone2019-DET-train/images")},
            "val": {"annotations": str(VISDRONE / "annotations/visdrone_val.json"),
                    "images": str(VISDRONE / "VisDrone2019-DET-val/images")},
        },
        model="ltdetr/dinov3-convnext-tiny",
        steps=STEPS,
        batch_size=4,
        gradient_accumulation_steps=4,
        num_workers=8,
        model_args=MODEL_ARGS,
        transform_args=TRANSFORM_ARGS,
        trainer_args={"val_every_num_steps": 1000, "log_every_n_steps": 500},
        save_checkpoint_args={"save_every_num_steps": 2000},
    )
    print("\n" + "=" * 60)
    print("VisDrone LT-DETR Dome 800px (dinov3-convnext-tiny) finished")
    print("=" * 60)
    print(f"steps:        {result.steps_run}")
    print(f"best ckpt:    {result.best_checkpoint_path}")
    print(f"best export:  {result.best_exported_model_path}")
    print(f"best metrics: {json.dumps(result.best_metrics, indent=2)}")
    print(f"wall time:    {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
