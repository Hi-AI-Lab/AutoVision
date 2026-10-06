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

10-epoch pilot 配方 v3(两个 twin 脚本对称;对齐官方 Dome-S-VisDrone 实跑):
- 分布 bin 空间对齐官方 D-FINE v2:每边 33 个非均匀 bin(reg_max+1),
  加权函数 W 曲率 reg_scale=4.0——旧线性 ±1w/65-bin 移植在 VisDrone
  小目标上 FGL 目标被钳死(fgl_dn 钉在 ~8 不收敛);
- LR 调度 multistep(官方 Dome:×0.8@epochs 80/120 of 160 → 本 pilot
  ×0.8@50%/75%;LR 终点保持峰值 64%)。深度 cosine+重启尾段实测有害
  (pilot2:峰值 0.1033 → 终点 0.0842,online 仅 0.0535);
- gradient_accumulation_steps=1(官方 batch 8 → 809 步/epoch;
  batch 16 × accum 4 只有 404 步);
- hidden_dim=256(官方 Dome-S encoder/decoder 宽度);
- EMA pilot 保持 0.995(160-epoch 全量跑回官方 0.9999/warmup 1000)。
Launch:

    cd /home/hdjs/Research/Project/AutoVision && \
    .venv/bin/python scripts/visdrone_ltdetr_dome_800.py
"""

import json
import time
from pathlib import Path

from autovision import train_object_detection

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
# Pilot v4 trains at the OFFICIAL Dome-S global batch (8 = 809 optimizer
# steps/epoch); bf16 + hidden 256 fits batch 8 at 800px on the 24GB card.
GLOBAL_BATCH = 8
STEPS = (TRAIN_IMAGES * EPOCHS) // GLOBAL_BATCH  # 8,088 @ 10 epochs
# Dome stops photometric/zoom-out/IoU-crop and multiscale at epoch 108/160.
AUG_STOP_FRACTION = 108 / 160
AUG_STOP_STEP = round(STEPS * AUG_STOP_FRACTION)  # 43,679

# Official Dome-S LR recipe scaled to the pilot: ×0.8 at 50 %/75 % of the
# run (Dome's epochs 80/120 of 160). The LR never drops below 64 % of
# peak — deep cosine annealing plus the flat-cosine restart tail
# measurably damaged end-of-run mAP in the pilots.
LR_MILESTONES = (0.5, 0.75)
LR_GAMMA = 0.8

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
    # --- Density branch (Dome-S-VisDrone: min 250 / max 500 / win 10) ---
    "density": {"max_num_select": 500},
    # --- Dome optimizer recipe (include/optimizer.yml + S overrides) ---
    # LR halved from the official 2e-4 equivalent: pilot4's peak-LR phase
    # (1.98e-4) was measurably unstable (mAP dipped 0.0723 -> 0.0439 at
    # epochs 2.5-3.7 and only recovered once the schedule decayed to
    # 1.58e-4). 1.4e-4 sqrt-scales to 9.9e-5 — the stable regime.
    "lr": 1.4e-4,
    "backbone_lr_factor": 0.1,     # Dome backbone lr 2e-5 / 2e-4
    "weight_decay": 6.5e-5,
    "gradient_clip_val": 0.1,
    "lr_warmup_steps": 500,        # Dome LinearWarmup warmup_duration 500
    # Official Dome-S MultiStepLR (×0.8 at 80/120 of 160 epochs).
    "lr_scheduler": "multistep",
    "lr_milestones": LR_MILESTONES,
    "lr_gamma": LR_GAMMA,
    # Dome ModelEMA decay 0.9999, warmups 1000 (include/optimizer.yml)
    # Pilot: 0.995 — 0.999's ~1k-step EMA window averages in early
    # weights, a measured +9.5% mAP drag at 4k steps. Restore 0.9999
    # (official) for the 160-epoch run.
    "ema_momentum": 0.995,
    "ema_warmup_steps": 200,
    # VisDrone official eval protocol (Dome maxDets=500).
    "eval_max_dets": 500,
}
# Dome-S-VisDrone effective augmentations: photometric 0.5 / zoom-out
# 0.5 / IoU-crop 0.8 / hflip 0.5 (family defaults already match), NO
# mosaic / NO mixup (the include/dataloader.yml overrides drop them),
# scale locked at the val resolution (pilot), all heavy augs stop at
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
        batch_size=8,
        gradient_accumulation_steps=1,
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
