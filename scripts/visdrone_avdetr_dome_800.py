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
- EMA pilot 保持 0.995(全量跑回官方 0.9999/warmup 1000)。

100-epoch 全量跑(本版本):STEPS=80,887 @ global batch 8,增广停用
54,599 步(108/160 比例),EMA 恢复官方 0.9999/warmup 1000,scale 锁定
800(SSL two_crop 前向翻倍,960 阶梯会 OOM)。三个消融开关:
- ``--erase``  box-aware 背景擦除(设计文档 16,V2 约束模式);
- ``--ssl``    per-object 对比 SSL,上游 M4 推荐起点 unsup/two_crop/
               λ=0.1(设计文档 14 §8;two_crop 单步前向 = 2B 张图);
- ``--siou``   box 回归 IoU 项 GIoU→SIoU(设计文档 15,D-FINE 专用,
               loss 系数继承 loss_giou 权重,E0 探针 0.953 无需缩放)。
显存探针(决定 --ssl 跑 batch 8 还是 4×accum2):

    cd /home/hdjs/Research/Project/AutoVision && \
    .venv/bin/python scripts/visdrone_ltdetr_dome_800.py \
        --probe-steps 20 --ssl --erase --siou

显存注意(24GB 卡实测,前次 100ep 运行):20 步探针峰值 ~18.6GiB reserved,
但长跑会因碎片化持续爬升——step ~550 曾触发 allocator OOM 警告(free 28MiB,
靠清缓存自愈)。验证是 fp32 前向(与训练同 batch),是最可能硬 OOM 的时刻;
engine 现在在验证前后 empty_cache 缓解。若重跑仍见 OOM 警告:
    a) --micro-batch 2 --grad-accum 4(global batch 仍为 8,激活减半);
    b) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,\
garbage_collection_threshold:0.80(阈值以上主动回收缓存块);
    c) 二者都不够时须砍 SSL two_crop(2B 前向是最大内存项)。
Launch(全开):

    cd /home/hdjs/Research/Project/AutoVision && \
    AUTOVISION_MIRROR_URL=https://hf-mirror.com \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python scripts/visdrone_ltdetr_dome_800.py \
        --ssl --erase --siou
"""

import argparse
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

# Dome-S-VisDrone: 160 epochs x 6471 train images. This run: 100 epochs at
# the same global batch 8 → 6471*100/8 optimizer steps.
EPOCHS = 100
TRAIN_IMAGES = 6471
# Pilot v4 trains at the OFFICIAL Dome-S global batch (8 = 809 optimizer
# steps/epoch); bf16 + hidden 256 fits batch 8 at 800px on the 24GB card.
# With --ssl the two_crop pipeline doubles the per-forward images (2B=16);
# if the probe OOMs, drop to --micro-batch 4 --grad-accum 2 (identical
# optimizer-step math: 8 pairs per optimizer step).
GLOBAL_BATCH = 8
STEPS = (TRAIN_IMAGES * EPOCHS) // GLOBAL_BATCH  # 80,887 @ 100 epochs
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
    "num_queries": 250,  # = Dome min_num_select
    "num_top_queries": 500,
    "reg_max": 32,
    "num_denoising": 100,
    # Crash guard: VisDrone's densest images (up to 902 GT) push the DN
    # query count to ~1800 and the decoder self-attention OOMs a 24 GB
    # card. Capping DN slots at 300/image only affects batches above the
    # cap (~0.2% of images) and only their DN supervision; typical batches
    # are bit-identical. Measured: GT=900 batch OOMs unbounded, fits capped.
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
    # --- Density branch (Dome-S-VisDrone: min 250 / max 500 / win 10) ---
    "density": {"max_num_select": 500},
    # --- Dome optimizer recipe (include/optimizer.yml + S overrides) ---
    # LR halved from the official 2e-4 equivalent: pilot4's peak-LR phase
    # (1.98e-4) was measurably unstable (mAP dipped 0.0723 -> 0.0439 at
    # epochs 2.5-3.7 and only recovered once the schedule decayed to
    # 1.58e-4). 1.4e-4 sqrt-scales to 9.9e-5 — the stable regime.
    "lr": 1.4e-4,
    "backbone_lr_factor": 0.1,  # Dome backbone lr 2e-5 / 2e-4
    "weight_decay": 6.5e-5,
    "gradient_clip_val": 0.1,
    "lr_warmup_steps": 500,  # Dome LinearWarmup warmup_duration 500
    # Official Dome-S MultiStepLR (×0.8 at 80/120 of 160 epochs).
    "lr_scheduler": "multistep",
    "lr_milestones": LR_MILESTONES,
    "lr_gamma": LR_GAMMA,
    # Dome ModelEMA decay 0.9999, warmups 1000 (include/optimizer.yml).
    # Restored for the 100-epoch full run: 0.995's ~200-step EMA window
    # was a pilot-only setting — at 80k steps the official 10k-step
    # window tracks the converged weights instead of averaging in the
    # early ones.
    "ema_momentum": 0.9999,
    "ema_warmup_steps": 1000,
    # LR sqrt-scaling reference batch. The recipe's documented effective
    # LR (1.4e-4 -> 9.9e-5 at global batch 8) assumes the fork-historical
    # denominator 16; pin it so the family-wide baseline-parity default
    # (32) can never silently rescale a resumed run.
    "lr_reference_batch": 16,
    # VisDrone official eval protocol (Dome maxDets=500).
    "eval_max_dets": 500,
    # Per-object contrastive SSL + SIGReg (design docs 14/16/18) —
    # identical to visdrone_ltdetr_800.py so density stays the only
    # ablation variable (full comment block there). λ_sr=0.1 is the
    # doc-18 reference pending calibration; two_crop doubles the
    # per-forward images, so the micro-batch defaults below halve to
    # compensate.
    "ssl": {
        "mode": "unsup",
        "view": "two_crop",
        "loss_coef": 0.1,
        "projector": "mlp2",
        "sigreg_coef": 0.1,
    },
}
# Dome-S-VisDrone effective augmentations: photometric 0.5 / zoom-out
# 0.5 / IoU-crop 0.8 / hflip 0.5 (family defaults already match), NO
# mosaic / NO mixup (the include/dataloader.yml overrides drop them),
# scale locked at the val resolution (pilot), all heavy augs stop at
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
    # (doc 16; identical keys in visdrone_ltdetr_800.py so density stays
    # the only ablation variable): prob 0.5 / ratio 0.4, constrained-
    # interior mode (every object keeps >= 2 patches; rf-detr-ssl-v2
    # design doc 8). Val stays erase-free. The stop below follows the
    # Dome aug-stop cadence (doc 16 P1.5): constant ratio until
    # 108/160, then off — the run ends on the dense distribution it
    # deploys with. ("auto" would resolve to the generic epoch-88 stop,
    # not the Dome cadence.)
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--erase",
        action="store_true",
        help="box-aware background erasure (V2 constrained mode: every "
        "object keeps >= 2 patches; rf-detr-ssl-v2 design doc 8); now "
        "ON by default in TRANSFORM_ARGS (shared twin recipe), so this "
        "flag only re-applies the same values — kept for CLI "
        "compatibility",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        help="per-object contrastive SSL branch + SIGReg (design docs "
        "14/16/18); now ON by default in MODEL_ARGS (shared twin "
        "recipe), so this flag is a no-op kept for CLI compatibility",
    )
    parser.add_argument(
        "--siou",
        action="store_true",
        help="swap the box loss IoU term GIoU -> SIoU (design doc 15; "
        "D-FINE decoder only; loss weight inherits loss_giou)",
    )
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=4,
        help="per-forward batch; the always-on ssl two_crop doubles it "
        "to 2B images (8 at the default). rfdetr 4090D @800px "
        "measurement: micro 8 under ssl OOMed after the first "
        "validation cycle — keep micro 4. grad-accum scales the "
        "optimizer step back to the Dome global batch",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
        help="gradient accumulation steps (micro_batch x accum should "
        "equal GLOBAL_BATCH to keep the Dome optimizer recipe; the "
        "default 2 restores global batch 8 from micro 4)",
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=None,
        help="run only this many steps into a scratch out dir and print "
        "peak CUDA memory (launch-decision probe for the two_crop "
        "pipeline)",
    )
    namespace = parser.parse_args()

    transform_args = dict(TRANSFORM_ARGS)
    if namespace.erase:
        transform_args.update(
            {
                "random_erase_prob": 0.5,
                "random_erase_ratio": 0.4,
                "random_erase_interior": True,
                "random_erase_min_keep": 2,
            }
        )

    model_args = dict(MODEL_ARGS)
    # NOTE: the former --ssl branch replaced model_args["ssl"] wholesale,
    # which under the shared twin recipe would silently drop the
    # projector/sigreg keys (sigreg_coef back to 0.0). SSL now lives in
    # MODEL_ARGS; --ssl is a CLI-compatibility no-op.
    if namespace.siou:
        # Design doc 15: siou_loss_coef=None inherits the loss_giou weight
        # (E0 probe: loss-scale ratio 0.953 — no down-scaling needed);
        # siou_theta keeps the paper default 4.
        model_args["box_regression_iou"] = "siou"

    flags = [
        name
        for name, on in (
            ("ssl", namespace.ssl),
            ("erase", namespace.erase),
            ("siou", namespace.siou),
        )
        if on
    ]
    if namespace.probe_steps is not None:
        out = "runs/probe_dome_mem"
        steps = namespace.probe_steps
        checkpoint_args = None
    else:
        out = "runs/visdrone_ltdetr_convnextt_dome_800_100ep" + (
            "-" + "-".join(flags) if flags else ""
        )
        steps = STEPS
        # Aligned with the 1000-step validation cadence: the previous run
        # was interrupted at step ~1005 with save_every 2000 — no last.ckpt
        # existed yet, so 1000 steps of training were unrecoverable. A
        # 1000-step checkpoint bounds the loss to one interval for the cost
        # of one ~0.6GB atomic write per 1000 steps (<1% overhead at
        # 2.5 s/step).
        checkpoint_args = {"save_every_num_steps": 1000}

    start = time.perf_counter()
    result = train_object_detection(
        out=out,
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
        steps=steps,
        batch_size=namespace.micro_batch,
        gradient_accumulation_steps=namespace.grad_accum,
        num_workers=8,
        model_args=model_args,
        transform_args=transform_args,
        trainer_args={"val_every_num_steps": 1000, "log_every_n_steps": 500},
        save_checkpoint_args=checkpoint_args,
    )
    if namespace.probe_steps is not None:
        import torch

        peak_alloc = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30
        print("\n" + "=" * 60)
        print(f"probe: {namespace.probe_steps} steps, flags={flags or ['none']}")
        print(f"micro_batch={namespace.micro_batch} grad_accum={namespace.grad_accum}")
        print(f"peak allocated: {peak_alloc:.1f} GiB")
        print(f"peak reserved:  {peak_reserved:.1f} GiB")
        print("=" * 60)
        return
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
