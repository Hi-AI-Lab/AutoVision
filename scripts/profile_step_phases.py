# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Phase-level timing of one Dome training step (one-off diagnostic).

Builds the exact training model of ``visdrone_ltdetr_dome_800.py``
(SSL + density + SIoU, micro-batch 4 two_crop) and times each phase of
the training step with a sync at every boundary:

    backbone -> encoder -> decoder -> criterion -> backward -> optim+EMA

Usage:
    .venv/bin/python scripts/profile_step_phases.py [--steps 15]
"""

import argparse
import time
from pathlib import Path

import torch

from autovision.tasks.detection.detr.config import (
    DETRObjectDetectionTrainArgs,
)
from autovision.tasks.detection.detr.train_model import (
    DETRObjectDetectionTrain,
)

BACKBONE_WEIGHTS = Path(
    "/home/hdjs/Research/Project/Dino-Dome-Research/weight/dinov3"
    "/convnext-tiny-ltdetr-coco/model.safetensors"
)

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
    "dn_max_gt_cap": 300,
    "reg_scale": 4.0,
    "hidden_dim": 256,
    "enc_dim_feedforward": 1024,
    "enc_expansion": 0.5,
    "enc_depth_mult": 0.34,
    "label_noise_ratio": 0.5,
    "box_noise_scale": 1.0,
    "density": {"max_num_select": 500},
    "ema_momentum": 0.9999,
    "ema_warmup_steps": 1000,
    "eval_max_dets": 500,
    "lr": 1.4e-4,
    "backbone_lr_factor": 0.1,
    "lr_scheduler": "multistep",
    "lr_milestones": (0.5, 0.75),
    "box_regression_iou": "siou",
    "ssl": {"mode": "unsup", "view": "two_crop", "loss_coef": 0.1},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--gt-per-image", type=int, default=60)
    namespace = parser.parse_args()

    args = DETRObjectDetectionTrainArgs(**MODEL_ARGS)
    wrapper = DETRObjectDetectionTrain(args)
    device = torch.device("cuda")
    wrapper.to(device)
    wrapper.train()
    optimizer, scheduler = wrapper.get_optimizer(
        total_steps=1000, global_batch_size=8, train_num_batches=100
    )

    batch = namespace.batch
    images = torch.randn(batch * 2, 3, 800, 800, device=device)  # two_crop
    targets = []
    for _ in range(batch * 2):
        n = namespace.gt_per_image
        targets.append(
            {
                "boxes": torch.rand(n, 4, device=device) * 0.5 + 0.1,
                "labels": torch.randint(0, 10, (n,), device=device),
            }
        )

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    phases = [
        "backbone", "encoder", "decoder", "criterion",
        "backward", "optim+ema",
    ]
    totals = {name: 0.0 for name in phases}

    def sync_clock():
        torch.cuda.synchronize()
        return time.perf_counter()

    for step in range(namespace.steps + 3):
        marks = {}
        t0 = sync_clock()
        with autocast:
            feats = list(wrapper.model.backbone(images))
        marks["backbone"] = sync_clock()
        with autocast:
            outs, defe_info = wrapper.model.encoder(feats, targets=targets)
        marks["encoder"] = sync_clock()
        with autocast:
            out = wrapper.model.decoder(outs, targets=targets, defe_info=defe_info)
            if wrapper.model.ssl_head is not None:
                q = out["query_features"]
                out["ssl_z1"] = wrapper.model.ssl_head(q)
                out["ssl_z2"] = wrapper.model.ssl_head(q)
        marks["decoder"] = sync_clock()
        wrapper.criterion.ssl_step = step
        outputs_list = list(out.get("aux_outputs", [])) + [
            {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
        ]
        with autocast:
            loss_dict = wrapper.criterion(
                outputs_list,
                targets,
                world_size=1,
                dn_outputs=out.get("dn_aux_outputs", []),
                dn_meta=out.get("dn_meta"),
                dn_pre_outputs=out.get("dn_pre_outputs"),
                corners_list=out.get("pred_corners_list"),
                ref_boxes_list=out.get("ref_boxes_list"),
                dn_corners_list=out.get("dn_pred_corners_list"),
                dn_ref_boxes_list=out.get("dn_ref_boxes_list"),
                query_mask=out.get("query_valid_mask"),
                defe_info=defe_info,
                pre_outputs=out.get("pre_outputs"),
                enc_aux_outputs=out.get("enc_aux_outputs"),
                enc_meta=out.get("enc_meta"),
                ssl_z1=out.get("ssl_z1"),
                ssl_z2=out.get("ssl_z2"),
            )
        marks["criterion"] = sync_clock()
        optimizer.zero_grad(set_to_none=True)
        loss_dict["loss"].backward()
        marks["backward"] = sync_clock()
        wrapper.clip_gradients(optimizer)
        optimizer.step()
        scheduler.step()
        wrapper.on_train_batch_end()
        marks["optim+ema"] = sync_clock()
        if step >= 3:
            for i, name in enumerate(phases):
                start = t0 if i == 0 else marks[phases[i - 1]]
                totals[name] += marks[name] - start
            t0 = marks["optim+ema"]

    print(f"\n{namespace.steps} timed steps | micro-batch {namespace.batch} "
          f"(two_crop -> {batch * 2} views) | gt/image {namespace.gt_per_image}")
    total = sum(totals.values())
    for name in phases:
        seconds = totals[name] / namespace.steps
        print(f"  {name:<10} {seconds * 1000:8.1f} ms  ({seconds / total * 100:4.1f} %)")
    print(f"  {'TOTAL':<10} {total / namespace.steps * 1000:8.1f} ms per step")


if __name__ == "__main__":
    main()
