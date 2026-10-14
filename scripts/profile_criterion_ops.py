# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Op-level profile of one Dome criterion call (one-off diagnostic).

Runs the exact criterion stack of ``visdrone_ltdetr_dome_800.py`` under
``torch.profiler`` and prints the ops with the largest SELF CPU time —
the launch-bound per-image loops show up as many small aten ops.

Usage:
    .venv/bin/python scripts/profile_criterion_ops.py [--gt-per-image 60]
"""

import argparse
from pathlib import Path

import torch

from autovision.tasks.detection.avdetr.config import (
    AVDETRObjectDetectionTrainArgs,
)
from autovision.tasks.detection.avdetr.train_model import (
    AVDETRObjectDetectionTrain,
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


def build_batch(batch: int, gt_per_image: int, device):
    images = torch.randn(batch * 2, 3, 800, 800, device=device)
    targets = []
    for _ in range(batch * 2):
        targets.append(
            {
                "boxes": torch.rand(gt_per_image, 4, device=device) * 0.5 + 0.1,
                "labels": torch.randint(0, 10, (gt_per_image,), device=device),
            }
        )
    return images, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-per-image", type=int, default=60)
    parser.add_argument("--steps", type=int, default=6)
    args_ns = parser.parse_args()

    args = AVDETRObjectDetectionTrainArgs(**MODEL_ARGS)
    wrapper = AVDETRObjectDetectionTrain(args)
    device = torch.device("cuda")
    wrapper.to(device)
    wrapper.train()

    images, targets = build_batch(4, args_ns.gt_per_image, device)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def run_step(step: int) -> torch.Tensor:
        with autocast:
            feats = list(wrapper.model.backbone(images))
            outs, defe_info = wrapper.model.encoder(feats, targets=targets)
            out = wrapper.model.decoder(outs, targets=targets, defe_info=defe_info)
            q = out["query_features"]
            out["ssl_z1"] = wrapper.model.ssl_head(q)
            out["ssl_z2"] = wrapper.model.ssl_head(q)
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
        return loss_dict["loss"]

    for step in range(3):  # warmup
        loss = run_step(step)
        loss.backward()
        wrapper.model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    # no_grad: attribution of the criterion's CPU-side loops only — without
    # it the per-step autograd graphs pile up and OOM the 24 GB card.
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for step in range(3, 3 + args_ns.steps):
            with torch.no_grad():
                loss = run_step(step)
            del loss
        torch.cuda.synchronize()
    print(
        f"\n=== criterion-side ops, {args_ns.steps} calls, "
        f"gt/image={args_ns.gt_per_image} — top 25 by SELF CPU time ==="
    )
    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=25, max_name_column_width=55
        )
    )


if __name__ == "__main__":
    main()
