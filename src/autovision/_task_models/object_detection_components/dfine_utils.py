# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""D-FINE utilities: distributional box regression + FGL/DDF loss math.

D-FINE (Peng et al., "D-FINE: Redefine Regression Task in DETRs as
Fine-grained Distribution Refinement", arXiv:2410.13842) replaces the box
delta head of RT-DETR with a *distributional* head: every decoder layer
predicts, per query and per box side, a categorical distribution over
``2 × reg_max + 1`` bins, and the box is obtained by integrating the
distribution over a fixed bin grid. Later layers predict a *residual*
distribution relative to the previous layer's decoded box — the
fine-grained distribution refinement (FDR) loop.

Distance parameterization (self-implemented per the paper's mechanism):

- distances are measured from the current reference box's own **edges** to
  the target box's edges (l, t, r, b) and normalized by the reference box's
  width (l, r) and height (t, b) — scale-invariant, a residual of one full
  reference dimension per layer is representable, and a zero distance
  vector reproduces the reference box exactly (the zero-initialized head
  makes refinement start at the reference);
- bin k of the distribution carries the weight
  ``(k - reg_max) / reg_max ∈ [-1, 1]``; signed values are required because
  the target box may extend beyond the reference box on either side;
- :func:`distance2bbox` maps (reference, distances) → cxcywh,
  :func:`bbox2distance` is its inverse (FGL target build);
- distances outside ``[-1, 1]`` saturate at the boundary bins (same
  clamping semantics as the official ``bbox2distance``); a too-coarse
  layer-0 reference is corrected by the next refinement layer, whose
  reference is already closer to the object.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from autovision._task_models.object_detection_components.denoising import (
    box_cxcywh_to_xyxy,
)

__all__ = [
    "bbox2distance",
    "ddf_loss",
    "distance2bbox",
    "fg_loss",
    "integration_weights",
    "integrate_corners",
    "matched_pair_iou",
    "soft_bin_targets",
]

_MIN_REF_DIM = 1e-4


def integration_weights(reg_max: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Bin grid of ``2 × reg_max + 1`` weights spanning ``[-1, 1]``."""
    bins = torch.arange(2 * reg_max + 1, device=device, dtype=dtype) - reg_max
    return bins / reg_max


def integrate_corners(corners: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Integrate per-side distributions into distances.

    ``corners``: (…, 4 × num_bins) raw logits, ``weights``: (num_bins,).
    Returns (…, 4) distances in reference-dimension units.
    """
    logits = corners.reshape(*corners.shape[:-1], 4, weights.shape[-1])
    probs = torch.softmax(logits, dim=-1)
    return (probs * weights).sum(dim=-1)


def distance2bbox(
    ref_boxes: torch.Tensor, distances: torch.Tensor, eps: float = _MIN_REF_DIM
) -> torch.Tensor:
    """Reference (…, 4) cxcywh + edge distances (…, 4) in reference units →
    cxcywh. Zero distances reproduce the reference box exactly."""
    cx, cy, w, h = ref_boxes.unbind(-1)
    w = w.clamp(min=eps)
    h = h.clamp(min=eps)
    x1 = (cx - w / 2) - distances[..., 0] * w
    y1 = (cy - h / 2) - distances[..., 1] * h
    x2 = (cx + w / 2) + distances[..., 2] * w
    y2 = (cy + h / 2) + distances[..., 3] * h
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def bbox2distance(
    ref_boxes: torch.Tensor, gt_boxes_xyxy: torch.Tensor, eps: float = _MIN_REF_DIM
) -> torch.Tensor:
    """Inverse of :func:`distance2bbox` (FGL target build).

    ``ref_boxes`` (N, 4) cxcywh, ``gt_boxes_xyxy`` (N, 4) absolute xyxy on
    the unit canvas. Returns (N, 4) edge distances in reference units,
    clamped to ``[-1, 1]``.
    """
    cx, cy, w, h = ref_boxes.unbind(-1)
    w = w.clamp(min=eps)
    h = h.clamp(min=eps)
    x1_ref, y1_ref = cx - w / 2, cy - h / 2
    x2_ref, y2_ref = cx + w / 2, cy + h / 2
    x1, y1, x2, y2 = gt_boxes_xyxy.unbind(-1)
    return torch.stack(
        [(x1_ref - x1) / w, (y1_ref - y1) / h, (x2 - x2_ref) / w, (y2 - y2_ref) / h],
        dim=-1,
    ).clamp(min=-1.0, max=1.0)


def soft_bin_targets(
    distances: torch.Tensor, reg_max: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Continuous bin positions with left/right interpolation (DFL style).

    ``distances``: (N, 4) in ``[-1, 1]``. Returns ``(bin_left, bin_right,
    weight_left, weight_right)`` — long/long/float/float, each (N, 4); the
    target distribution puts ``weight_left`` on ``bin_left`` and
    ``weight_right`` on ``bin_right``.

    The bin clamp happens AFTER the floor: clamping the continuous
    position to ``2·reg_max - 1e-6`` is a no-op in float32 (the value
    rounds back up to ``2·reg_max``), which pushed ``right`` one bin past
    the distribution on exact-distance targets and corrupted the gather.
    """
    positions = ((distances + 1.0) * reg_max).clamp(0.0, 2.0 * reg_max)
    left = torch.minimum(
        positions.floor(), torch.full_like(positions, 2.0 * reg_max - 1.0)
    )
    right = left + 1
    weight_left = (right - positions).clamp(0.0, 1.0)
    weight_right = 1.0 - weight_left
    return left.long(), right.long(), weight_left, weight_right


def matched_pair_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Row-aligned IoU of matched pairs (normalized cxcywh inputs)."""
    from torchvision.ops import box_iou

    ious = torch.diag(
        box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes))
    )
    return ious.clamp(min=0.0)


def fg_loss(
    corners: torch.Tensor,
    ref_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    num_boxes: torch.Tensor,
    reg_max: int,
) -> torch.Tensor:
    """Fine-grained Localization loss (FGL) for one decoder layer.

    DFL-style soft-label cross-entropy over the four side distributions,
    with IoU-aware weighting: each matched query contributes its side CE
    scaled by the matched IoU (high-quality boxes dominate — the paper's
    quality-aware target). Targets are computed relative to THIS layer's
    reference box (the layer-0 anchor lock of the paper falls out of the
    refinement loop: layer 0's reference is the encoder proposal).

    ``corners``: (B, Q, 4 × num_bins) raw logits; ``ref_boxes``: (B, Q, 4)
    the reference each query refined from (detached); ``pred_boxes``:
    (B, Q, 4) this layer's decoded boxes (IoU weighting).
    """
    num_bins = 2 * reg_max + 1
    bins = corners.reshape(corners.shape[0], corners.shape[1], 4, num_bins)
    log_probs = torch.log_softmax(bins, dim=-1)
    loss = corners.sum() * 0.0
    for batch_index, (src, dst) in enumerate(indices):
        if src.numel() == 0:
            continue
        gt_xyxy = box_cxcywh_to_xyxy(
            targets[batch_index]["boxes"][dst].to(ref_boxes.device)
        )
        reference = ref_boxes[batch_index, src].detach()
        distances = bbox2distance(reference, gt_xyxy)
        left, right, weight_left, weight_right = soft_bin_targets(distances, reg_max)
        layer_log_probs = log_probs[batch_index, src]  # (n, 4, bins)
        lp_left = layer_log_probs.gather(2, left.unsqueeze(-1)).squeeze(-1)
        lp_right = layer_log_probs.gather(2, right.unsqueeze(-1)).squeeze(-1)
        side_loss = -(weight_left * lp_left + weight_right * lp_right)  # (n, 4)
        quality = matched_pair_iou(
            pred_boxes[batch_index, src].detach(),
            targets[batch_index]["boxes"][dst].to(pred_boxes.device),
        )
        loss = loss + (side_loss * quality.unsqueeze(-1)).sum()
    return loss / num_boxes.clamp(min=1)


def ddf_loss(
    student_corners: torch.Tensor,
    teacher_corners: torch.Tensor,
    student_boxes: torch.Tensor,
    teacher_boxes: torch.Tensor,
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    num_boxes: torch.Tensor,
    reg_max: int,
) -> torch.Tensor:
    """Decoupled Distillation Focal loss (DDF) for one intermediate layer.

    The final layer's side distributions act as the teacher for earlier
    layers (the "decoupled" part: only the regression branch is distilled,
    classification is untouched). Per matched query the student's
    distribution is pulled toward the teacher's with a KL divergence
    (``KL(T ‖ S)`` — the teacher entropy is subtracted so the loss is
    exactly zero once the student matches the teacher, for any teacher
    distribution), weighted by the paper's quality-focal modulation:

    ``w = IoU(student box, teacher box)² × (1 - Σ T·S)``

    — high-agreement pairs carry the distillation signal, and the weight
    vanishes as the distributions converge (the loss self-extinguishes).
    Teacher tensors are detached; the weight is detached (modulation only).
    """
    num_bins = 2 * reg_max + 1
    teacher_log_probs = torch.log_softmax(
        teacher_corners.reshape(teacher_corners.shape[0], teacher_corners.shape[1], 4, num_bins),
        dim=-1,
    ).detach()
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = torch.log_softmax(
        student_corners.reshape(student_corners.shape[0], student_corners.shape[1], 4, num_bins),
        dim=-1,
    )
    student_probs = student_log_probs.exp()
    loss = student_corners.sum() * 0.0
    for batch_index, (src, _dst) in enumerate(indices):
        if src.numel() == 0:
            continue
        teacher = teacher_probs[batch_index, src]
        divergence = (teacher * (
            teacher_log_probs[batch_index, src] - student_log_probs[batch_index, src]
        )).sum(dim=-1)  # (n, 4)
        agreement = (teacher * student_probs[batch_index, src]).sum(dim=-1).detach()
        quality = matched_pair_iou(
            student_boxes[batch_index, src].detach(),
            teacher_boxes[batch_index, src].detach(),
        )
        weight = (quality**2).unsqueeze(-1) * (1.0 - agreement)
        loss = loss + (divergence * weight).sum()
    return loss / num_boxes.clamp(min=1)
