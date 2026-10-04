# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Geometry-coupled crop helpers for the PaKA dense relational loss.

These are data-augmentation utilities (not SSL math): they build extra views
whose crop geometry depends on another view's crop, which the standard
stateless ``ViewTransform`` cannot express. Views carry an 8-element
``geometry`` tensor ``[x0, y0, x1, y1, image_w, image_h, hflip, vflip]`` in
original-image pixels so the PaKA loss can ROI-align crops against each
other.

- ``_GeometryViewTransform``: a single view (random resized crop + flips +
  photometric + normalize) that records its crop geometry.
- ``render_clean_global``: augmentation-free re-render of a recorded global
  crop (the PaKA clean teacher).
- ``sample_high_overlap_box``: small local box fully inside a parent global
  crop (containment = 1.0 satisfies the PaKA overlap constraint).
- ``render_augmented_local``: photometric-augmented render of such a box.
"""

from __future__ import annotations

import math
import random

import torch
from torch import Tensor, nn
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

from autovision._transforms.transform_args import (
    ColorJitterArgs,
    GaussianBlurArgs,
    NormalizeArgs,
    SolarizeArgs,
)
from autovision._transforms.view_transform import RandomGaussianBlur, RandomSolarize

__all__ = [
    "_GeometryViewTransform",
    "render_clean_global",
    "render_augmented_local",
    "sample_high_overlap_box",
]

# DINO-style local-crop scale: the PaKA student local is a small sub-region
# of the teacher global crop (mirroring DINO's local-crop scale range).
_LOCAL_AREA_MIN = 0.05
_LOCAL_AREA_MAX = 0.40


def _build_photometric_ops(
    color_jitter: ColorJitterArgs | None,
    random_gray_scale: float | None,
    gaussian_blur: GaussianBlurArgs | None,
    solarize: SolarizeArgs | None,
) -> nn.Module:
    """Photometric chain (jitter -> grayscale -> blur -> solarize), no geometry."""
    ops: list[nn.Module] = []
    if color_jitter is not None:
        jitter = v2.ColorJitter(
            brightness=color_jitter.strength * color_jitter.brightness,
            contrast=color_jitter.strength * color_jitter.contrast,
            saturation=color_jitter.strength * color_jitter.saturation,
            hue=color_jitter.strength * color_jitter.hue,
        )
        ops.append(v2.RandomApply([jitter], p=color_jitter.prob))
    if random_gray_scale is not None:
        ops.append(v2.RandomGrayscale(p=random_gray_scale))
    if gaussian_blur is not None:
        ops.append(RandomGaussianBlur(gaussian_blur))
    if solarize is not None:
        ops.append(RandomSolarize(solarize))
    if not ops:
        return nn.Identity()
    return v2.Compose(ops)


class _GeometryViewTransform:
    """One augmented view that records its crop geometry.

    Op order mirrors ``ViewTransform``: random resized crop -> flips ->
    photometric -> to float + normalize. The sampled crop box and flips are
    returned as the 8-element geometry tensor so other views can be aligned
    against this one.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        random_resize: tuple[float, float],
        random_flip_h: float,
        random_flip_v: float,
        color_jitter: ColorJitterArgs | None,
        random_gray_scale: float | None,
        gaussian_blur: GaussianBlurArgs | None,
        solarize: SolarizeArgs | None,
        normalize: NormalizeArgs,
    ) -> None:
        self.size = tuple(image_size)
        self.scale = tuple(random_resize)
        self.flip_h_prob = random_flip_h
        self.flip_v_prob = random_flip_v
        self.photometric = _build_photometric_ops(
            color_jitter, random_gray_scale, gaussian_blur, solarize
        )
        self.mean = list(normalize.mean)
        self.std = list(normalize.std)

    def __call__(self, image: Tensor) -> dict[str, Tensor]:
        height, width = int(image.shape[-2]), int(image.shape[-1])
        top, left, crop_h, crop_w = v2.RandomResizedCrop.get_params(
            image, scale=self.scale, ratio=(3.0 / 4.0, 4.0 / 3.0)
        )
        view = TF.resized_crop(
            image, top, left, crop_h, crop_w, list(self.size), antialias=True
        )
        hflip = random.random() < self.flip_h_prob
        vflip = random.random() < self.flip_v_prob
        if hflip:
            view = TF.hflip(view)
        if vflip:
            view = TF.vflip(view)
        view = self.photometric(view)
        view = TF.to_dtype(view, torch.float32, scale=True)
        view = TF.normalize(view, mean=self.mean, std=self.std)
        geometry = torch.tensor(
            [
                float(left),
                float(top),
                float(left + crop_w),
                float(top + crop_h),
                float(width),
                float(height),
                1.0 if hflip else 0.0,
                1.0 if vflip else 0.0,
            ],
            dtype=torch.float32,
        )
        return {"image": view, "geometry": geometry}


def _crop_resize_flips(
    image: Tensor,
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    hflip: bool,
    vflip: bool,
) -> Tensor:
    x0, y0, x1, y1 = box
    view = TF.crop(image, y0, x0, y1 - y0, x1 - x0)
    view = TF.resize(view, list(size), antialias=True)
    if hflip:
        view = TF.hflip(view)
    if vflip:
        view = TF.vflip(view)
    return view


def render_clean_global(
    image: Tensor,
    geometry: Tensor,
    size: tuple[int, int],
    normalize: NormalizeArgs,
) -> dict[str, Tensor]:
    """Render a global crop with NO photometric augmentation (clean teacher).

    Crops ``image`` to the box stored in ``geometry``, resizes to ``size``,
    re-applies the recorded flips, and normalizes. The returned view re-uses
    ``geometry`` verbatim: the PaKA ROI alignment can treat the clean and the
    augmented global as the same region.
    """
    x0 = int(round(float(geometry[0])))
    y0 = int(round(float(geometry[1])))
    x1 = int(round(float(geometry[2])))
    y1 = int(round(float(geometry[3])))
    hflip = float(geometry[6]) > 0.5
    vflip = float(geometry[7]) > 0.5
    view = _crop_resize_flips(image, (x0, y0, x1, y1), tuple(size), hflip, vflip)
    view = TF.normalize(
        TF.to_dtype(view, torch.float32, scale=True),
        mean=list(normalize.mean),
        std=list(normalize.std),
    )
    return {"image": view, "geometry": geometry.clone()}


def sample_high_overlap_box(
    parent_box: tuple[int, int, int, int],
    min_iou: float,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    """Sample a small local box fully inside ``parent_box``.

    The PaKA overlap constraint is a CONTAINMENT ratio (fraction of the local
    inside the global); sampling the local fully inside the parent satisfies
    it by construction. The local area is drawn in
    ``[_LOCAL_AREA_MIN, _LOCAL_AREA_MAX]`` of the parent area with a random
    aspect ratio and position.
    """
    del min_iou  # always satisfied by construction; kept for API clarity
    px0, py0, px1, py1 = parent_box
    pw = px1 - px0
    ph = py1 - py0
    if pw <= 1 or ph <= 1:
        return parent_box
    parent_area = float(pw * ph)
    target_area = rng.uniform(_LOCAL_AREA_MIN, _LOCAL_AREA_MAX) * parent_area
    aspect = rng.uniform(3.0 / 4.0, 4.0 / 3.0)
    w = round(math.sqrt(target_area * aspect))
    h = round(math.sqrt(target_area / aspect))
    w = min(max(1, w), pw)
    h = min(max(1, h), ph)
    x0 = px0 + rng.randint(0, pw - w)
    y0 = py0 + rng.randint(0, ph - h)
    return x0, y0, x0 + w, y0 + h


def render_augmented_local(
    image: Tensor,
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    hflip: bool,
    vflip: bool,
    color_jitter: ColorJitterArgs | None,
    random_gray_scale: float | None,
    gaussian_blur: GaussianBlurArgs | None,
    solarize: SolarizeArgs | None,
    normalize: NormalizeArgs,
) -> dict[str, Tensor]:
    """Render an augmented student local crop from a pre-sampled box.

    The crop box is decided by the caller (so geometry + containment are
    known); the render applies crop -> resize -> flips -> standard DINO
    photometric -> normalize, and records the box + flips as geometry.
    """
    image_h, image_w = int(image.shape[-2]), int(image.shape[-1])
    view = _crop_resize_flips(image, box, tuple(size), hflip, vflip)
    view = _build_photometric_ops(
        color_jitter, random_gray_scale, gaussian_blur, solarize
    )(view)
    view = TF.normalize(
        TF.to_dtype(view, torch.float32, scale=True),
        mean=list(normalize.mean),
        std=list(normalize.std),
    )
    geometry = torch.tensor(
        [
            float(box[0]),
            float(box[1]),
            float(box[2]),
            float(box[3]),
            float(image_w),
            float(image_h),
            1.0 if hflip else 0.0,
            1.0 if vflip else 0.0,
        ],
        dtype=torch.float32,
    )
    return {"image": view, "geometry": geometry}
