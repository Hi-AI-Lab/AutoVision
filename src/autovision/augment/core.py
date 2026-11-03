# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Geometric-augmentation protocol: one transform, four targets (design D24).

The architecture precondition for porting composite augmentations
(mosaic/mixup/zoom-out — see ``docs/known_limits.md``): a composite
transform samples its random parameters ONCE and applies the same
geometry to the image, the instances (boxes/segments/keypoints), the
semantic mask, and the depth map, via four target hooks. Per-task
hand-rolled geometry is what this protocol exists to prevent — the
task faces only override the hooks they understand.

Value conventions the hooks must honor (documented, enforced by tests of
each concrete transform rather than by the protocol itself):

- semantic masks: nearest-neighbor resampling, fill value ``255`` (ignore);
- depth: nearest-neighbor resampling, fill value ``0`` (invalid) — bilinear
  interpolation or nonzero fill would fabricate plausible-looking depths
  at object boundaries where the ground truth is void;
- keypoints falling outside the canvas: visibility forced to 0.

This module defines the protocol only; it ships no concrete transforms.
Method-style transforms (``MethodTransformArgs`` classes) remain the
public augmentation surface — the protocol is the implementation layer
for composite transforms that must move several targets in lockstep.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

__all__ = ["GeoParams", "GeoTransform", "SEMANTIC_IGNORE_VALUE"]

#: Fill value for semantic-mask canvases (the ignore label convention).
SEMANTIC_IGNORE_VALUE = 255


@dataclass(frozen=True)
class GeoParams:
    """Random parameters shared by all four target hooks of one sample.

    Concrete transforms subclass this with their own fields (angles,
    scales, flip decisions, canvas geometry) and keep it frozen: the same
    instance must drive every hook of one sample, so hooks read it
    read-only.
    """

    random_draws: tuple[float, ...] = ()


class GeoTransform(ABC):
    """Template-method base: sample once, apply to four targets.

    Subclasses implement :meth:`sample_params` plus the hooks for the
    targets they support. :meth:`__call__` is fixed orchestration and
    passes the same frozen params object to every hook — that shared
    identity is the whole point of the protocol, so it is deliberately
    not overridable.
    """

    def __call__(
        self,
        *,
        image: np.ndarray,
        params: GeoParams | None = None,
        instances: Any | None = None,
        semantic_mask: np.ndarray | None = None,
        depth: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Transform one sample's targets with shared geometry.

        Args:
            image: ``(..., H, W)`` or ``(H, W, C)`` array (hook's contract).
            params: Pre-sampled parameters (e.g. a mosaic donor already
                drew them); sampled fresh when ``None``.
            instances: Task-owned instance geometry (boxes/segments/
                keypoints container), transformed if provided.
            semantic_mask: ``(H, W)`` label map, transformed if provided.
            depth: ``(H, W)`` or ``(H, W, 1)`` depth map, transformed if
                provided.

        Returns:
            ``{"image", "instances", "semantic_mask", "depth"}`` with the
            provided targets replaced by their transformed versions
            (absent targets stay ``None``).
        """
        if params is None:
            params = self.sample_params(image)
        out: dict[str, Any] = {"image": None, "instances": None, "semantic_mask": None, "depth": None}
        out["image"] = self.apply_image(image, params)
        if instances is not None:
            out["instances"] = self.apply_instances(instances, params)
        if semantic_mask is not None:
            out["semantic_mask"] = self.apply_semantic_mask(semantic_mask, params)
        if depth is not None:
            out["depth"] = self.apply_depth(depth, params)
        return out

    @abstractmethod
    def sample_params(self, image: np.ndarray) -> GeoParams:
        """Draw the random parameters for one sample (called once)."""

    @abstractmethod
    def apply_image(self, image: np.ndarray, params: GeoParams) -> np.ndarray:
        """Transform the image with ``params``."""

    def apply_instances(self, instances: Any, params: GeoParams) -> Any:
        """Transform instance geometry with ``params`` (default: no-op)."""
        return instances

    def apply_semantic_mask(
        self, semantic_mask: np.ndarray, params: GeoParams
    ) -> np.ndarray:
        """Transform a semantic label map; nearest + 255 fill by default.

        The default implementation raises so a transform that moves pixels
        cannot silently leave the mask behind — declare the hook explicitly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} samples geometry but does not declare "
            "apply_semantic_mask; give it a real hook or do not pass a mask."
        )

    def apply_depth(self, depth: np.ndarray, params: GeoParams) -> np.ndarray:
        """Transform a depth map; nearest + 0 fill by default (see module)."""
        raise NotImplementedError(
            f"{type(self).__name__} samples geometry but does not declare "
            "apply_depth; give it a real hook or do not pass a depth map."
        )


def nearest_resize_to(
    source: np.ndarray, size_hw: tuple[int, int]
) -> np.ndarray:
    """Nearest-neighbor resize of a (H, W) or (H, W, C) map (shared helper).

    Torch-v2 transforms handle images; label/depth maps need EXACT
    nearest-neighbor (no antialiasing, no interpolation of class ids or
    invalid pixels), so this helper exists for the mask/depth hooks. The
    internal float round-trip is exact only for values up to 2**24 — fine
    for uint8 class ids and uint16 depth (millimeters), the two formats
    this protocol serves.
    """
    height, width = size_hw
    import torch.nn.functional as F

    tensor = torch.from_numpy(np.ascontiguousarray(source))
    needs_channel = tensor.ndim == 2
    if needs_channel:
        tensor = tensor.unsqueeze(0).unsqueeze(0).float()
    else:  # (H, W, C)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()
    out = F.interpolate(
        tensor, size=(height, width), mode="nearest-exact"
    )
    out = out.squeeze(0)
    if needs_channel:
        out = out.squeeze(0)
    else:
        out = out.permute(1, 2, 0)
    return out.numpy().astype(source.dtype)
