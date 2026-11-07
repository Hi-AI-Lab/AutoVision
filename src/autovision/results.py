# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Typed container for one image's inference result.

Task models deliberately return plain dictionaries (``postprocess`` is part of
the export contract). Consumers used to re-implement the same key handling,
length checks and JSON conversion in every module. :class:`InferenceResult`
is a thin boundary object for those consumers: it validates once, supports
selection/filtering, and owns the mask serialization policy (RLE by default).

The container holds ``torch.Tensor`` or ``numpy.ndarray`` values and does not
copy or move them. It is intentionally not used inside training or export
code, where the native tensor/dict contract stays authoritative.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
import torch

from autovision.errors import AutoVisionError, ResultValidationError
from autovision.keys import (
    BBOXES,
    DEPTH,
    INPUT_H,
    INPUT_W,
    KEYPOINTS,
    LABELS,
    MASK,
    MASKS,
    POLYGONS,
    SCORES,
    SEGMENT_IDS,
)

__all__ = ["InferenceResult", "mask_to_bool"]

_KNOWN_KEYS = {
    BBOXES,
    SCORES,
    LABELS,
    MASKS,
    KEYPOINTS,
    POLYGONS,
    MASK,
    DEPTH,
    SEGMENT_IDS,
    INPUT_W,
    INPUT_H,
}

#: Cardinality above which dense arrays are summarized instead of serialized.
DEFAULT_MAX_ELEMENTS = 10_000

ArrayLike: TypeAlias = torch.Tensor | np.ndarray


def _shape(value: ArrayLike) -> tuple[int, ...]:
    return tuple(int(size) for size in value.shape)


def _array_length(value: ArrayLike) -> int:
    return int(value.shape[0]) if value.ndim else 1


def _to_tensor_if_sequence(value: Any, name: str) -> Any:
    """Convert list/tuple inputs to tensors; leave tensors and arrays alone."""
    if not isinstance(value, (list, tuple)):
        return value
    if not value:
        empty_dtypes: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
            BBOXES: ((0, 4), torch.float32),
            POLYGONS: ((0, 4, 2), torch.float32),
            KEYPOINTS: ((0, 0, 3), torch.float32),
            MASKS: ((0, 0, 0), torch.bool),
            SCORES: ((0,), torch.float32),
            LABELS: ((0,), torch.long),
            SEGMENT_IDS: ((0,), torch.long),
        }
        if name in empty_dtypes:
            shape, dtype = empty_dtypes[name]
            return torch.empty(shape, dtype=dtype)
    return torch.as_tensor(value)


def _normalize_index(index: Any, value: ArrayLike) -> Any:
    """Make a list-like selection index usable by both tensor backends."""
    if isinstance(index, (list, tuple)):
        array = np.asarray(index)
        if np.issubdtype(array.dtype, np.bool_):
            if torch.is_tensor(value):
                return torch.as_tensor(
                    array, dtype=torch.bool, device=value.device
                )
            return array
        if torch.is_tensor(value):
            return torch.as_tensor(
                array, dtype=torch.long, device=value.device
            )
        return array.astype(np.int64)
    if isinstance(index, np.ndarray):
        if np.issubdtype(index.dtype, np.bool_):
            if torch.is_tensor(value):
                return torch.as_tensor(index, device=value.device)
            return index
        if torch.is_tensor(value):
            return torch.as_tensor(
                index.astype(np.int64), device=value.device
            )
        return index.astype(np.int64)
    if torch.is_tensor(index) and isinstance(value, np.ndarray):
        return index.detach().cpu().numpy()
    if torch.is_tensor(index) and torch.is_tensor(value):
        # Mixed-device results are normal (e.g. CUDA scores + CPU masks).
        return index.to(device=value.device)
    return index


def _select(value: ArrayLike | None, index: Any) -> ArrayLike | None:
    if value is None:
        return None
    return value[_normalize_index(index, value)]


def _as_numpy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Mapping):
        return {key: _as_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_numpy(item) for item in value]
    return value


def _summarize(value: ArrayLike) -> dict[str, Any]:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "summarized": True,
        "dtype": str(array.dtype),
    }
    if array.size == 0:
        return summary
    try:
        finite = array[np.isfinite(array)]
    except TypeError:  # non-numeric payloads (e.g. object/bool)
        finite = array
    if finite.size == 0:
        return summary
    try:
        summary["min"] = float(np.min(finite))
        summary["max"] = float(np.max(finite))
        summary["mean"] = float(np.mean(finite))
    except (TypeError, ValueError):
        # Non-numeric payloads (strings, datetimes, mixed objects) have no
        # numeric statistics: ``np.min`` raises TypeError/UFuncTypeError and
        # ``float(<str>)`` raises ValueError. A shape/dtype/length summary
        # still bounds the payload instead of crashing serialization (H3-1d).
        summary["length"] = int(array.size)
    return summary


def _sanitize_json(value: Any) -> Any:
    """Replace non-finite floats with ``None`` (strict JSON has no NaN/Inf)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    return value


def _jsonable(value: Any, *, max_elements: int) -> Any:
    """Convert tensors/arrays to JSON-native values with a size guard."""
    if torch.is_tensor(value):
        if value.numel() > max_elements:
            return _summarize(value)
        return _sanitize_json(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        if value.size > max_elements:
            return _summarize(value)
        return _sanitize_json(value.tolist())
    if isinstance(value, np.generic):
        return _sanitize_json(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, max_elements=max_elements)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, max_elements=max_elements) for item in value]
    return _sanitize_json(value)


def mask_to_bool(value: ArrayLike) -> np.ndarray:
    """Normalize bool/int/float masks to a boolean NumPy array.

    Integer masks use ``> 0`` so panoptic-style ``-1`` (unassigned) counts
    as background, mirroring the float path's ``> 0.5`` (H3-1g).
    """
    array = (
        value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    )
    if np.issubdtype(array.dtype, np.bool_):
        return array
    if np.issubdtype(array.dtype, np.integer):
        return array > 0
    return array > 0.5


def _encode_binary_mask(mask: ArrayLike) -> dict[str, Any]:
    """Encode one 2-D mask as COCO RLE (pycocotools is a core dependency).

    Bool masks are used as-is; integer masks use ``> 0`` and float masks use
    ``> 0.5`` as the binary threshold.
    """
    from pycocotools import mask as mask_utils

    array = mask_to_bool(mask)
    if array.ndim != 2:
        raise ResultValidationError(f"RLE expects a 2-D mask, got shape {array.shape}.")
    encoded = mask_utils.encode(np.asfortranarray(array.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(size) for size in encoded["size"]], "counts": counts}


def _encode_label_map(label_map: ArrayLike) -> dict[str, Any]:
    """Run-length encode a 2-D integer label map (lossless, JSON-native)."""
    array = (
        label_map.detach().cpu().numpy()
        if isinstance(label_map, torch.Tensor)
        else np.asarray(label_map)
    )
    if array.ndim != 2:
        raise ResultValidationError(f"label-map RLE expects a 2-D map, got shape {array.shape}.")
    flat = array.reshape(-1)
    if flat.size == 0:
        return {"size": list(array.shape), "counts": [], "values": []}
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], changes))
    lengths = np.diff(np.concatenate((starts, [flat.size]))).tolist()
    return {
        "size": [int(size) for size in array.shape],
        "counts": [int(length) for length in lengths],
        "values": [int(value) for value in flat[starts]],
    }


@dataclass
class InferenceResult:
    """One image's postprocessed output, independent of the task family.

    Attributes:
        bboxes: ``(N, 4)`` xyxy boxes in original-frame pixels.
        scores: ``(N,)`` confidences (per segment for panoptic).
        labels: ``(N,)`` class ids.
        masks: ``(N, H, W)`` instance masks or a panoptic ``(H, W, 2)``
            ``(class_id, segment_id)`` map.
        keypoints: ``(N, K, 3)`` pose keypoints with ``[x, y, visibility]``.
        polygons: ``(N, 4, 2)`` oriented-box corners.
        mask: ``(H, W)`` semantic-segmentation label map.
        depth: ``(H, W)`` depth map.
        segment_ids: ``(S,)`` panoptic segment ids.
        input_w/input_h: Network input size recorded by segmentation decoders.
        metadata: Per-image metadata (source, frame index, timestamps, ...).
        extra: Task-specific fields not covered by the typed attributes.
        task: Optional task name; inferred from the populated fields when
            omitted.
    """

    bboxes: ArrayLike | None = None
    scores: ArrayLike | None = None
    labels: ArrayLike | None = None
    masks: ArrayLike | None = None
    keypoints: ArrayLike | None = None
    polygons: ArrayLike | None = None
    mask: ArrayLike | None = None
    depth: ArrayLike | None = None
    segment_ids: ArrayLike | None = None
    input_w: int | None = None
    input_h: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    task: str | None = None

    def __post_init__(self) -> None:
        for name in (
            BBOXES,
            SCORES,
            LABELS,
            MASKS,
            KEYPOINTS,
            POLYGONS,
            MASK,
            DEPTH,
            SEGMENT_IDS,
        ):
            setattr(self, name, _to_tensor_if_sequence(getattr(self, name), name))

        collisions = set(self.extra) & _KNOWN_KEYS
        if collisions:
            raise ResultValidationError(
                "InferenceResult.extra must not duplicate typed fields: "
                f"{sorted(collisions)}."
            )
        # Unique values of a float semantic map entering the RLE path; cached
        # so repeated serialization does not rescan a large map (H1-11).
        self._float_mask_unique: np.ndarray | None = None
        self._validate()
        self._warn_ambiguous_panoptic_shape()

    # ── construction ───────────────────────────────────────────────────
    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
        task: str | None = None,
    ) -> "InferenceResult":
        """Build a container from a ``postprocess`` record.

        ``boxes`` is accepted as an alias for ``bboxes``; every untyped key is
        preserved in :attr:`extra` so ``to_record()`` round-trips task-specific
        outputs (``segment_ids``, decoder input sizes, ...).
        """
        values = dict(record)
        if BBOXES not in values and "boxes" in values:
            values[BBOXES] = values.pop("boxes")

        record_meta = values.pop("metadata", None)
        if record_meta is None:
            record_meta = values.pop("meta", None)
        merged_meta: dict[str, Any] = {}
        if isinstance(record_meta, Mapping):
            merged_meta.update(record_meta)
        if metadata:
            merged_meta.update(metadata)

        typed = {key: values.pop(key) for key in list(values) if key in _KNOWN_KEYS}
        return cls(metadata=merged_meta, extra=values, task=task, **typed)

    # ── queries ────────────────────────────────────────────────────────
    @property
    def boxes(self) -> ArrayLike | None:
        """Alias for :attr:`bboxes` (common downstream naming)."""
        return self.bboxes

    @property
    def inferred_task(self) -> str:
        """Task name inferred from populated fields (explicit ``task`` wins)."""
        if self.task:
            return self.task
        if self.polygons is not None:
            return "oriented_object_detection"
        if self.keypoints is not None:
            return "pose_estimation"
        if self.depth is not None:
            return "depth_estimation"
        if self.segment_ids is not None or self._looks_like_panoptic_map():
            return "panoptic_segmentation"
        if self.mask is not None:
            return "semantic_segmentation"
        if self.masks is not None:
            # EOMT instance segmentation can return masks + labels without
            # boxes; it is not image classification.
            return "instance_segmentation"
        if self.bboxes is not None:
            return "object_detection"
        if self.labels is not None:
            return "image_classification"
        return "unknown"

    @property
    def is_empty(self) -> bool:
        """True when no per-instance row or image-level payload exists."""
        return len(self) == 0

    @property
    def is_panoptic(self) -> bool:
        """True for a panoptic ``(H, W, 2)`` label map.

        ``segment_ids`` disambiguate authoritatively; without them the
        shape-only heuristic in :meth:`_looks_like_panoptic_map` decides.
        """
        return self._looks_like_panoptic_map()

    @property
    def has_instance_masks(self) -> bool:
        """True when ``masks`` is an ``(N, H, W)`` per-instance stack."""
        return self._masks_are_per_instance()

    def _primary(self) -> ArrayLike | None:
        for value in (self.bboxes, self.polygons, self.keypoints):
            if value is not None:
                return value
        return None

    def _looks_like_panoptic_map(self) -> bool:
        """Shape-level panoptic decision shared by every mask consumer (H3-1b).

        Panoptic decoders emit ``(H, W, 2)`` ``(class_id, segment_id)`` maps;
        instance stacks are ``(N, H, W)``. ``segment_ids`` is the only
        reliable discriminator, but producers that omit it made a ``(H, W,
        2)`` map read as N per-instance rows with degenerate ``(W, 2)``
        masks, so the heuristic classifies every box-less ``(H, W, 2)``
        -shaped ``masks`` as panoptic — ``segment_ids`` present or not.

        Ambiguity boundary: a genuine ``(N, H, W)`` instance stack over a
        2-pixel-wide frame also has ``shape[-1] == 2`` and is misclassified
        as panoptic. The trade-off is deliberate: panoptic maps never carry
        boxes in this contract, so an explicit ``bboxes`` row contract keeps
        such stacks per-instance, while a box-less ``W == 2`` stack is far
        rarer than the missing-``segment_ids`` map this fixes. Constructors
        hitting the boundary get a warning pointing at ``segment_ids`` (see
        :meth:`_warn_ambiguous_panoptic_shape`).
        """
        if self.masks is None or self.masks.ndim != 3 or int(self.masks.shape[-1]) != 2:
            return False
        return self.bboxes is None

    def _masks_are_per_instance(self) -> bool:
        if self.masks is None or self.masks.ndim != 3 or self._looks_like_panoptic_map():
            return False
        primary = self._primary()
        if primary is not None:
            return int(self.masks.shape[0]) == _array_length(primary)
        if self.labels is not None:
            return int(self.masks.shape[0]) == _array_length(self.labels)
        return True

    def __len__(self) -> int:
        if self.bboxes is not None:
            return _array_length(self.bboxes)
        if self.polygons is not None:
            return _array_length(self.polygons)
        if self.keypoints is not None:
            return _array_length(self.keypoints)
        if self.segment_ids is not None:
            return _array_length(self.segment_ids)
        if self.labels is not None:
            return _array_length(self.labels)
        if self.masks is not None:
            return 1 if self._looks_like_panoptic_map() else int(self.masks.shape[0])
        if self.mask is not None or self.depth is not None:
            return 1
        return 0

    # ── selection ──────────────────────────────────────────────────────
    def select(self, index: Any) -> "InferenceResult":
        """Return the selected per-instance rows as a fresh container.

        Typed per-instance arrays are sliced; image-level ``mask``/``depth``
        and task-specific ``extra`` fields are copied unchanged.
        """
        if isinstance(index, (int, np.integer)):
            if len(self) == 0:
                # An empty container would surface a bare ``IndexError`` from
                # the array backends; fail with a framework error instead.
                raise AutoVisionError(
                    f"select({int(index)}) needs a non-empty result: this "
                    "container holds no per-instance rows to index."
                )
            index = [int(index)]
        if self._looks_like_panoptic_map() or (
            self.segment_ids is not None and self.bboxes is None and self.keypoints is None
        ):
            raise ResultValidationError(
                "select() is not defined for image-level or segment-based "
                "results; slice the per-pixel map directly."
            )
        masks = self.masks if self._masks_are_per_instance() else None
        selected = InferenceResult(
            bboxes=_select(self.bboxes, index),
            scores=_select(self.scores, index),
            labels=_select(self.labels, index),
            masks=_select(masks, index),
            keypoints=_select(self.keypoints, index),
            polygons=_select(self.polygons, index),
            mask=self.mask,
            depth=self.depth,
            segment_ids=self.segment_ids,
            input_w=self.input_w,
            input_h=self.input_h,
            metadata=dict(self.metadata),
            extra=dict(self.extra),
            task=self.task,
        )
        if self.masks is not None and masks is None:
            # Image-level masks (panoptic maps) are copied, not sliced.
            selected.masks = self.masks
        return selected

    def filter(self, mask: Any) -> "InferenceResult":
        """Select rows with a boolean mask (tensor, array, or bool list)."""
        return self.select(mask)

    def filter_scores(self, min_score: float) -> "InferenceResult":
        """Drop rows whose score is below ``min_score`` (no-op without scores).

        NaN scores fail every threshold comparison and are dropped; a
        warning reports the discarded count so row loss stays visible
        instead of silently shrinking the result (H3-1j).
        """
        if self.scores is None or self.is_panoptic:
            return self
        if self.mask is not None or self.depth is not None:
            # Image-level payloads have no per-instance rows to filter.
            return self
        scores = (
            self.scores.detach().cpu().numpy()
            if isinstance(self.scores, torch.Tensor)
            else np.asarray(self.scores)
        )
        nan_rows = int(np.isnan(scores).sum())
        if nan_rows:
            warnings.warn(
                f"filter_scores dropped {nan_rows} row(s) with NaN scores; "
                "NaN fails every threshold comparison and cannot be kept.",
                UserWarning,
                stacklevel=2,  # past this method: the filter_scores caller
            )
        return self.select(self.scores >= float(min_score))

    # ── conversion ─────────────────────────────────────────────────────
    def to_numpy(self) -> "InferenceResult":
        """Return a container with NumPy array fields.

        Tensor inputs are detached/copied to CPU; NumPy inputs are reused
        without an extra copy.
        """
        return InferenceResult(
            bboxes=_as_numpy(self.bboxes),
            scores=_as_numpy(self.scores),
            labels=_as_numpy(self.labels),
            masks=_as_numpy(self.masks),
            keypoints=_as_numpy(self.keypoints),
            polygons=_as_numpy(self.polygons),
            mask=_as_numpy(self.mask),
            depth=_as_numpy(self.depth),
            segment_ids=_as_numpy(self.segment_ids),
            input_w=self.input_w,
            input_h=self.input_h,
            metadata=_as_numpy(self.metadata),
            extra=_as_numpy(self.extra),
            task=self.task,
        )

    def to_record(self) -> dict[str, Any]:
        """Return the original-style record (arrays untouched, ``None`` omitted)."""
        record: dict[str, Any] = {}
        for key in (
            BBOXES,
            SCORES,
            LABELS,
            MASKS,
            KEYPOINTS,
            POLYGONS,
            MASK,
            DEPTH,
            SEGMENT_IDS,
        ):
            value = getattr(self, key)
            if value is not None:
                record[key] = value
        if self.input_w is not None:
            record[INPUT_W] = self.input_w
        if self.input_h is not None:
            record[INPUT_H] = self.input_h
        record.update(self.extra)
        if self.metadata:
            record["metadata"] = dict(self.metadata)
        return record

    def to_json(
        self,
        *,
        masks: str = "rle",
        max_elements: int = DEFAULT_MAX_ELEMENTS,
    ) -> dict[str, Any]:
        """JSON-native record with a bounded payload.

        Args:
            masks: ``"rle"`` (default) encodes instance/semantic masks as COCO
                RLE (instance: a list of ``{"size","counts"}``; semantic:
                the same dict for a binary map or ``{"size","counts","values"}``
                for an integer label map; panoptic: ``{"size","class_ids",
                "segment_ids"}`` with a run map per channel). ``"dense"`` emits
                nested lists; ``"none"`` omits mask payloads.
            max_elements: Applies to image-level payloads (depth, panoptic
                maps) and task-specific ``extra`` fields. Per-instance rows
                (boxes/scores/labels/keypoints/polygons/segment ids) are
                always emitted in full so detection counts are never hidden
                behind a summary.
        """
        if masks not in {"rle", "dense", "none"}:
            raise ResultValidationError(
                f"masks must be 'rle', 'dense', or 'none', got {masks!r}."
            )
        if max_elements < 1:
            raise ResultValidationError("max_elements must be >= 1.")
        uncapped = 10**18

        payload: dict[str, Any] = {}
        for key in (
            BBOXES,
            SCORES,
            LABELS,
            KEYPOINTS,
            POLYGONS,
            SEGMENT_IDS,
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = _jsonable(value, max_elements=uncapped)
        if self.depth is not None:
            payload[DEPTH] = _jsonable(self.depth, max_elements=max_elements)

        if self.masks is not None and masks != "none":
            payload[MASKS] = self._serialize_masks(
                mode=masks, max_elements=max_elements
            )
        if self.mask is not None and masks != "none":
            mask_array = (
                self.mask.detach().cpu().numpy()
                if isinstance(self.mask, torch.Tensor)
                else np.asarray(self.mask)
            )
            if masks == "rle" and np.issubdtype(mask_array.dtype, np.bool_):
                payload[MASK] = _encode_binary_mask(self.mask)
            elif masks == "rle" and not np.issubdtype(mask_array.dtype, np.integer):
                self._warn_float_label_map_rle(mask_array)
                payload[MASK] = _encode_binary_mask(self.mask)
            elif masks == "rle":
                payload[MASK] = _encode_label_map(self.mask)
            else:
                payload[MASK] = _jsonable(self.mask, max_elements=uncapped)

        if self.input_w is not None:
            payload[INPUT_W] = int(self.input_w)
        if self.input_h is not None:
            payload[INPUT_H] = int(self.input_h)
        for key, value in self.extra.items():
            payload[key] = _jsonable(value, max_elements=max_elements)
        if self.metadata:
            payload["metadata"] = _jsonable(self.metadata, max_elements=max_elements)
        if self.task is not None:
            payload["task"] = self.task
        return payload

    def _warn_float_label_map_rle(self, mask_array: np.ndarray) -> None:
        """Deprecation gate for float semantic maps on the RLE path (H1-11).

        Through 0.6.x the behavior stays the silent ``> 0.5`` binarization;
        0.7.0 will reject these maps. One warning is emitted per serialization
        event; the ``np.unique`` scan backing the message is cached on the
        instance so repeated ``to_json`` calls skip the rescan.
        """
        if self._float_mask_unique is None:
            self._float_mask_unique = np.unique(mask_array)
        unique = self._float_mask_unique
        shown = ", ".join(str(float(value)) for value in unique[:8]) + (
            ", ..." if unique.size > 8 else ""
        )
        warnings.warn(
            "Semantic float label maps are binarized (> 0.5) by masks='rle' "
            f"(unique values: {shown}); deprecated in 0.6.1, float label "
            "maps will be rejected in 0.7.0; convert to integer dtype or use "
            "masks='dense'/'none'.",
            FutureWarning,
            stacklevel=3,  # past this helper and to_json: the caller
        )

    def _check_dense_mask_budget(self, max_elements: int) -> None:
        """Hard resource gate for explicit ``masks="dense"`` mode (H3-1a).

        Dense mode opts out of summarization on purpose, so the gate only
        rejects payloads far beyond the caller's budget (``max_elements``
        times 100) — a runaway mask shape must fail loud instead of building
        an unbounded JSON blob.
        """
        assert self.masks is not None
        # isinstance narrows for mypy: torch.is_tensor is not a TypeGuard here.
        if isinstance(self.masks, np.ndarray):
            numel = int(self.masks.size)
        else:
            numel = int(self.masks.numel())
        if numel > max_elements * 100:
            raise AutoVisionError(
                f"dense mask payload has {numel} elements, above the "
                f"max_elements*100 safety limit ({max_elements * 100}); use "
                "masks='rle' or shrink the input."
            )

    def _serialize_masks(self, *, mode: str, max_elements: int) -> Any:
        assert self.masks is not None
        if self._masks_are_per_instance():
            if mode == "rle":
                return [
                    _encode_binary_mask(self.masks[index])
                    for index in range(int(self.masks.shape[0]))
                ]
            # Explicit dense mode: never hide rows behind a summary, but the
            # opt-in still passes the H3-1a resource gate.
            self._check_dense_mask_budget(max_elements)
            return _jsonable(self.masks, max_elements=10**18)
        if self.is_panoptic and mode == "rle":
            return {
                "size": list(self.masks.shape[:2]),
                "class_ids": _encode_label_map(self.masks[..., 0]),
                "segment_ids": _encode_label_map(self.masks[..., 1]),
            }
        if mode == "dense":
            self._check_dense_mask_budget(max_elements)
            return _jsonable(self.masks, max_elements=10**18)
        # Unknown image-level mask payloads keep the dense/capped form.
        return _jsonable(self.masks, max_elements=max_elements)

    # ── validation ─────────────────────────────────────────────────────
    def _validate(self) -> None:
        if self.bboxes is not None:
            self._validate_shape(BBOXES, self.bboxes, 2, 4)
        for name in (SCORES, LABELS, SEGMENT_IDS):
            value = getattr(self, name)
            if value is not None:
                self._validate_ndim(name, value, 1)
        if self.keypoints is not None:
            if self.keypoints.ndim not in (2, 3):
                raise ResultValidationError(
                    "keypoints must be (N, 3K) or (N, K, 3), got shape "
                    f"{_shape(self.keypoints)}."
                )
        if self.polygons is not None:
            if self.polygons.ndim == 2 and self.polygons.shape[-1] == 8:
                reshaped = torch.as_tensor(self.polygons).reshape(-1, 4, 2)
                self.polygons = (
                    reshaped
                    if torch.is_tensor(self.polygons)
                    else reshaped.numpy()
                )
            if self.polygons.ndim != 3 or int(self.polygons.shape[-1]) != 2:
                raise ResultValidationError(
                    "polygons must be (N, 4, 2) or (N, 8), got shape "
                    f"{_shape(self.polygons)}."
                )
        if self.masks is not None and self.masks.ndim != 3:
            raise ResultValidationError(
                "masks must be (N, H, W) or (H, W, 2), got shape "
                f"{_shape(self.masks)}."
            )
        for name in (MASK, DEPTH):
            value = getattr(self, name)
            if value is not None:
                self._validate_ndim(name, value, 2)

        primary = self._primary()
        primary_len = _array_length(primary) if primary is not None else None
        if primary_len is not None:
            for name in (SCORES, LABELS, MASKS):
                value = getattr(self, name)
                if value is None:
                    continue
                length = _array_length(value)
                if length != primary_len:
                    raise ResultValidationError(
                        f"{name} has {length} rows but {BBOXES if self.bboxes is not None else 'the primary field'} "
                        f"has {primary_len}."
                    )
        if self.segment_ids is not None and self.scores is not None:
            if _array_length(self.segment_ids) != _array_length(self.scores):
                raise ResultValidationError(
                    f"segment_ids has {_array_length(self.segment_ids)} entries but "
                    f"scores has {_array_length(self.scores)}."
                )
        if (
            primary is None
            and self.segment_ids is None
            and self.labels is not None
            and self.scores is not None
            and _array_length(self.labels) != _array_length(self.scores)
        ):
            raise ResultValidationError(
                f"labels has {_array_length(self.labels)} rows but scores has "
                f"{_array_length(self.scores)}."
            )
        if (
            self.masks is not None
            and self.masks.ndim == 3
            and self._primary() is None
            and not self._looks_like_panoptic_map()
            and self.labels is not None
            and int(self.masks.shape[0]) != _array_length(self.labels)
        ):
            raise ResultValidationError(
                f"masks has {int(self.masks.shape[0])} rows but labels has "
                f"{_array_length(self.labels)}."
            )
        if self.input_w is not None and int(self.input_w) <= 0:
            raise ResultValidationError("input_w must be positive.")
        if self.input_h is not None and int(self.input_h) <= 0:
            raise ResultValidationError("input_h must be positive.")

    def _warn_ambiguous_panoptic_shape(self) -> None:
        """Constructor-time warning for ``(H, W, 2)`` masks without ids (H3-1b).

        The shape-only heuristic in :meth:`_looks_like_panoptic_map` decides
        panoptic for box-less ``(H, W, 2)`` masks even when ``segment_ids``
        is missing; this names that decision once per construction so
        producers restore an unambiguous contract instead of leaning on the
        heuristic.
        """
        if (
            self.masks is None
            or self.segment_ids is not None
            or self.bboxes is not None
            or self.masks.ndim != 3
            or int(self.masks.shape[-1]) != 2
        ):
            return
        warnings.warn(
            "masks has (H, W, 2) shape but segment_ids is missing; it is "
            "treated as a panoptic (class_id, segment_id) map, which would "
            "misread a genuine (N, H, W) instance stack on a 2-pixel-wide "
            "frame. Pass segment_ids explicitly to remove the ambiguity.",
            UserWarning,
            stacklevel=3,  # past this helper and __post_init__: the caller
        )

    @staticmethod
    def _validate_ndim(name: str, value: ArrayLike, ndim: int) -> None:
        if value.ndim != ndim:
            raise ResultValidationError(
                f"{name} must be a {ndim}-D array, got shape {_shape(value)}."
            )

    @staticmethod
    def _validate_shape(
        name: str, value: ArrayLike, ndim: int, last: int
    ) -> None:
        if value.ndim != ndim or int(value.shape[-1]) != last:
            raise ResultValidationError(
                f"{name} must have shape (N, {last}), got {_shape(value)}."
            )
