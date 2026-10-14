# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Slicing-aided inference (SAHI) for large images.

``tile_image`` tiles an image with overlap (always appending the final
aligned tile); ``combine_object_detection_tiles`` merges tile predictions
with a global (full-image) prediction: per-label NMS inside tiles, offset
boxes to original coordinates, then global-vs-local same-label IoU
suppression (baseline parameters: nms 0.3, global/local 0.1).
"""

from __future__ import annotations

import torch
from torchvision.ops import box_iou, nms

__all__ = ["combine_object_detection_tiles", "tile_image"]


def tile_image(
    image: torch.Tensor,
    overlap: float,
    tile_size: int,
    padding_mode: str = "resize",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tile a (C, H, W) image into (N, C, tile, tile).

    Returns ``(tiles, tiles_coordinates)`` with coordinates (x_start,
    y_start). Images smaller than the tile are returned as a single tile.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if padding_mode not in ("resize", "pad"):
        raise ValueError(f"padding_mode must be 'resize' or 'pad', got '{padding_mode}'")
    _, height, width = image.shape
    if height <= tile_size and width <= tile_size:
        if height < tile_size or width < tile_size:
            if padding_mode == "pad":
                image = torch.nn.functional.pad(
                    image, (0, tile_size - width, 0, tile_size - height)
                )
            else:
                image = torch.nn.functional.interpolate(
                    image[None].float(), size=(tile_size, tile_size),
                    mode="bilinear", align_corners=False,
                )[0].to(image.dtype)
        return image[None], torch.zeros(1, 2, dtype=torch.int64)

    step = max(1, int((1.0 - overlap) * tile_size))
    max_y = max(height - tile_size, 0)
    max_x = max(width - tile_size, 0)
    y_starts = list(range(0, max_y + 1, step)) or [0]
    x_starts = list(range(0, max_x + 1, step)) or [0]
    if y_starts[-1] != max_y:
        y_starts.append(max_y)
    if x_starts[-1] != max_x:
        x_starts.append(max_x)

    tiles, coordinates = [], []
    for y in y_starts:
        for x in x_starts:
            tile = image[:, y : y + tile_size, x : x + tile_size]
            _, tile_h, tile_w = tile.shape
            if tile_h < tile_size or tile_w < tile_size:
                if padding_mode == "resize":
                    tile = torch.nn.functional.interpolate(
                        tile[None].float(), size=(tile_size, tile_size),
                        mode="bilinear", align_corners=False,
                    )[0].to(image.dtype)
                else:
                    tile = torch.nn.functional.pad(
                        tile, (0, tile_size - tile_w, 0, tile_size - tile_h)
                    )
            tiles.append(tile)
            coordinates.append((x, y))
    return torch.stack(tiles), torch.tensor(coordinates, dtype=torch.int64)


def combine_object_detection_tiles(
    global_output: dict[str, torch.Tensor],
    tile_outputs: list[dict[str, torch.Tensor]],
    tile_coordinates: torch.Tensor,
    original_size: tuple[int, int],
    nms_iou_threshold: float = 0.3,
    global_local_iou_threshold: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Merge a full-image prediction with per-tile predictions.

    ``tile_outputs[i]``: {"bboxes": (n_i, 4) xyxy **in tile coordinates**,
    "scores", "labels"}; boxes are offset to original coordinates by tile i.
    """
    width, height = original_size
    kept_boxes, kept_scores, kept_labels = [], [], []

    global_boxes = global_output["bboxes"]
    global_scores = global_output["scores"]
    global_labels = global_output["labels"]
    for index in range(global_boxes.shape[0]):
        kept_boxes.append(global_boxes[index])
        kept_scores.append(global_scores[index])
        kept_labels.append(global_labels[index])
    if kept_boxes:
        # Per-label NMS (consistent with the tile NMS).
        keep_indices = []
        for label in global_labels.unique():
            mask = global_labels == label
            order = nms(global_boxes[mask], global_scores[mask], nms_iou_threshold)
            keep_indices.append(torch.nonzero(mask).flatten()[order])
        keep_indices = torch.cat(keep_indices)
        global_boxes_t = global_boxes[keep_indices]
        global_scores_t = global_scores[keep_indices]
        global_labels_t = global_labels[keep_indices]
    else:
        global_boxes_t = torch.zeros(0, 4)
        global_scores_t = torch.zeros(0)
        global_labels_t = torch.zeros(0, dtype=torch.int64)

    tile_boxes_all, tile_scores_all, tile_labels_all = [], [], []
    for tile_index, output in enumerate(tile_outputs):
        boxes = output["bboxes"]
        if boxes.numel() == 0:
            continue
        x_start = float(tile_coordinates[tile_index][0])
        y_start = float(tile_coordinates[tile_index][1])
        boxes = boxes.clone()
        boxes[:, 0] += x_start
        boxes[:, 2] += x_start
        boxes[:, 1] += y_start
        boxes[:, 3] += y_start
        # Clip to the original canvas.
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
        for label in output["labels"].unique():
            mask = output["labels"] == label
            order = nms(boxes[mask], output["scores"][mask], nms_iou_threshold)
            tile_boxes_all.append(boxes[mask][order])
            tile_scores_all.append(output["scores"][mask][order])
            tile_labels_all.append(output["labels"][mask][order])

    if not tile_boxes_all:
        return {"bboxes": global_boxes_t, "scores": global_scores_t,
                "labels": global_labels_t}
    tile_boxes_t = torch.cat(tile_boxes_all)
    tile_scores_t = torch.cat(tile_scores_all)
    tile_labels_t = torch.cat(tile_labels_all)

    # Drop tile boxes duplicating a same-label global box.
    keep_mask = torch.ones(tile_boxes_t.shape[0], dtype=torch.bool)
    for index in range(tile_boxes_t.shape[0]):
        same_label = global_labels_t == tile_labels_t[index]
        if same_label.any() and global_boxes_t[same_label].shape[0] > 0:
            ious = box_iou(
                tile_boxes_t[index : index + 1], global_boxes_t[same_label]
            )
            if (ious > global_local_iou_threshold).any():
                keep_mask[index] = False
    merged_boxes = torch.cat([global_boxes_t, tile_boxes_t[keep_mask]])
    merged_scores = torch.cat([global_scores_t, tile_scores_t[keep_mask]])
    merged_labels = torch.cat([global_labels_t, tile_labels_t[keep_mask]])
    order = merged_scores.argsort(descending=True)
    return {
        "bboxes": merged_boxes[order],
        "scores": merged_scores[order],
        "labels": merged_labels[order],
    }
