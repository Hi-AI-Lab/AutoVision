# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""DEIMv2 object detection task model (inference/export facing).

Per framework policy (see the RF-DETR task model), the detector is
HuggingFace ``Deimv2ForObjectDetection`` (transformers >= 5.9, gated at
runtime — 5.9.0 is the first release shipping the Deimv2 classes, verified
by wheel inspection). The HF class carries the official criterion
(Hungarian matcher + VFL/MAL + L1 + GIoU + the D-FINE-style FGL/DDF
distributional losses, with DEIMv2's dense one-to-one auxiliary matching)
inside ``forward(labels=...)`` — the task model forwards it instead of
re-implementing it. Inference keeps the AutoVision three-stage contract:
``forward`` returns raw ``(logits, boxes)`` and ``postprocess`` decodes
sigmoid top-k to original pixels via the shared DETR decode.

Weights are the transformers-format conversions of the official
Intellindust COCO checkpoints, pinned in the weight manifest (the official
``Intellindust/*`` repos ship the original-framework state dict and are
not loadable here). The checkpoint's ``eval_size`` is overridden to the
AutoVision image size at load so the encoder's cached positional
embeddings match the fixed network input; anchor grids are recomputed per
input, so other resolutions stay correct. Backbone BatchNorm stays
trainable at load (``freeze_backbone_batch_norms=False``), matching the
official ``freeze_norm: False``.
"""

from __future__ import annotations

import pathlib
from typing import Any

import torch
from torch import nn

from autovision.augment.transform_args import NormalizeArgs
from autovision.encoders.weight_verification import load_pretrained_verified
from autovision.export import ONNXExportMixin
from autovision.tasks.detection.deimv2.config import (
    resolve_deimv2_variant,
)
from autovision.tasks.detection.shared.decode import (
    decode_detection_outputs,
)
from autovision.tasks.task_model import TaskNetwork
from autovision.tasks.task_model_io import BaseModelOutput

__all__ = ["Deimv2ObjectDetection", "Deimv2ObjectDetectionOutput"]


def _deimv2_classes() -> dict[str, Any] | None:
    """Live detection of the DEIMv2 detection classes in transformers."""
    try:
        import transformers
    except ImportError:
        return None
    config = getattr(transformers, "Deimv2Config", None)
    od_model = getattr(transformers, "Deimv2ForObjectDetection", None)
    if None in (config, od_model):
        return None
    return {"config": config, "od_model": od_model}


def _support_error(model_name: str) -> ImportError:
    return ImportError(
        f"DEIMv2 model '{model_name}' requires a transformers version with "
        f"DEIMv2 support (classes Deimv2Config/Deimv2ForObjectDetection, "
        f"transformers >= 5.9). Upgrade: pip install -U 'autovision[deimv2]'."
    )


class Deimv2ObjectDetectionOutput(BaseModelOutput):
    """Exported graph outputs: class logits + normalized cxcywh boxes."""

    def __init__(self, logits: torch.Tensor, boxes: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.boxes = boxes


def _retile(head: nn.Linear, num_labels: int) -> nn.Linear:
    """Resize one classification head (DEIMv2 semantics): growing tiles the
    existing rows, shrinking truncates — the pretrained head is never
    randomly reset."""
    old_n = head.weight.shape[0]
    if old_n == num_labels:
        return head
    new = nn.Linear(head.in_features, num_labels, bias=head.bias is not None)
    with torch.no_grad():
        index = torch.arange(num_labels, device=head.weight.device) % old_n
        new.weight.copy_(head.weight.data[index])
        if head.bias is not None:
            new.bias.copy_(head.bias.data[index])
    return new


def _reinitialize_class_head(model: nn.Module, num_labels: int) -> None:
    """Resize the classification heads to ``num_labels``.

    Covers the per-layer decoder heads (``class_embed``, also referenced by
    ``model.decoder.class_embed`` — the same ModuleList, mutated in place)
    and the two-stage encoder head (``model.enc_score_head``).
    ``config.num_labels`` is synced too: the HF loss criterion is built
    from the config on every forward, and its one-hot targets must match
    the retiled head width.
    """
    model.config.num_labels = num_labels
    cached_loss = model.__dict__.get("_loss_function")
    if cached_loss is not None and hasattr(cached_loss, "num_classes"):
        cached_loss.num_classes = num_labels
    for index in range(len(model.class_embed)):
        model.class_embed[index] = _retile(model.class_embed[index], num_labels)
    enc_score_head = getattr(model.model, "enc_score_head", None)
    if enc_score_head is not None:
        model.model.enc_score_head = _retile(enc_score_head, num_labels)


class Deimv2ObjectDetection(ONNXExportMixin, TaskNetwork):
    """DEIMv2 detector (HF transformers) in the AutoVision task contract."""

    def __init__(
        self,
        num_classes: int,
        model: str = "deimv2-hgnetv2-n",
        image_size: tuple[int, int] = (640, 640),
        num_select: int = 300,
        load_weights: bool = False,
        config_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            init_args=locals(), ignore_args={"load_weights", "kwargs", "self"}
        )
        info = resolve_deimv2_variant(model)
        self.num_classes = num_classes
        self.model_name = model
        self.image_size = tuple(image_size)
        self.num_top_queries = num_select
        # DEIMv2 trains on ImageNet normalization (official configs).
        self.normalize = NormalizeArgs()

        classes = _deimv2_classes()
        if classes is None:
            raise _support_error(f"deimv2/{model}")
        if load_weights:
            if not info.get("hf"):
                # Registry entries without a transformers-format conversion
                # (deimv2-dinov3-l / fsdc-vittest, design doc 22 PR-0) can
                # only be built locally — fail with the reason, not a
                # bogus hub download of "None".
                raise ValueError(
                    f"deimv2 variant '{model}' has no transformers-format "
                    f"weight conversion; build locally with "
                    f"load_weights=False."
                )
            # Load the verified checkpoint with its native class count, then
            # tile/truncate the head — growing randomly-initialized heads is
            # exactly what the official loader avoids. eval_size tracks the
            # AutoVision input size: the encoder caches positional embeddings
            # against it in eval mode. Backbone BatchNorm stays trainable —
            # the official configs set freeze_norm: False, and frozen COCO
            # BN statistics stall drone-domain adaptation (transformers
            # freezes them by default).
            hf_model = load_pretrained_verified(
                classes["od_model"],
                str(info["hf"]),
                {
                    "eval_size": [self.image_size[0], self.image_size[1]],
                    "freeze_backbone_batch_norms": False,
                    # Config overrides applied on top of the checkpoint's
                    # own config (FSDC 4-level pyramid expansion, design
                    # doc 22 §4.5 — must precede construction because the
                    # deformable attention bakes num_points_scale buffers
                    # from the config).
                    **(config_overrides or {}),
                },
            )
        else:
            # Local (offline) build from the registry's architecture table,
            # mirroring the transformers-format checkpoint configs.
            config_kwargs = dict(info["config"])
            config_kwargs.update(config_overrides or {})
            backbone_kwargs = dict(config_kwargs["backbone_config"])
            if backbone_kwargs.get("model_type") == "dinov3_vit":
                backbone_kwargs.setdefault("image_size", max(self.image_size))
            config_kwargs["backbone_config"] = backbone_kwargs
            config_kwargs["num_labels"] = num_classes
            config_kwargs["eval_size"] = [self.image_size[0], self.image_size[1]]
            config_kwargs["freeze_backbone_batch_norms"] = False
            config = classes["config"](**config_kwargs)
            hf_model = classes["od_model"](config)
        _reinitialize_class_head(hf_model, num_classes)
        self._model = hf_model

    # ── three-stage contract ────────────────────────────────────────────
    def preprocess_image(self, image: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        from torchvision.transforms.v2 import functional as TF

        from autovision.data.file_helpers import open_image

        if isinstance(image, (str, pathlib.Path)):
            tensor = open_image(image).float() / 255.0
        elif isinstance(image, torch.Tensor):
            tensor = image.float()
            if tensor.ndim == 3 and tensor.max() > 1.5:
                tensor = tensor / 255.0
        else:
            tensor = TF.pil_to_tensor(image).float() / 255.0
        _, height, width = tensor.shape
        resized = TF.resize(
            tensor,
            [self.image_size[0], self.image_size[1]],
            antialias=True,
        )
        return resized, {"orig_w": width, "orig_h": height}

    def preprocess_batch(self, batch: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.normalize.mean, device=batch.device).view(-1, 1, 1)
        std = torch.tensor(self.normalize.std, device=batch.device).view(-1, 1, 1)
        return (batch - mean) / std

    def _labels(
        self, targets: list[dict[str, torch.Tensor]] | None
    ) -> list[dict] | None:
        if targets is None:
            return None
        return [
            {"class_labels": target["labels"], "boxes": target["boxes"]}
            for target in targets
        ]

    def _forward_train(
        self, x: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None = None
    ) -> dict:
        outputs = self._model(pixel_values=x, labels=self._labels(targets))
        return {
            "pred_logits": outputs["logits"],
            "pred_boxes": outputs["pred_boxes"],
            "loss": outputs.get("loss"),
            "loss_dict": outputs.get("loss_dict") or {},
        }

    def forward_backend(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, boxes); the tuple form keeps the export graph
        free of custom dataclasses (dynamo limitation)."""
        out = self._forward_train(x)
        return out["pred_logits"], out["pred_boxes"]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_backend(x)

    def _as_output(self, raw) -> Deimv2ObjectDetectionOutput:
        if isinstance(raw, Deimv2ObjectDetectionOutput):
            return raw
        logits, boxes = raw
        return Deimv2ObjectDetectionOutput(logits=logits, boxes=boxes)

    def postprocess(
        self,
        raw_outputs: Any,
        metadata: list[dict[str, Any]],
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Decode raw outputs to per-image boxes in original-frame pixels
        (sigmoid top-k; official ``PostProcess`` semantics)."""
        return decode_detection_outputs(
            raw_outputs,
            metadata,
            num_top_queries=self.num_top_queries,
            threshold=threshold,
        )

    # ── export ──────────────────────────────────────────────────────────
    @property
    def model_input_spec(self):
        from autovision.tasks.task_model_io import ModelInputSpec, TensorSpec

        return ModelInputSpec(
            input_specs={
                "images": TensorSpec(
                    shape=(3, *self.image_size), dtype=torch.float32, is_batched=True
                )
            }
        )

    def verify_onnx_export_outputs(
        self, torch_outputs: tuple, onnx_outputs: list
    ) -> None:
        torch.testing.assert_close(
            torch_outputs[0].sum(dim=1),
            torch.as_tensor(onnx_outputs[0]).float().sum(dim=1),
            atol=5e-3,
            rtol=1e-1,
            equal_nan=True,
            check_device=False,
            check_dtype=False,
        )
        torch.testing.assert_close(
            torch_outputs[1].sum(dim=1),
            torch.as_tensor(onnx_outputs[1]).float().sum(dim=1),
            atol=5e-3,
            rtol=1e-1,
            equal_nan=True,
            check_device=False,
            check_dtype=False,
        )
