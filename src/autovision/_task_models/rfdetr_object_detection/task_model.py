# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""RF-DETR object detection task model (inference/export facing).

Per framework policy (see ``autovision._models.rfdetr``), the detector is
HuggingFace ``RfDetrForObjectDetection`` (transformers >= 5.9, gated at
runtime). The HF class carries the official training criterion (IA-BCE +
L1 + GIoU behind a Hungarian matcher) inside ``forward(labels=...)`` — the
task model forwards it instead of re-implementing it. Inference keeps the
AutoVision three-stage contract: ``forward`` returns raw
``(logits, boxes)`` and ``postprocess`` decodes sigmoid top-k to original
pixels via the shared DETR decode.
"""

from __future__ import annotations

import pathlib
from typing import Any

import torch
from torch import nn

from autovision._export import ONNXExportMixin
from autovision._models.weight_verification import load_pretrained_verified
from autovision._task_models.object_detection_components.decode import (
    decode_detection_outputs,
)
from autovision._task_models.rfdetr_object_detection.config import (
    resolve_rfdetr_variant,
)
from autovision._task_models.task_model import TaskModel
from autovision._task_models.task_model_io import BaseModelOutput
from autovision._transforms.transform_args import NormalizeArgs

__all__ = ["RFDETRObjectDetection", "RFDETRObjectDetectionOutput"]

# Reuse the SSL package's variant table (HF repo ids + decoder depths +
# backbone layouts — same checkpoints, verified supply chain).
from autovision._models.rfdetr.rfdetr_package import (  # noqa: E402
    _MODEL_INFO as _SSL_MODEL_INFO,
)
from autovision._models.rfdetr.rfdetr_package import (  # noqa: E402
    _support_error as _ssl_support_error,
)


def _od_classes() -> dict[str, Any] | None:
    """Live detection of the RF-DETR detection classes in transformers."""
    try:
        import transformers
    except ImportError:
        return None
    config = getattr(transformers, "RfDetrConfig", None)
    od_model = getattr(transformers, "RfDetrForObjectDetection", None)
    backbone_config = getattr(transformers, "RfDetrDinov2Config", None)
    if None in (config, od_model, backbone_config):
        return None
    return {
        "config": config,
        "od_model": od_model,
        "backbone_config": backbone_config,
    }


class RFDETRObjectDetectionOutput(BaseModelOutput):
    """Exported graph outputs: class logits + normalized cxcywh boxes."""

    def __init__(self, logits: torch.Tensor, boxes: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.boxes = boxes


def _reinitialize_class_head(model: nn.Module, num_labels: int) -> None:
    """Resize the classification heads to ``num_labels`` (rf-detr semantics).

    Mirrors the official ``reinitialize_detection_head``: growing tiles the
    existing rows, shrinking truncates — the pretrained head is never
    randomly reset. Covers the decoder head and the per-group two-stage
    encoder heads.
    """

    def _retile(head: nn.Linear) -> nn.Linear:
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

    model.class_embed = _retile(model.class_embed)
    enc_heads = getattr(model.model, "enc_out_class_embed", None)
    if enc_heads is not None:
        for index in range(len(enc_heads)):
            enc_heads[index] = _retile(enc_heads[index])


class RFDETRObjectDetection(ONNXExportMixin, TaskModel):
    """RF-DETR detector (HF transformers) in the AutoVision task contract."""

    def __init__(
        self,
        num_classes: int,
        model: str = "rf-detr-nano",
        image_size: tuple[int, int] = (384, 384),
        num_queries: int = 300,
        num_select: int = 300,
        load_weights: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            init_args=locals(), ignore_args={"load_weights", "kwargs", "self"}
        )
        resolve_rfdetr_variant(model)
        if model.endswith("-notpretrained"):
            raise ValueError(
                f"Unknown rfdetr detection model '{model}'. The '-notpretrained' "
                "suffix is a pretrain-package concept; set load_weights=False "
                "here instead."
            )
        self.num_classes = num_classes
        self.model_name = model
        self.image_size = tuple(image_size)
        self.num_top_queries = num_select
        self.normalize = NormalizeArgs()

        classes = _od_classes()
        if classes is None:
            raise _ssl_support_error(f"rfdetr/{model}")
        if load_weights:
            # Load the verified checkpoint with its native class count, then
            # tile/truncate the head — growing randomly-initialized heads is
            # exactly what the official loader avoids.
            hf_model = load_pretrained_verified(classes["od_model"], _repo_id(model))
        else:
            info = _SSL_MODEL_INFO[model]
            backbone_kwargs = dict(info["backbone"])
            backbone_kwargs["image_size"] = 224  # tiny hermetic builds
            backbone_config = classes["backbone_config"](
                hidden_size=384,
                num_hidden_layers=12,
                num_attention_heads=6,
                use_mask_token=True,
                **backbone_kwargs,
            )
            config = classes["config"](
                backbone_config=backbone_config,
                num_labels=num_classes,
                decoder_layers=info["decoder_layers"],
            )
            hf_model = classes["od_model"](config)
        _reinitialize_class_head(hf_model, num_classes)
        self._model = hf_model

    # ── three-stage contract ────────────────────────────────────────────
    def preprocess_image(self, image: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        from torchvision.transforms.v2 import functional as TF

        from autovision._data.file_helpers import open_image

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

    def _as_output(self, raw) -> RFDETRObjectDetectionOutput:
        if isinstance(raw, RFDETRObjectDetectionOutput):
            return raw
        logits, boxes = raw
        return RFDETRObjectDetectionOutput(logits=logits, boxes=boxes)

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
        from autovision._task_models.task_model_io import ModelInputSpec, TensorSpec

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


def _repo_id(model: str) -> str:
    """``rf-detr-nano[-notpretrained]`` → verified HF repo id."""
    info = _SSL_MODEL_INFO[model.removesuffix("-notpretrained")]
    return str(info["hf"])
