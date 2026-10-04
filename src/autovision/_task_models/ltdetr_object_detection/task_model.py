# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""LT-DETR object detection task model (inference/export facing).

Assembly (baseline ``dinov3_ltdetr``): SSL backbone wrapper -> stride
8/16/32 pyramid -> shared HybridEncoder -> RT-DETRv2 or D-FINE decoder.
Reuses the framework's shared detection components and the shared top-k
decoder, so the coordinate contract is identical to the RT-DETR task
family. Per-variant encoder/decoder recipes come from the registry spec.
"""

from __future__ import annotations

import pathlib
from typing import Any, Literal

import torch
from torch import nn

from autovision._models import get_wrapped_model
from autovision._models.dinov2.dinov2_wrapper import TransformersDinov2Wrapper
from autovision._export import ONNXExportMixin, ONNXExportPrecisionPolicy
from autovision.errors import CheckpointError
from autovision._task_models.object_detection_components.decode import (
    decode_detection_outputs,
)
from autovision._task_models.object_detection_components.dfine_decoder import (
    DFINEv2Decoder,
)
from autovision._task_models.object_detection_components.hybrid_encoder import (
    HybridEncoder,
)
from autovision._task_models.object_detection_components.rtdetrv2_decoder import (
    RTDETRv2Decoder,
)
from autovision._task_models.rtdetr_object_detection.task_model import (
    ObjectDetectionOutput,
)
from autovision._task_models.task_model import TaskModel
from autovision._transforms.transform_args import NormalizeArgs

from .backbone_wrappers import (
    ConvNeXtDetectionBackbone,
    ECViTDetectionBackbone,
    ViTDetectionBackbone,
)
from .config import (
    LTDETRVariantSpec,
    check_image_size,
    resolve_ltdetr_variant,
)

__all__ = ["LTDETRObjectDetection"]


def _build_vit_test_backbone() -> TransformersDinov2Wrapper:
    """Tiny offline DINOv2 build for the ``vittest`` registry variant."""
    import transformers

    config = transformers.Dinov2Config(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        image_size=224,
        patch_size=14,
        num_register_tokens=0,
    )
    return TransformersDinov2Wrapper(transformers.Dinov2Model(config))


def _build_ecvit_backbone(
    spec: LTDETRVariantSpec, backbone_weights: str | None
) -> "ECViTModelWrapper":
    from autovision._models.ecvit.ecvit import ECViTModelWrapper

    return ECViTModelWrapper(spec.backbone_name, weights_path=backbone_weights)


class LTDETRObjectDetection(ONNXExportMixin, TaskModel):
    """SSL-backbone LT-DETR detector (three-stage inference contract)."""

    onnx_export_precision_policy = ONNXExportPrecisionPolicy(
        fp32_onnx_op_types=("Softmax", "MatMul")
    )

    def __init__(
        self,
        num_classes: int,
        model_name: str = "dinov2-vits14-notpretrained",
        image_size: tuple[int, int] | None = None,
        decoder: Literal["rtdetrv2", "dfine", "auto"] = "auto",
        reg_max: int = 32,
        hidden_dim: int | None = None,
        num_queries: int | None = None,
        dec_layers: int | None = None,
        num_heads: int | None = None,
        num_top_queries: int = 300,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        backbone_pretrained: bool = False,
        backbone_weights: str | None = None,
        load_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(init_args=locals(), ignore_args={"load_weights", "kwargs", "self"})
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.normalize = NormalizeArgs()

        # resolve_ltdetr_variant is idempotent: accepts both the public
        # "ltdetr/<backbone>" name and the bare backbone key stored in
        # training init-args / exported checkpoints.
        _, spec = resolve_ltdetr_variant(model_name)
        self.variant_spec: LTDETRVariantSpec = spec
        if image_size is None:
            # Default 640 snapped UP to the patch grid (644 for patch 14),
            # mirroring the train-args resolution.
            divisor = 2 * spec.patch_size
            side = -(-640 // divisor) * divisor
            self.image_size = (side, side)
        else:
            check_image_size(tuple(image_size), spec)
            self.image_size = tuple(image_size)
        hidden_dim = hidden_dim or spec.hidden_dim
        dec_layers = dec_layers or spec.dec_layers
        num_heads = num_heads or spec.dec_num_heads
        num_queries = num_queries or spec.num_queries
        self.hidden_dim = hidden_dim
        if decoder == "auto":
            decoder = spec.decoder_default
        if decoder not in ("rtdetrv2", "dfine"):
            raise ValueError(f"Unknown LT-DETR decoder: '{decoder}'.")
        self.decoder_name = decoder

        if spec.requires_dinov3 and not _dinov3_supported():
            raise ValueError(
                f"LT-DETR variant '{model_name}' needs the DINOv3 classes "
                "from 'transformers', which are not available in the "
                "installed version."
            )

        if spec.backbone_name.startswith("vittest"):
            backbone_wrapper: Any = _build_vit_test_backbone()
        elif spec.family == "ecvit":
            backbone_wrapper = _build_ecvit_backbone(
                spec, backbone_weights=backbone_weights
            )
        else:
            full_name = f"{spec.backbone_package}/{spec.backbone_name}"
            backbone_wrapper = get_wrapped_model(
                full_name, num_input_channels=3, load_weights=load_weights
            )
        self.backbone_wrapper = backbone_wrapper

        if spec.family == "vit":
            self.backbone: nn.Module = ViTDetectionBackbone(
                backbone_wrapper,
                interaction_indexes=spec.interaction_indexes,
                finetune=True,
                use_sta=spec.use_sta,
                conv_inplane=spec.conv_inplane,
                hidden_dim=hidden_dim,
                project_features=spec.project_features,
                resize_features=True,
            )
        elif spec.family == "ecvit":
            self.backbone = ECViTDetectionBackbone(backbone_wrapper)
        else:
            self.backbone = ConvNeXtDetectionBackbone(
                backbone_wrapper, finetune=True
            )
        in_channels = self.backbone.in_channels

        if backbone_weights is not None and spec.family != "ecvit":
            # ECViT loads its backbone weights inside the wrapper.
            self._load_backbone_weights(backbone_weights)

        self.encoder = HybridEncoder(
            in_channels,
            hidden_dim=hidden_dim,
            dim_feedforward=spec.enc_dim_feedforward,
            expansion=spec.enc_expansion,
            depth_mult=spec.enc_depth_mult,
        )
        decoder_kwargs: dict[str, Any] = dict(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=[hidden_dim] * 3,
            feat_strides=[8, 16, 32],
            num_levels=3,
            num_points=list(spec.dec_num_points),
            nhead=num_heads,
            num_layers=dec_layers,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
        )
        if decoder == "dfine":
            self.decoder: nn.Module = DFINEv2Decoder(
                **decoder_kwargs, reg_max=reg_max
            )
        else:
            self.decoder = RTDETRv2Decoder(**decoder_kwargs)

    # ── backbone weights ────────────────────────────────────────────────
    def _load_backbone_weights(self, backbone_weights: str) -> None:
        try:
            state = torch.load(
                backbone_weights, map_location="cpu", weights_only=True
            )
        except Exception as error:
            raise CheckpointError(
                f"Failed to load backbone weights '{backbone_weights}' "
                f"under the safe loader (weights_only=True): {error}. "
                f"Legacy pickles with non-tensor payloads are rejected; "
                f"convert the file once with torch.load(..., "
                f"weights_only=False) in a trusted context and re-save "
                f"the plain state_dict with torch.save."
            ) from error
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        target = self.backbone.backbone_model
        incompatible = target.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                f"backbone_weights '{backbone_weights}' contains keys unknown to "
                f"the backbone: {incompatible.unexpected_keys[:5]}..."
            )

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

    def _forward_train(
        self, x: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None = None
    ) -> dict:
        feats = list(self.backbone(x))
        outs = self.encoder(feats)
        return self.decoder(outs, targets=targets)

    def forward_backend(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._forward_train(x)
        return out["pred_logits"], out["pred_boxes"]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_backend(x)

    def _as_output(self, raw: Any) -> ObjectDetectionOutput:
        if isinstance(raw, ObjectDetectionOutput):
            return raw
        logits, boxes = raw
        return ObjectDetectionOutput(logits=logits, boxes=boxes)

    def postprocess(
        self,
        raw_outputs: Any,
        metadata: list[dict[str, Any]],
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Shared DETR decode: normalized cxcywh -> original-pixel xyxy."""
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


def _dinov3_supported() -> bool:
    try:
        import transformers

        return hasattr(transformers, "Dinov3ConvNextModel") and hasattr(
            transformers, "Dinov3ViTModel"
        )
    except ImportError:
        return False
