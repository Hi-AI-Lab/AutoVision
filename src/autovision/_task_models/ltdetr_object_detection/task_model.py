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
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

from autovision._export import ONNXExportMixin, ONNXExportPrecisionPolicy
from autovision._models import get_wrapped_model
from autovision._models.dinov2.dinov2_wrapper import TransformersDinov2Wrapper
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
from autovision._task_models.object_detection_components.ssl_contrastive import (
    SSLProjectionHead,
)
from autovision._task_models.rtdetr_object_detection.task_model import (
    ObjectDetectionOutput,
)
from autovision._task_models.task_model import TaskModel
from autovision._transforms.transform_args import NormalizeArgs
from autovision.errors import CheckpointError

if TYPE_CHECKING:
    from autovision._models.ecvit.ecvit import ECViTModelWrapper

from .backbone_wrappers import (
    ConvNeXtDetectionBackbone,
    ECViTDetectionBackbone,
    ViTDetectionBackbone,
)
from .config import (
    LTDETRDensityArgs,
    LTDETRSSLArgs,
    LTDETRVariantSpec,
    check_image_size,
    resolve_ltdetr_variant,
)
from .dome_decoder import DomeDFINEv2Decoder
from .dome_encoder import DomeHybridEncoder

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


def _relabel_prefix_drift(state: dict, target: torch.nn.Module) -> dict:
    """Absorb cross-version backbone state-dict key renames.

    External ``backbone_weights`` files are converted against one
    transformers layout and consumed under another (5.19 moved the
    DINOv3-ConvNeXt weights under a ``model.`` prefix while the final
    ``layer_norm`` stayed top-level). Applied ONLY when it explains every
    file key — a pure key-space rename; tensors are never touched, and
    any unexplainable key returns the state unchanged so the caller's
    strict check reports the raw names.
    """
    expected = target.state_dict().keys()
    expected_set = set(expected)
    if set(state) == expected_set:
        return state
    relabeled = {}
    for key, value in state.items():
        if key in expected_set:
            relabeled[key] = value
        elif f"model.{key}" in expected_set:
            relabeled[f"model.{key}"] = value
        else:
            return state
    return relabeled


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
        reg_scale: float = 4.0,
        enc_dim_feedforward: int | Literal["auto"] = "auto",
        enc_expansion: float | Literal["auto"] = "auto",
        enc_depth_mult: float | Literal["auto"] = "auto",
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
        pyramid_levels: int = 3,
        density: LTDETRDensityArgs | dict | None = None,
        ssl: LTDETRSSLArgs | dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            init_args=locals(), ignore_args={"load_weights", "kwargs", "self"}
        )
        # Exported checkpoints are deserialized with the pickle-free safe
        # loader (weights_only=True), which rejects arbitrary objects —
        # snapshot the density/ssl configs as plain dicts; __init__
        # re-coerces dicts back into the args classes below, so the round
        # trip is lossless. Same for Path-valued args: the export safe
        # loader (weights_only=True) rejects pathlib types, and this
        # task model treats backbone_weights as a str either way.
        if isinstance(self._init_args.get("density"), LTDETRDensityArgs):
            self._init_args["density"] = self._init_args["density"].model_dump()
        if isinstance(self._init_args.get("ssl"), LTDETRSSLArgs):
            self._init_args["ssl"] = self._init_args["ssl"].model_dump()
        if isinstance(self._init_args.get("backbone_weights"), pathlib.PurePath):
            self._init_args["backbone_weights"] = str(
                self._init_args["backbone_weights"]
            )
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

        # Density-guided branch (Dome-DETR port). ``None`` keeps the
        # baseline contract bit-for-bit: no dome modules are constructed.
        if pyramid_levels not in (3, 4):
            raise ValueError(f"pyramid_levels must be 3 or 4, got {pyramid_levels}.")
        if pyramid_levels == 4 and spec.family != "convnext":
            raise ValueError(
                f"pyramid_levels=4 needs a ConvNeXt-family backbone (the "
                f"only wrapper exposing a stride-4 stage); "
                f"'{spec.backbone_name}' is family '{spec.family}'."
            )
        if isinstance(density, dict):
            density = LTDETRDensityArgs(**density)
        if density is not None:
            if decoder != "dfine":
                raise ValueError(
                    "density (dynamic queries / DeFE / MWAS) requires the "
                    "D-FINE decoder; pass decoder='dfine' (the evaluated "
                    "v1 combination, design doc 13 §3.3)."
                )
            if num_queries > density.max_num_select:
                raise ValueError(
                    f"num_queries ({num_queries}) must not exceed "
                    f"density.max_num_select ({density.max_num_select})."
                )
        self.density_args: LTDETRDensityArgs | None = density
        self._density_enabled = density is not None

        # Per-object contrastive SSL branch (design doc 14). ``None`` or
        # ``mode="off"`` keeps the baseline contract: no head is constructed
        # (mode="off" + two_crop is the batch-matched no-SSL baseline — the
        # doubled data pipeline runs, the loss does not).
        if isinstance(ssl, dict):
            ssl = LTDETRSSLArgs(**ssl)
        if ssl is not None and decoder != "dfine":
            # v1 boundary (design doc 14 §3.3): the RT-DETRv2 decoder has
            # no query_features emission, so the whole ssl config —
            # including the mode="off" batch-matched baseline — is
            # restricted to the D-FINE decoder family.
            raise ValueError(
                "ssl (per-object contrastive head) requires the D-FINE "
                "decoder in v1; pass decoder='dfine' (design doc 14 §3.3)."
            )
        self.ssl_args: LTDETRSSLArgs | None = ssl
        # Route B (design doc 14 §4.10): label-free SSL-only pretraining off
        # the decoder's top-k candidate features. The density-gated selection
        # is not supported as an instance source (zero-padded candidates).
        self._ssl_candidates_mode = (
            ssl is not None and ssl.instance_source == "encoder_candidates"
        )
        if self._ssl_candidates_mode and self._density_enabled:
            raise ValueError(
                "ssl instance_source='encoder_candidates' (SSL-only "
                "pretraining) is not supported with the density branch; "
                "disable density (design doc 14 §4.10)."
            )

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
                backbone_wrapper,
                finetune=True,
                patch_size=spec.patch_size,
                num_stages=pyramid_levels,
            )
        in_channels = self.backbone.in_channels

        if backbone_weights is not None and spec.family != "ecvit":
            # ECViT loads its backbone weights inside the wrapper.
            self._load_backbone_weights(backbone_weights)

        # Encoder width knobs: "auto" follows the variant spec (the
        # ConvNeXt specs carry the heavy baseline recipe — 2048/1.0/1.0);
        # explicit values align with the official Dome-S "S" encoder
        # (1024/0.5/0.34), a ~4-6x lighter randomly-initialized module.
        enc_ffn = (
            spec.enc_dim_feedforward
            if enc_dim_feedforward == "auto"
            else int(enc_dim_feedforward)
        )
        enc_exp = (
            spec.enc_expansion if enc_expansion == "auto" else float(enc_expansion)
        )
        enc_depth = (
            spec.enc_depth_mult if enc_depth_mult == "auto" else float(enc_depth_mult)
        )

        if self._density_enabled:
            self.encoder = DomeHybridEncoder(
                in_channels,
                hidden_dim=hidden_dim,
                dim_feedforward=enc_ffn,
                expansion=enc_exp,
                depth_mult=enc_depth,
                use_encoder_idx=[pyramid_levels - 1],
                density=density,
            )
        else:
            self.encoder = HybridEncoder(
                in_channels,
                hidden_dim=hidden_dim,
                dim_feedforward=enc_ffn,
                expansion=enc_exp,
                depth_mult=enc_depth,
                use_encoder_idx=[pyramid_levels - 1],
            )
        # Pyramid strides: 3-level [8, 16, 32]; 4-level prepends the
        # ConvNeXt stride-4 stage (Dome-faithful, design doc 13 D1).
        feat_strides = [8, 16, 32] if pyramid_levels == 3 else [4, 8, 16, 32]
        num_points = list(spec.dec_num_points)
        if len(num_points) != pyramid_levels:
            if len(set(num_points)) != 1:
                raise ValueError(
                    f"dec_num_points {num_points} cannot be expanded to "
                    f"pyramid_levels={pyramid_levels}; specify per-level "
                    f"values explicitly."
                )
            num_points = num_points[:1] * pyramid_levels
        decoder_kwargs: dict[str, Any] = dict(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=[hidden_dim] * pyramid_levels,
            feat_strides=feat_strides,
            num_levels=pyramid_levels,
            num_points=num_points,
            nhead=num_heads,
            num_layers=dec_layers,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
        )
        if decoder == "dfine":
            if self._density_enabled:
                self.decoder: nn.Module = DomeDFINEv2Decoder(
                    **decoder_kwargs,
                    reg_max=reg_max,
                    reg_scale=reg_scale,
                    min_num_select=density.resolved_min_num_select(num_queries),
                    max_num_select=density.max_num_select,
                    nms_iou_base=density.nms_iou_base,
                    nms_iou_density_scale=density.nms_iou_density_scale,
                )
            else:
                self.decoder = DFINEv2Decoder(
                    **decoder_kwargs,
                    reg_max=reg_max,
                    reg_scale=reg_scale,
                    ssl_candidate_features=self._ssl_candidates_mode,
                )
        else:
            self.decoder = RTDETRv2Decoder(**decoder_kwargs)

        # The SSL head is built LAST so its random init never shifts the
        # RNG stream of the baseline modules: with a fixed seed, an
        # ssl-enabled model and the baseline model carry IDENTICAL
        # backbone/encoder/decoder weights (bit-for-bit eval contract).
        self.ssl_head: nn.Module | None = None
        self._ssl_sigreg = ssl is not None and ssl.sigreg_coef > 0
        if ssl is not None and ssl.mode != "off":
            self.ssl_head = SSLProjectionHead(
                hidden_dim=hidden_dim,
                projection_dim=ssl.projector_dim,
                kind=ssl.projector,
                dropout=ssl.projector_dropout,
            )

    # ── backbone weights ────────────────────────────────────────────────
    def _load_backbone_weights(self, backbone_weights: str) -> None:
        try:
            if str(backbone_weights).endswith(".safetensors"):
                # Safetensors is the preferred source format: no pickle
                # surface at all, tensors only.
                from safetensors.torch import load_file as safetensors_load_file

                state = safetensors_load_file(str(backbone_weights), device="cpu")
            else:
                state = torch.load(
                    backbone_weights, map_location="cpu", weights_only=True
                )
        except Exception as error:
            raise CheckpointError(
                f"Failed to load backbone weights '{backbone_weights}' "
                f"under the safe loader (weights_only=True / safetensors): "
                f"{error}. Legacy pickles with non-tensor payloads are "
                f"rejected; convert the file once with torch.load(..., "
                f"weights_only=False) in a trusted context and re-save "
                f"the plain state_dict with torch.save."
            ) from error
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        target = self.backbone.backbone_model
        state = _relabel_prefix_drift(state, target)
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
        if self._density_enabled:
            outs, defe_info = self.encoder(feats, targets=targets)
            out = self.decoder(outs, targets=targets, defe_info=defe_info)
        else:
            outs = self.encoder(feats)
            out = self.decoder(outs, targets=targets)
        if self.ssl_head is not None:
            if self._ssl_candidates_mode:
                # Route B: the pool is the top-k candidate features. Emitted
                # in train AND eval — the contrastive loss is the only
                # objective, so the validation loss is the held-out SSL loss.
                candidate_features = out.get("ssl_candidate_features")
                if candidate_features is None:
                    raise ValueError(
                        "ssl instance_source='encoder_candidates' requires the "
                        "decoder to emit ssl_candidate_features; the SSL-only "
                        "mode requires the D-FINE decoder family "
                        "(design doc 14 §4.10)."
                    )
                if self._ssl_sigreg:
                    out["ssl_z1"], out["ssl_s1"] = self.ssl_head.forward_with_raw(
                        candidate_features
                    )
                    out["ssl_z2"], out["ssl_s2"] = self.ssl_head.forward_with_raw(
                        candidate_features
                    )
                else:
                    out["ssl_z1"] = self.ssl_head(candidate_features)
                    out["ssl_z2"] = self.ssl_head(candidate_features)
            elif self.training:
                # Two independent dropout draws through the projection head
                # give the two views of every query; the criterion selects
                # the matched rows (and the cross-view pairs under two_crop).
                query_features = out.get("query_features")
                if query_features is None:
                    raise ValueError(
                        "ssl_head is enabled but the decoder produced no "
                        "query_features; the SSL head requires the D-FINE "
                        "decoder family (design doc 14 §4.3)."
                    )
                if self._ssl_sigreg:
                    # SIGReg (design doc 16 R1-A1): also emit the RAW
                    # pre-normalization projector outputs — the unit-sphere
                    # sample can never look Gaussian, so the distribution
                    # test must see the raw projector space.
                    out["ssl_z1"], out["ssl_s1"] = self.ssl_head.forward_with_raw(
                        query_features
                    )
                    out["ssl_z2"], out["ssl_s2"] = self.ssl_head.forward_with_raw(
                        query_features
                    )
                else:
                    out["ssl_z1"] = self.ssl_head(query_features)
                    out["ssl_z2"] = self.ssl_head(query_features)
        return out

    # deploy(): inherited from TaskModel — it recursively calls
    # ``convert_to_deploy`` on every module, which is exactly the export
    # hook the density-guided modules implement (design doc 13 §4.8):
    # DomeHybridEncoder/DomeDFINEv2Decoder fall back to the fixed-proposal
    # baseline graph, while density-off models behave exactly as before.

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

        # transformers ships the classes as ``DINOv3*`` (4.x lazy exports
        # and 5.x alike); keep the ``Dinov3*`` fallbacks for older layouts.
        vit = getattr(transformers, "DINOv3ViTModel", None) or getattr(
            transformers, "Dinov3ViTModel", None
        )
        convnext = getattr(transformers, "DINOv3ConvNextModel", None) or getattr(
            transformers, "Dinov3ConvNextModel", None
        )
        return vit is not None and convnext is not None
    except ImportError:
        return False
