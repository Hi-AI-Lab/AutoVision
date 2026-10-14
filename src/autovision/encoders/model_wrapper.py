# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Model wrapper protocols.

Every backbone is adapted to the framework through the ``EncoderProtocol``
protocol group — structural subtyping, no base class required. The composed
``EncoderProtocol`` protocol is what the pretraining path consumes;
``ArchitectureInfoGettable`` is optional and drives ``"auto"`` decisions such
as weight decay per architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict

from torch import Tensor, nn
from typing_extensions import (
    NotRequired,
    Required,
    get_protocol_members,
    runtime_checkable,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "ArchitectureInfo",
    "ArchitectureInfoGettable",
    "FeatureDim",
    "ForwardFeatures",
    "ForwardFeaturesOutput",
    "ForwardPool",
    "ForwardPoolOutput",
    "ModelGetter",
    "EncoderProtocol",
    "MultiScaleFeatureCNN",
    "MultiScaleFeatureDims",
    "MultiScaleFeatureStrides",
    "MultiScaleFeatureViT",
    "NNModule",
    "PatchSize",
    "ForwardMultiScaleFeatures",
    "missing_model_wrapper_attrs",
]


class ForwardFeaturesOutput(TypedDict, total=False):
    features: Required[Tensor]  # (B, C, H_out, W_out)
    cls_token: NotRequired[Tensor]  # (B, C)


class ForwardPoolOutput(TypedDict, total=False):
    pooled_features: Required[Tensor]  # (B, C, 1, 1)


class ArchitectureInfo(TypedDict):
    model_type: Literal["convolutional", "transformer", "hybrid"]
    norm_type: Literal["batchnorm", "layernorm"]


@runtime_checkable
class ForwardFeatures(Protocol):
    def forward_features(self, x: Tensor, *args: Any, **kwargs: Any) -> ForwardFeaturesOutput: ...


@runtime_checkable
class ForwardPool(Protocol):
    def forward_pool(self, x: ForwardFeaturesOutput, *args: Any, **kwargs: Any) -> ForwardPoolOutput: ...


@runtime_checkable
class FeatureDim(Protocol):
    def feature_dim(self) -> int: ...


@runtime_checkable
class ModelGetter(Protocol):
    def get_model(self) -> Any: ...


@runtime_checkable
class ArchitectureInfoGettable(Protocol):
    def architecture_info(self) -> ArchitectureInfo: ...


@runtime_checkable
class NNModule(Protocol):
    """Structural subset of ``torch.nn.Module``."""

    def state_dict(self, *args: Any, **kwargs: Any) -> Any: ...

    def parameters(self, *args: Any, **kwargs: Any) -> Any: ...

    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> Any: ...

    def eval(self) -> Any: ...

    def modules(self) -> Any: ...

    def requires_grad_(self, requires_grad: bool = True) -> Any: ...


class EncoderProtocol(ForwardFeatures, ForwardPool, FeatureDim, ModelGetter, NNModule, Protocol):
    """Everything the pretraining path needs from a backbone."""


@runtime_checkable
class MultiScaleFeatureDims(Protocol):
    def multiscale_feature_dims(self) -> list[int]: ...


@runtime_checkable
class MultiScaleFeatureStrides(Protocol):
    def multiscale_feature_strides(self) -> list[int]: ...


@runtime_checkable
class PatchSize(Protocol):
    def patch_size(self) -> int: ...


@runtime_checkable
class ForwardMultiScaleFeatures(Protocol):
    def forward_multiscale_features(
        self, x: Tensor, layer_indices: Any
    ) -> list[ForwardFeaturesOutput]: ...


class MultiScaleFeatureViT(EncoderProtocol, ForwardMultiScaleFeatures, MultiScaleFeatureDims, PatchSize, Protocol): ...


class MultiScaleFeatureCNN(
    EncoderProtocol, ForwardMultiScaleFeatures, MultiScaleFeatureDims, MultiScaleFeatureStrides, Protocol
): ...


def missing_model_wrapper_attrs(
    model_wrapper: Any, exclude_module_attrs: bool = False
) -> list[str]:
    """Names required by :class:`EncoderProtocol` that the object does not have.

    Used to build actionable ``UnknownModelError`` messages.
    """
    members = get_protocol_members(EncoderProtocol)
    module_attrs = set(dir(nn.Module))
    missing = []
    for name in members:
        if exclude_module_attrs and name in module_attrs:
            continue
        if not hasattr(model_wrapper, name):
            missing.append(name)
    return sorted(missing)
