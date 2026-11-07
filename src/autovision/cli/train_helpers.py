# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pretraining command helpers: ordered config resolution and assembly.

Resolution order (design doc §5.3, mirrors the baseline):

    transform_args -> epochs -> strategy/precision -> batch_size/num_workers
    -> optimizer_args(wrapped_model) -> method_args(scaling, optimizer_args,
    wrapped_model)

Each resolver validates its input against the method-provided config class,
resolves ``"auto"`` in place, and asserts the result is fully resolved.
"""

from __future__ import annotations

from typing import Any, TypeVar

from torch.utils.data import DataLoader

from autovision.augment.method_transform import MethodTransform
from autovision.augment.transform_args import MethodTransformArgs
from autovision.config.validate import (
    assert_config_resolved,
    pydantic_model_validate,
)
from autovision.errors import ConfigValidationError, UnresolvedAutoError
from autovision.pretext.recipe import PretextRecipe
from autovision.scaling import (
    ScalingInfo,
    epochs_from_steps,
)

__all__ = [
    "get_dataloader",
    "get_epochs",
    "get_method",
    "get_method_args",
    "get_optimizer_args",
    "get_transform",
    "get_transform_args",
]

T = TypeVar("T")


def get_transform_args(
    method_cls: type[PretextRecipe], transform_args: dict | MethodTransformArgs | None
) -> MethodTransformArgs:
    args_cls = method_cls.transform_cls().transform_args_cls()
    args = pydantic_model_validate(args_cls, dict(transform_args or {}))
    args.resolve_auto_fields()
    args.resolve_incompatible()
    assert_config_resolved(args)
    return args


def get_transform(
    method_cls: type[PretextRecipe], transform_args: MethodTransformArgs
) -> MethodTransform:
    assert_config_resolved(transform_args)
    return method_cls.transform_cls()(transform_args)


def get_dataset_size(dataset) -> int:
    try:
        return len(dataset)
    except TypeError as error:
        raise UnresolvedAutoError(
            "Dataset size is required to resolve 'epochs=auto'."
        ) from error


def get_epochs(
    method_cls: type[PretextRecipe],
    epochs: int | str,
    dataset_size: int,
    batch_size: int,
) -> int:
    """Resolve ``epochs``: user value > step-based recipe > default epochs."""
    if epochs != "auto":
        # SERVE-13: a float like 2.5 used to silently truncate and a
        # non-numeric string died on a bare ValueError with no parameter
        # name in the message.
        try:
            resolved = int(epochs)
        except (TypeError, ValueError) as error:
            raise ConfigValidationError(
                f"epochs must be an integer or 'auto', got {epochs!r}."
            ) from error
        if str(epochs).strip() not in {str(resolved), f"+{resolved}"}:
            raise ConfigValidationError(
                f"epochs must be a whole number, got {epochs!r} "
                f"(would truncate to {resolved})."
            )
        return resolved
    args_cls = method_cls.method_args_cls()
    recipe_steps = getattr(args_cls, "auto_steps", None)
    if recipe_steps is not None:
        return epochs_from_steps(recipe_steps, batch_size, dataset_size)
    recipe_epochs = getattr(args_cls, "auto_epochs", None)
    if recipe_epochs is None:
        raise ConfigValidationError(
            f"{args_cls.__name__} defines neither 'auto_steps' nor "
            f"'auto_epochs'; set epochs explicitly."
        )
    return int(recipe_epochs)


def get_scaling_info(dataset_size: int, epochs: int) -> ScalingInfo:
    return ScalingInfo(dataset_size=dataset_size, epochs=epochs)


def get_optimizer_args(
    method_cls: type[PretextRecipe],
    optim: str,
    optim_args: dict | Any | None,
    wrapped_model: Any,
) -> Any:
    args_cls = method_cls.optimizer_args_cls(optim)
    args = pydantic_model_validate(args_cls, dict(optim_args or {}))
    args.resolve_auto_fields(wrapped_model=wrapped_model)
    assert_config_resolved(args)
    return args


def get_method_args(
    method_cls: type[PretextRecipe],
    method_args: dict | Any | None,
    scaling: ScalingInfo,
    optimizer_args: Any,
    wrapped_model: Any,
) -> Any:
    args_cls = method_cls.method_args_cls()
    args = pydantic_model_validate(args_cls, dict(method_args or {}))
    args.resolve_auto_fields(
        scaling=scaling,
        optimizer_args=optimizer_args,
        wrapped_model=wrapped_model,
    )
    assert_config_resolved(args)
    return args


def get_method(
    method_cls: type[PretextRecipe],
    method_args: Any,
    optimizer_args: Any,
    embedding_model: Any,
    global_batch_size: int,
    num_input_channels: int,
) -> PretextRecipe:
    return method_cls(
        method_args=method_args,
        optimizer_args=optimizer_args,
        embedding_model=embedding_model,
        global_batch_size=global_batch_size,
        num_input_channels=num_input_channels,
    )


def get_dataloader(
    dataset,
    batch_size_per_device: int,
    num_workers: int,
    collate_fn: Any = None,
    seed: int | None = None,
) -> DataLoader:
    timeout = (
        _dataloader_timeout() if num_workers > 0 else 0.0
    )
    worker_init_fn = None
    generator = None
    if seed is not None:
        from autovision.engine.engine import (
            dataloader_seed_rank,
            make_dataloader_generator,
            seed_worker,
        )

        worker_init_fn = seed_worker
        # H1-4: offset the stream per rank so DDP ranks do not replay the
        # same shuffle sequence.
        generator = make_dataloader_generator(seed, rank=dataloader_seed_rank())
    return DataLoader(
        dataset,
        batch_size=batch_size_per_device,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        generator=generator,
        timeout=timeout,
    )


def _dataloader_timeout() -> float:
    from autovision.runtime.env import Env

    return Env.DATALOADER_TIMEOUT_SEC.value
