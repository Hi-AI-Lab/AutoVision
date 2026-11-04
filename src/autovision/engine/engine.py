# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Training engine: step loop, precision, checkpointing, optional DDP.

Deliberately thin and framework-free. The loop implements exactly what the
methods need: forward/backward per step, gradient clipping, LR scheduler
steps, NaN guards, atomic checkpointing per epoch, and resume. Multi-device
training works under ``torchrun`` (environment-initialized process group);
single-process runs need no setup.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from autovision.engine.amp_check import resolve_amp_precision
from autovision.engine.callbacks import CallbackRegistry
from autovision.observability.otel import end_span, otel_span, start_span
from autovision.runtime.accelerator import (
    ddp_backend_for,
    npu_available,
)
from autovision.runtime.checkpoint import Checkpoint, CheckpointMetadata
from autovision.runtime.distributed import get_local_rank
from autovision.runtime.env import Env

__all__ = ["Engine", "FitResult"]

logger = logging.getLogger("autovision")


def _capture_rng_state() -> dict[str, Any]:
    """Capture Python/NumPy/Torch RNG states in a ``weights_only``-safe form."""
    # Runtime value is the legacy 5-tuple; NumPy's stubs type it as a dict.
    numpy_state: Any = np.random.get_state()
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": list(random.getstate()),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ],
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Any) -> None:
    """Restore a state captured by :func:`_capture_rng_state` (best effort)."""
    if not isinstance(state, dict):
        return
    try:
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "python" in state:
            python_state = state["python"]
            random.setstate((python_state[0], tuple(python_state[1]), python_state[2]))
        if "numpy" in state:
            numpy_state = state["numpy"]
            np.random.set_state(
                (
                    numpy_state[0],
                    np.asarray(numpy_state[1], dtype=np.uint32),
                    numpy_state[2],
                    numpy_state[3],
                    numpy_state[4],
                )
            )
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except (RuntimeError, TypeError, ValueError) as error:
        # A checkpoint moved across CUDA device counts should not fail the
        # resume outright; reproducibility is degraded, not training.
        logger.warning("Could not restore RNG state from checkpoint: %s", error)


def _validate_ddp_sampler(dataloader: DataLoader) -> None:
    """Reject custom samplers that DistributedSampler would silently replace."""
    sampler = getattr(dataloader, "sampler", None)
    if isinstance(sampler, (RandomSampler, SequentialSampler)):
        return
    raise ValueError(
        "Custom samplers (hard-example/long-tail/weighted) are not supported "
        "with WORLD_SIZE > 1: the engine would replace them with "
        "DistributedSampler and silently drop the weights. Run single-device "
        f"or sample outside the loader (got {type(sampler).__name__})."
    )


def _assert_finite_gradients(optimizer: Any, clip_norm: float | None) -> None:
    """Reject non-finite gradients before ``optimizer.step()``.

    A finite loss can still produce Inf/NaN gradients (AMP overflow, a bad
    normalization layer, or a corrupted batch). When the model clips, the
    pre-clip norm is already computed; otherwise one extra inf-norm reduction
    is paid to keep the check unconditional.

    Raises :class:`_TrainingCorruptedError` so callers with NaN recovery
    (design D32) can roll back instead of losing the run.
    """
    if not _grad_norm_is_finite(optimizer, clip_norm):
        raise _TrainingCorruptedError(
            "Non-finite gradient norm after a finite loss; check AMP "
            "scaling, normalization, or the batch. Lower the learning rate "
            "or enable gradient clipping."
        )


def _grad_norm_is_finite(optimizer: Any, clip_norm: float | None) -> bool:
    """True when the (possibly pre-computed) gradient norm is finite.

    Without an explicit ``clip_norm`` one extra inf-norm reduction is paid;
    the norm is returned by :func:`clip_grad_norm_` anyway, so nothing is
    synchronized twice.
    """
    norm = clip_norm
    if norm is None:
        params = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        norm = float(torch.nn.utils.clip_grad_norm_(params, float("inf")))
    return math.isfinite(float(norm))


#: D21: how many automatic batch halvings one run may attempt.
_OOM_MAX_ATTEMPTS = 3
#: D20: rolling retention count for ``epoch{N}.ckpt`` archives.
_EPOCH_ARCHIVE_KEEP = 3
#: D32: consecutive NaN/Inf recoveries before the run is declared dead.
_NAN_RECOVERY_MAX_ATTEMPTS = 3
#: D34: clear accelerator caches only above this reserved-memory fraction.
_CACHE_CLEAR_RESERVED_FRACTION = 0.5


class _TrainingCorruptedError(RuntimeError):
    """A step/epoch produced non-finite loss or gradients (design D32).

    Raised instead of a bare ``RuntimeError`` so an engine with a valid
    ``last.ckpt`` can roll back and retry; after
    :data:`_NAN_RECOVERY_MAX_ATTEMPTS` consecutive failures the run is
    declared dead with the original signal chained.
    """


def _assert_state_dict_finite(state: Any, path: Path) -> None:
    """Refuse to load a checkpoint whose numeric payload contains NaN/Inf.

    Recovery re-enters training from this state — a poisoned checkpoint
    would reinstall the very NaN the recovery is escaping. The walk covers
    the whole payload, not just the top level: the ``state_dict`` weights
    plus the ``engine_state`` a rollback restores alongside them (optimizer
    moments, scaler scale, EMA sidecar, best-metric baseline). Only
    floating tensors and Python floats are checked (int/bool/RNG-byte
    states cannot be non-finite).
    """
    if not isinstance(state, dict):
        return
    poisoned: list[str] = []

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else str(key), value)
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(f"{prefix}[{index}]", value)
        elif torch.is_tensor(node) and node.is_floating_point():
            if not bool(torch.isfinite(node).all()):
                poisoned.append(prefix)
        elif isinstance(node, float) and not math.isfinite(node):
            poisoned.append(prefix)

    walk("", state)
    if poisoned:
        raise RuntimeError(
            f"Checkpoint '{path}' is corrupted with NaN/Inf "
            f"(first offending keys: {poisoned[:5]}); recovery cannot "
            "proceed from it."
        )


def _sync_corruption_verdict(
    device: torch.device, world_size: int, corrupted: bool
) -> bool:
    """Broadcast a local corruption detection to every rank (design D32).

    Every rank posts a 1 (clean) or 0 (corrupted) int32 and the MIN is
    reduced, so a NaN on any single rank flips all ranks to the recovery
    path from the same checkpoint. Call only from points where every rank
    participates with the same collective count.
    """
    if world_size <= 1 or not torch.distributed.is_initialized():
        return bool(corrupted)
    flag = torch.tensor(
        [0 if corrupted else 1], device=device, dtype=torch.int32
    )
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return int(flag.item()) == 0


def _maybe_clear_accelerator_cache(
    device: torch.device, *, threshold: float = _CACHE_CLEAR_RESERVED_FRACTION
) -> None:
    """Release cached accelerator memory only when it is actually hoarding.

    An unconditional ``empty_cache`` between train and validation forces a
    full cudaFree + re-cudaMalloc round trip on every boundary even when
    the cache holds a few MB. Clear when the caching allocator's reserved
    share of total memory exceeds ``threshold``; MPS clears unconditionally
    (known allocator leak), CPU is a no-op, and a failed occupancy query
    degrades to always-clear (the historical behavior) rather than risking
    an OOM'd validation forward (design D34).
    """
    if device.type == "mps":
        torch.mps.empty_cache()  # type: ignore[attr-defined]
        return
    if device.type not in ("cuda", "npu"):
        return
    try:
        if device.type == "cuda":
            reserved = torch.cuda.memory_reserved(device)
            total = torch.cuda.get_device_properties(device).total_memory
        else:
            reserved = torch.npu.memory_reserved(device)
            total = torch.npu.get_device_properties(device).total_memory
        hoarding = total <= 0 or reserved / total > threshold
    except Exception:  # noqa: BLE001 - introspection is best-effort
        hoarding = True
    if not hoarding:
        return
    if device.type == "cuda":
        torch.cuda.empty_cache()
    else:
        torch.npu.empty_cache()


def _repair_ema_if_present(model: nn.Module, *, allow: bool = True) -> int:
    """Resync a poisoned EMA from live weights before saving (design D33).

    The EMA is a running average: one non-finite update poisons it for the
    rest of the run, while the live model may have already recovered. A
    non-finite EMA silently enters every checkpoint written afterwards.
    Returns the number of repaired tensors. ``allow=False`` skips the pass
    (FSDP shards the live model, so a per-tensor comparison is impossible).

    The EMA copy wraps the TASK model (``TrainModel.model``), so the live
    comparison source is resolved through ``get_task_model()`` when the
    host provides it — comparing against the host's own state_dict would
    key-mismatch (``model.*`` / ``ema_model.*`` prefixes) and silently
    repair nothing.
    """
    if not allow:
        return 0
    ema = getattr(model, "ema_model", None)
    repair = getattr(ema, "repair_from", None)
    if repair is None:
        return 0
    get_task_model = getattr(model, "get_task_model", None)
    source = (
        get_task_model()
        if callable(get_task_model)
        else model
    )
    return int(repair(source))


def _is_rank_zero_internal() -> bool:
    from autovision.runtime.distributed import is_global_rank_zero

    return is_global_rank_zero()


def _set_accelerator_device(device: torch.device) -> None:
    """Make ``device`` the current device for this process.

    DDP requires every rank to select its local device before the process
    group is created / the model is placed: without it, APIs that allocate
    on the *current* device (and some NCCL initialization paths) use
    ``cuda:0`` on every rank.
    """
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    elif device.type == "npu" and device.index is not None:
        torch.npu.set_device(device)


def _log_metrics_safely(logger_fn: Any, metrics: dict[str, Any], step: int) -> None:
    """Call an experiment logger without letting it abort training.

    W&B/MLflow/TensorBoard are I/O-backed side channels: a network or disk
    failure must degrade observability, not kill a long-running fit (the
    callback path already follows this contract).
    """
    if logger_fn is None:
        return
    try:
        logger_fn(metrics, step)
    except Exception:  # noqa: BLE001 - experiment loggers are non-fatal
        logger.warning(
            "logger_fn failed at step %d; continuing training.",
            step, exc_info=True,
        )


def _rank_zero_write_barrier(
    device: torch.device, world_size: int, success: bool
) -> None:
    """Propagate a rank-zero artifact-write failure to every rank.

    Without this, rank zero exits while the other ranks enter the next
    collective and hang until the process-group timeout. The barrier runs on
    every rank even when rank zero failed (the flag is all-reduced).
    """
    if world_size <= 1 or not torch.distributed.is_initialized():
        return
    flag = torch.tensor(
        [1 if success else 0], device=device, dtype=torch.int32
    )
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    if int(flag.item()) == 0:
        raise RuntimeError(
            "rank zero failed to write a checkpoint/export artifact; "
            "aborting all ranks (see rank-zero logs)."
        )


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch globally.

    Dataloader worker processes re-seed from the global torch RNG per worker
    (PyTorch default ``base_seed`` mechanism), so worker randomness follows
    the global seed.

    ``AUTOVISION_DETERMINISTIC=1`` additionally asks PyTorch for deterministic
    kernels. This is opt-in: some ops have no deterministic implementation and
    will raise at runtime, and deterministic kernels can be noticeably slower.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if Env.DETERMINISTIC.value:
        # cuBLAS requires this workspace shape before CUDA is initialized;
        # setting it late is harmless but may not affect already-created
        # handles.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def enable_cuda_math_modes(device: torch.device) -> None:
    """TF32 tensor cores + cudnn autotuning for non-deterministic CUDA runs.

    Fixed-size detector inputs (scale jitter enumerates a handful of
    shapes, then the cudnn cache is warm) leave conv-heavy families
    launch-bound; the autotuner picks faster conv algos per shape.
    ``AUTOVISION_DETERMINISTIC=1`` keeps the deterministic/safe defaults.
    """
    if device.type != "cuda" or Env.DETERMINISTIC.value:
        return
    # Ampere+ tensor cores for the fp32 remainder (EMA, optimizer,
    # non-autocast ops); autocast-covered bf16 ops are unaffected.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _build_grad_scaler(precision: str, device_type: str) -> Any | None:
    """Loss scaler for ``16-mixed``; bf16/fp32 do not need one.

    fp16 autocast without loss scaling silently underflows small gradients
    (common on T4/V100-class GPUs, which do not have bf16). torch's
    ``GradScaler`` is used when the device backend supports it; NPU builds
    that lack the API fall back with a warning rather than failing the run.
    """
    if precision != "16-mixed" or device_type not in ("cuda", "npu"):
        return None
    try:
        return torch.amp.GradScaler(device_type)
    except (AttributeError, TypeError, RuntimeError):
        if device_type == "cuda":
            return torch.cuda.amp.GradScaler()
        logger.warning(
            "GradScaler is unavailable for device '%s'; fp16 runs without "
            "loss scaling.", device_type,
        )
        return None


#: Substrings that identify an allocator OOM (message matching catches the
#: driver-level variants torch.OutOfMemoryError does not cover).
_OOM_MESSAGE_MARKERS: tuple[str, ...] = (
    "out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_internal_error",
    "unable to find an engine",
)
#: D21: how many automatic batch halvings one run may attempt.
_OOM_MAX_ATTEMPTS = 3
#: D20: rolling retention count for ``epoch{N}.ckpt`` archives.
_EPOCH_ARCHIVE_KEEP = 3


def _is_oom_error(error: BaseException) -> bool:
    """True for CUDA/NPU allocator exhaustion (typed or message-matched)."""
    if isinstance(error, torch.OutOfMemoryError):
        return True
    if isinstance(error, RuntimeError):
        message = str(error).lower()
        return any(marker in message for marker in _OOM_MESSAGE_MARKERS)
    return False


def make_dataloader_generator(seed: int, rank: int = 0) -> torch.Generator:
    """Per-dataloader torch generator seeded from the run seed (D23a).

    ``rank`` offsets the stream so DDP ranks shuffle differently; callers
    without a rank context use the default. The offset must live here and
    not at call sites, so every builder path shares one derivation.
    """
    generator = torch.Generator()
    generator.manual_seed((int(seed) + int(rank)) % (2**63))
    return generator


def seed_worker(worker_id: int) -> None:  # pragma: no cover - runs in workers
    """``worker_init_fn`` that derives numpy/random seeds per worker (D23a).

    torch seeds its own RNG in every worker from the dataloader generator,
    but numpy/random would otherwise be inherited FORK-identical across
    workers — every worker would apply the same numpy-driven augmentation
    sequence. This is the DataLoader-side counterpart of ``seed_everything``.
    """
    base = torch.initial_seed() % (2**32)
    np.random.seed(base)
    random.seed(base)


def _sanitize_state_dict_for_serialization(state: dict) -> dict:
    """Serialization hygiene (D20): clamp fp16 overflow, normalize NCHW.

    Returns a dict whose fp16 tensors with non-finite values are replaced
    by ``nan_to_num`` clones (live weights are never mutated — a NaN EMA
    must not be silently fixed in place, only kept out of the file) and
    whose 4-d float tensors are standard-contiguous (channels-last strides
    would break ``.view()`` in downstream fuse/export paths).
    """
    out: dict = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        if value.dtype == torch.float16 and not bool(torch.isfinite(value).all()):
            logger.warning(
                "Clamping non-finite fp16 tensor '%s' before checkpoint "
                "serialization.", key,
            )
            out[key] = torch.nan_to_num(value.detach()).contiguous()
            continue
        if value.ndim == 4 and value.is_floating_point() and not value.is_contiguous():
            out[key] = value.contiguous()
            continue
        out[key] = value
    return out


def _prune_epoch_archives(checkpoints_dir: Path, keep: int) -> None:
    """Keep only the newest ``keep`` ``epoch<N>.ckpt`` archives (D20)."""
    archives = []
    for path in checkpoints_dir.glob("epoch*.ckpt"):
        digits = path.stem.removeprefix("epoch")
        if digits.isdigit():
            archives.append((int(digits), path))
    archives.sort(reverse=True)
    for _, path in archives[keep:]:
        try:
            path.unlink()
        except OSError:
            logger.debug("Could not remove stale archive '%s'.", path)


def _clone_loader_with_batch_size(loader: DataLoader, batch_size: int) -> DataLoader:
    """Re-materialize a DataLoader with a smaller per-device batch (D21).

    The sampler instance is carried over (epoch continuity), as are the
    worker/collate/pinning settings. Used only on the single-process OOM
    recovery path — under DDP a mid-run batch change must be synchronized
    across ranks, which this helper deliberately does not attempt.
    """
    num_workers = loader.num_workers
    return DataLoader(
        loader.dataset,
        batch_size=batch_size,
        sampler=loader.sampler,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=loader.worker_init_fn,
        generator=loader.generator,
        collate_fn=loader.collate_fn,
        timeout=loader.timeout,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        prefetch_factor=loader.prefetch_factor if num_workers > 0 else None,
        persistent_workers=loader.persistent_workers if num_workers > 0 else False,
    )


class _TrainWindow:
    """On-device running sums of the training ``log_dict`` between log points.

    Sums stay as 0-dim tensors on the source device: ``float()`` would sync
    the GPU once per loss component per step, which is exactly what the
    deferred logging in the training wrapper is designed to avoid. One sync
    per component per LOG, not per step.
    """

    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.count = 0
        self.t0 = time.perf_counter()

    def update(self, log_dict: dict) -> None:
        for key, value in log_dict.items():
            if not torch.is_tensor(value):
                continue  # non-tensor stats are logged as-is at log time
            current = self.sums.get(key)
            self.sums[key] = value.detach() + current if current is not None else value.detach()
        self.count += 1

    def means(self) -> dict[str, float]:
        divisor = max(self.count, 1)
        return {key: float(value) / divisor for key, value in self.sums.items()}

    def seconds_per_step(self) -> float:
        return (time.perf_counter() - self.t0) / max(self.count, 1)

    def reset(self) -> None:
        self.sums = {}
        self.count = 0
        self.t0 = time.perf_counter()


def _log_fit_banner(
    train_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    steps: int,
    gradient_accumulation_steps: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    precision: str,
    world_size: int,
) -> None:
    """Startup summary in the reference detection-log style: model size,
    optimizer groups, LR windows, EMA, and data counts — everything needed
    to sanity-check a run without reading the config."""
    parameters = list(train_model.parameters())
    total = sum(p.numel() for p in parameters)
    trainable = sum(p.numel() for p in parameters if p.requires_grad)
    logger.info(
        "Model: %s params (%s trainable, %s frozen)",
        f"{total:,}", f"{trainable:,}", f"{total - trainable:,}",
    )
    from autovision.runtime.model_info import estimate_gflops

    gflops = estimate_gflops(train_model)
    if gflops is not None:
        logger.info("Model: %.2f GFLOPs (thop, batch-1 forward)", gflops)
    group_summary = " | ".join(
        f"{group.get('name', f'group{index}')} lr {group['lr']:.2e}"
        for index, group in enumerate(optimizer.param_groups)
    )
    logger.info("Optimizer: %s", group_summary)
    warmup = getattr(scheduler, "warmup_steps", None)
    flat = getattr(scheduler, "flat_steps", None)
    no_aug = getattr(scheduler, "no_aug_steps", None)
    total_steps = getattr(scheduler, "total_steps", None)
    milestones = getattr(scheduler, "milestones", None)
    if (
        warmup is not None and flat is not None
        and no_aug is not None and total_steps is not None
    ):
        # Name only the non-empty phases: a 2-step smoke run would otherwise
        # log "warmup 2 → flat until 2 → cosine until 2 → tail to 2", and
        # the tail holds min_factor × peak, not the peak LR itself.
        cosine_end = total_steps - no_aug
        phases = []
        if warmup > 0:
            phases.append(f"warmup {warmup}")
        if flat > 0:
            phases.append(f"flat until {warmup + flat}")
        if cosine_end > warmup + flat:
            phases.append(f"cosine until {cosine_end}")
        min_factor = getattr(scheduler, "min_factor", None)
        if no_aug > 0:
            floor = f" at ×{min_factor}" if min_factor is not None else ""
            phases.append(f"tail{floor} until {total_steps}")
        summary = " → ".join(phases) if phases else "constant"
        logger.info("LR schedule: %s (steps %d)", summary, total_steps)
    elif (
        warmup is not None and total_steps is not None and milestones is not None
    ):
        logger.info(
            "LR schedule: warmup %d → constant with ×%s drops at steps %s "
            "(total %d)",
            warmup,
            getattr(scheduler, "gamma", "?"),
            list(milestones),
            total_steps,
        )
    ema_model = getattr(train_model, "ema_model", None)
    if ema_model is not None:
        logger.info(
            "EMA: decay %s, warmup %s steps (validation uses EMA weights)",
            getattr(ema_model, "decay", "?"),
            getattr(ema_model, "warmup_steps", "?"),
        )
    max_dets = getattr(train_model, "_max_dets", None)
    if max_dets is not None:
        logger.info("Eval: COCO bbox mAP at maxDets=%d", max_dets)
    logger.info(
        "Data: train %s samples (%d batches × accum %d)%s | "
        "precision %s | world_size %d | steps %d",
        f"{len(train_dataloader.dataset):,}",
        len(train_dataloader),
        gradient_accumulation_steps,
        (
            f", val {len(val_dataloader.dataset):,} samples"
            if val_dataloader is not None else ""
        ),
        precision,
        world_size,
        steps,
    )


def resolve_device(devices: int | str = "auto") -> torch.device:
    """Resolve the torch device from the devices specification.

    ``"auto"`` picks CUDA > Ascend NPU (torch_npu) > MPS > CPU (honoring
    LOCAL_RANK under torchrun); explicit specs (``"cpu"``, ``"cuda:1"``,
    ``"npu:0"``, ``int`` indices) pass through. ``"-1"`` / ``"cuda:-1"``
    are idle-GPU placeholders (design doc D18, see
    :func:`autovision.runtime.autodevice.resolve_device_spec`) resolving
    to the most-idle visible card, degrading to sequential picks when
    NVML is unavailable and raising when the host cannot satisfy the
    placeholders. NPU is software-side support only
    (design §12.4) — real-hardware validation is M4+.
    """
    if isinstance(devices, int):
        if torch.cuda.is_available():
            return torch.device("cuda", devices % torch.cuda.device_count())
        if npu_available():
            return torch.device("npu", devices % torch.npu.device_count())
        return torch.device("cpu")
    if isinstance(devices, str) and devices != "auto":
        from autovision.runtime.autodevice import resolve_device_spec

        spec = resolve_device_spec(devices)
        tokens = [token.strip() for token in spec.split(",") if token.strip()]
        if len(tokens) > 1:
            logger.warning(
                "Device spec '%s' lists %d devices but this is a "
                "single-device context: resolving to the primary '%s'. "
                "Multi-process runs must expand the spec (and constrain "
                "CUDA_VISIBLE_DEVICES) before launching.",
                devices,
                len(tokens),
                tokens[0],
            )
        return torch.device(tokens[0])
    if torch.cuda.is_available():
        local_rank = get_local_rank()
        if local_rank is not None and torch.cuda.device_count() > 0:
            return torch.device("cuda", local_rank % torch.cuda.device_count())
        return torch.device("cuda")
    if npu_available():
        local_rank = get_local_rank()
        if local_rank is not None and torch.npu.device_count() > 0:
            return torch.device("npu", local_rank % torch.npu.device_count())
        return torch.device("npu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class FitResult:
    out_dir: Path
    epochs_run: int
    global_step: int
    checkpoint_path: Path | None = None
    exported_model_path: Path | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class Engine:
    """Drives the training loop for a :class:`~autovision.pretext.PretextRecipe`."""

    def fit(
        self,
        method: nn.Module,
        dataloader: DataLoader,
        *,
        out_dir: Path,
        max_epochs: int,
        devices: int | str = "auto",
        precision: str = "auto",
        seed: int = 0,
        checkpoint: Path | None = None,
        resume_interrupted: bool = False,
        logger_fn: Any = None,
        log_every_n_steps: int = 50,
        normalize_args: dict | None = None,
        amp_check: bool = True,
        save_period_num_epochs: int = 0,
        callbacks: list[Any] | None = None,
    ) -> FitResult:
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
        if save_period_num_epochs < 0:
            raise ValueError(
                f"save_period_num_epochs must be >= 0, got {save_period_num_epochs}"
            )
        if len(dataloader) == 0:
            raise ValueError(
                "Training dataloader is empty; check the dataset root, filters, "
                "and batch_size (a zero-length loader would silently train for "
                "zero steps)."
            )
        seed_everything(seed)
        device = resolve_device(devices)
        _set_accelerator_device(device)
        method.to(device)

        # ── distributed setup (torchrun) ──
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        ddp_method: nn.Module = method
        sampler = None
        if world_size > 1:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend=ddp_backend_for(device.type),
                    timeout=timedelta(seconds=Env.DDP_TIMEOUT_SEC.value),
                )
            ddp_method = nn.parallel.DistributedDataParallel(
                method,
                device_ids=[device.index] if device.type in ("cuda", "npu") else None,
                # The per-rank teacher queue must not be broadcast from rank 0.
                broadcast_buffers=False,
                # Queue-based methods skip some heads while the queue fills
                # (and EMA/teacher branches may be step-dependent); without
                # this, DDP aborts with "expected to have finished reduction
                # in the prior iteration". Baseline uses
                # ddp_find_unused_parameters_true for the same reason.
                # Opt out (stable step shapes) via AUTOVISION_DDP_FIND_UNUSED=0.
                find_unused_parameters=Env.DDP_FIND_UNUSED.value,
            )
            _validate_ddp_sampler(dataloader)
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataloader.dataset, shuffle=True, drop_last=True
            )

        # Total steps must match the *distributed* loader length so the
        # cosine schedule reaches its end value exactly.
        if sampler is not None:
            steps_per_epoch = (
                len(dataloader.dataset) // world_size
            ) // dataloader.batch_size
        else:
            steps_per_epoch = len(dataloader)
        # Rank gate frozen while the process group is alive: the fit
        # finally-block destroys the group on every exit path (success
        # included), and a post-training get_rank() on the dead group
        # raises "Default process group has not been initialized" (the
        # torchrun e2e regression).
        is_rank_zero = world_size == 1 or torch.distributed.get_rank() == 0
        optimizer, scheduler = method.configure_optimizers(
            total_steps=max_epochs * steps_per_epoch, num_epochs=max_epochs
        )
        method.total_steps = max_epochs * steps_per_epoch

        # AMP gates (D16): device support first, then the numeric probe —
        # both fail closed to fp32 and never abort the run.
        precision = resolve_amp_precision(
            method, device=device, precision=precision, amp_check=amp_check
        )
        autocast_dtype = {
            "bf16-mixed": torch.bfloat16,
            "16-mixed": torch.float16,
        }.get(precision)
        scaler = _build_grad_scaler(precision, device.type)

        # ── checkpoint resolution ──
        start_epoch = 0
        global_step = 0
        if checkpoint is not None and resume_interrupted:
            raise ValueError(
                "'checkpoint' and 'resume_interrupted' cannot be set at the same time."
            )
        if resume_interrupted:
            last_path = out_dir / "checkpoints" / "last.ckpt"
            if last_path.exists():
                start_epoch, global_step = self._resume(
                    last_path, method, optimizer, scheduler, device, scaler=scaler
                )
            else:
                logger.info("No last.ckpt found; starting from scratch.")
        elif checkpoint is not None:
            self._load_weights(checkpoint, method, device)

        epochs_run = 0
        run_metrics: dict[str, float] = {}
        callback_registry = CallbackRegistry()
        callback_registry.adapt(callbacks or [])
        self.callbacks = callback_registry
        if is_rank_zero:
            callback_registry.run("on_train_start", max_epochs, out_dir)
        fit_span = start_span(
            "pretrain.fit",
            {
                "device": str(device),
                "precision": precision,
                "world_size": world_size,
                "steps_per_epoch": steps_per_epoch,
                "max_epochs": max_epochs,
            },
        )
        try:
            # D32: while (not for) so NaN recovery can rewind `epoch` and
            # re-run the poisoned epoch from the last checkpoint.
            epoch = start_epoch
            nan_attempts = 0
            while epoch < max_epochs:
                with otel_span("engine.epoch", {"epoch": epoch}):
                    if sampler is not None:
                        sampler.set_epoch(epoch)
                    epoch_loader = (
                        DataLoader(
                            dataloader.dataset,
                            batch_size=dataloader.batch_size,
                            sampler=sampler,
                            shuffle=False,
                            num_workers=dataloader.num_workers,
                            drop_last=True,
                            collate_fn=dataloader.collate_fn,
                            timeout=dataloader.timeout,
                            pin_memory=dataloader.pin_memory,
                        )
                        if sampler is not None
                        else dataloader
                    )
                    method.train()
                    epoch_corrupted = False
                    for batch_idx, batch in enumerate(epoch_loader):
                        method.global_step = global_step
                        batch = self._to_device(batch, device)
                        autocast = (
                            torch.autocast(device_type=device.type, dtype=autocast_dtype)
                            if autocast_dtype is not None
                            else nullcontext()
                        )
                        with autocast, otel_span(
                            "engine.step", {"step": global_step, "batch": batch_idx}
                        ):
                            # PretextRecipe.forward == training_step; DDP wraps forward.
                            loss = ddp_method(batch, batch_idx)
                        if not torch.isfinite(loss):
                            epoch_corrupted = True
                            if world_size == 1:
                                # No collectives to keep aligned: skip the
                                # poisoned update and the rest of the epoch.
                                break
                            # DDP: fall through and finish the batch (and the
                            # epoch) normally so every rank posts the same
                            # collective sequence; the epoch-end verdict
                            # barrier decides recovery and the rollback
                            # discards these updates.
                        optimizer.zero_grad(set_to_none=True)
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            # Gradients are still scaled here; unscale before
                            # clipping/inspection so the norm is meaningful.
                            scaler.unscale_(optimizer)
                        else:
                            loss.backward()
                        clip_norm = method.clip_gradients(optimizer)
                        # With a GradScaler, inf/NaN gradients are the
                        # expected overflow signal: scaler.step() skips the
                        # update and backs off the scale. Raising here would
                        # defeat AMP's recovery path.
                        if scaler is None and not epoch_corrupted:
                            if not _grad_norm_is_finite(optimizer, clip_norm):
                                epoch_corrupted = True
                                if world_size == 1:
                                    break
                        method.on_before_optimizer_step(optimizer)
                        if scaler is not None:
                            scale_before = scaler.get_scale()
                            scaler.step(optimizer)
                            scaler.update()
                            # GradScaler skips the step on inf/nan (halving
                            # the scale); advancing the scheduler then would
                            # silently consume a schedule step.
                            stepped = scaler.get_scale() >= scale_before
                        else:
                            optimizer.step()
                            stepped = True
                        if stepped:
                            scheduler.step()
                        global_step += 1

                        if logger_fn is not None and (
                            world_size == 1 or torch.distributed.get_rank() == 0
                        ):
                            if global_step % log_every_n_steps == 0:
                                metrics = {"train_loss": float(loss.detach())}
                                log_dict = method.consume_log_dict()
                                if log_dict:
                                    metrics.update(log_dict)
                                _log_metrics_safely(logger_fn, metrics, global_step)
                        else:
                            method.consume_log_dict()  # keep step state clean on all ranks
                    # D32 verdict barrier: UNCONDITIONAL on the local flag —
                    # a rank whose epoch looked clean must still post the
                    # verdict, or a rank-local NaN (the exact scenario D32
                    # exists for) leaves the ranks posting mismatched
                    # collectives: the poisoned rank's verdict pairs with a
                    # clean rank's write barrier and every rank dies with a
                    # misleading barrier error. Single-process runs
                    # short-circuit inside the helper.
                    local_corrupted = epoch_corrupted
                    epoch_corrupted = _sync_corruption_verdict(
                        device, world_size, epoch_corrupted
                    )
                    if epoch_corrupted:
                        nan_attempts += 1
                        if nan_attempts > _NAN_RECOVERY_MAX_ATTEMPTS:
                            raise RuntimeError(
                                "Training failed: non-finite loss persisted "
                                f"across {_NAN_RECOVERY_MAX_ATTEMPTS} "
                                f"rollbacks to {out_dir / 'checkpoints' / 'last.ckpt'} "
                                f"(last detection at epoch {epoch}, step "
                                f"{global_step}); check the data pipeline or "
                                "lower the learning rate."
                            )
                        last_path = out_dir / "checkpoints" / "last.ckpt"
                        if not last_path.exists():
                            raise RuntimeError(
                                f"Non-finite loss at epoch {epoch} (step "
                                f"{global_step}) and no last.ckpt exists to "
                                "roll back to; fix the data or lower the "
                                "learning rate."
                            )
                        logger.warning(
                            "Non-finite loss at epoch %d (%s; step %d); "
                            "recovery attempt %d/%d — rolling back to %s "
                            "and re-running the epoch.",
                            epoch,
                            "detected on this rank" if local_corrupted
                            else "reported by a peer rank",
                            global_step, nan_attempts,
                            _NAN_RECOVERY_MAX_ATTEMPTS, last_path,
                        )
                        resumed_epoch, resumed_step = self._resume(
                            last_path,
                            method,
                            optimizer,
                            scheduler,
                            device,
                            scaler=scaler,
                            validate_finite=True,
                        )
                        epoch = resumed_epoch
                        global_step = resumed_step
                        continue
                    # A clean epoch restores the recovery budget (upstream
                    # resets per clean epoch; scattered transients each get
                    # their full 3 attempts).
                    nan_attempts = 0
                    epochs_run += 1

                # In-training kNN probe (design §7.5), rank zero only. Under
                # DDP the other ranks block in the next allreduce while the
                # probe runs — keep every_n_epochs high enough that the probe
                # cost stays well below the collective watchdog timeout.
                knn_runner = getattr(method, "knn_eval_runner", None)
                if (
                    knn_runner is not None
                    and knn_runner.should_run(epoch)
                    and (world_size == 1 or torch.distributed.get_rank() == 0)
                ):
                    with otel_span("engine.validation", {"epoch": epoch}):
                        knn_metrics = knn_runner.run()
                    run_metrics.update(knn_metrics)
                    _log_metrics_safely(logger_fn, knn_metrics, global_step)

                # Checkpoint at the end of every epoch (rank zero only). The
                # barrier makes a disk/permission failure fatal for every
                # rank instead of leaving peers in the next collective.
                write_ok = True
                write_error: Exception | None = None
                archive_epoch = (
                    save_period_num_epochs > 0
                    and (epoch + 1) % save_period_num_epochs == 0
                )
                if world_size == 1 or torch.distributed.get_rank() == 0:
                    try:
                        with otel_span("engine.checkpoint", {"epoch": epoch}):
                            if archive_epoch:
                                # D20: serialize ONCE, write last + epochN
                                # from the same payload. The metadata epoch
                                # stays 0-BASED (the resume convention — a
                                # 1-based value here would make a resume or
                                # D32 rollback silently skip one epoch); only
                                # the epoch{N}.ckpt FILENAME is 1-based.
                                archive = self._build_checkpoint(
                                    method=method,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    epoch=epoch,
                                    global_step=global_step,
                                    normalize_args=normalize_args,
                                    scaler=scaler,
                                )
                                payload = archive.serialize()
                                checkpoints_dir = out_dir / "checkpoints"
                                archive.save_serialized(
                                    checkpoints_dir / "last.ckpt", payload
                                )
                                archive.save_serialized(
                                    checkpoints_dir / f"epoch{epoch + 1}.ckpt",
                                    payload,
                                )
                                _prune_epoch_archives(
                                    checkpoints_dir, keep=_EPOCH_ARCHIVE_KEEP
                                )
                            else:
                                self.save_checkpoint(
                                    out_dir / "checkpoints" / "last.ckpt",
                                    method=method,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    epoch=epoch,
                                    global_step=global_step,
                                    normalize_args=normalize_args,
                                    scaler=scaler,
                                )
                    except Exception as error:  # noqa: BLE001 - propagated below
                        write_ok = False
                        write_error = error
                _rank_zero_write_barrier(device, world_size, write_ok)
                if write_error is not None:
                    raise write_error
                epoch += 1
        finally:
            end_span(fit_span)
            # Never leave a hung process group behind on error paths.
            if world_size > 1 and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

        fit_result = FitResult(
            out_dir=out_dir,
            epochs_run=epochs_run,
            global_step=global_step,
            checkpoint_path=out_dir / "checkpoints" / "last.ckpt",
            metrics=run_metrics,
        )
        if is_rank_zero:
            # Same payload contract as fit_task: the result object itself.
            callback_registry.run("on_train_end", fit_result)
        return fit_result

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _to_device(batch: dict, device: torch.device) -> dict:
        out = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(device, non_blocking=True)
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                out[key] = [v.to(device, non_blocking=True) for v in value]
            else:
                out[key] = value
        return out

    @staticmethod
    def _resume(
        path: Path,
        method: nn.Module,
        optimizer,
        scheduler,
        device: torch.device,
        *,
        scaler: Any | None = None,
        validate_finite: bool = False,
    ) -> tuple[int, int]:
        ckpt = Checkpoint.from_path(path, map_location=str(device))
        if validate_finite:
            # D32: a NaN-recovery resume must not reinstall the very NaN it
            # is escaping; regular resumes keep their historical tolerance.
            # The engine_state (optimizer moments, scaler, RNG) rides beside
            # the weights and is walked by the same guard.
            _assert_state_dict_finite(ckpt.state_dict, path)
            _assert_state_dict_finite(ckpt.autovision.engine_state, path)
        raw = {"state_dict": dict(ckpt.state_dict)}
        method.on_load_checkpoint(raw)
        method.load_state_dict(raw["state_dict"])
        engine_state = ckpt.autovision.engine_state
        if "optimizer" in engine_state:
            optimizer.load_state_dict(engine_state["optimizer"])
        if "scheduler" in engine_state:
            scheduler.load_state_dict(engine_state["scheduler"])
        if scaler is not None and "scaler" in engine_state:
            scaler.load_state_dict(engine_state["scaler"])
        _restore_rng_state(engine_state.get("rng"))
        logger.info(
            "Resumed from '%s' at epoch %d (step %d).",
            path, ckpt.autovision.epoch + 1, ckpt.autovision.global_step,
        )
        return ckpt.autovision.epoch + 1, ckpt.autovision.global_step

    @staticmethod
    def _load_weights(path: Path, method: nn.Module, device: torch.device) -> None:
        ckpt = Checkpoint.from_path(path, map_location="cpu")
        raw = {"state_dict": ckpt.state_dict}
        method.on_load_checkpoint(raw)
        method.load_state_dict(raw["state_dict"])
        logger.info("Loaded weights from '%s' (training state discarded).", path)

    @staticmethod
    def _build_checkpoint(
        method: nn.Module,
        optimizer,
        scheduler,
        epoch: int,
        global_step: int,
        normalize_args: dict | None = None,
        scaler: Any | None = None,
    ) -> Checkpoint:
        """Assemble the Checkpoint object with D20 serialization hygiene."""
        # D33: a NaN EMA must not enter the file — resync from live weights
        # first (pretraining has no FSDP, so the comparison is always safe).
        _repair_ema_if_present(method)
        checkpoint = {"state_dict": method.state_dict()}
        method.on_save_checkpoint(checkpoint)
        engine_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": _capture_rng_state(),
        }
        if scaler is not None:
            engine_state["scaler"] = scaler.state_dict()
        metadata = CheckpointMetadata(
            epoch=epoch,
            global_step=global_step,
            normalize_args=normalize_args,
            engine_state=engine_state,
        )
        return Checkpoint(
            state_dict=_sanitize_state_dict_for_serialization(
                checkpoint["state_dict"]
            ),
            autovision=metadata,
        )

    @classmethod
    def save_checkpoint(
        cls,
        path: Path,
        method: nn.Module,
        optimizer,
        scheduler,
        epoch: int,
        global_step: int,
        normalize_args: dict | None = None,
        scaler: Any | None = None,
    ) -> None:
        cls._build_checkpoint(
            method=method,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            normalize_args=normalize_args,
            scaler=scaler,
        ).save(path)


@dataclass
class TaskFitResult:
    out_dir: Path
    steps_run: int
    checkpoint_path: Path | None = None
    best_checkpoint_path: Path | None = None
    exported_model_path: Path | None = None
    best_exported_model_path: Path | None = None
    best_metrics: dict[str, float] = field(default_factory=dict)
    stopped_early: bool = False
    stop_step: int | None = None


#: Fit knobs a resumed run may legitimately change without a warning
#: (schedule tuning mid-run); any other drift against the checkpoint is
#: reported loudly (design doc D17).
RESUME_OVERRIDABLE_KEYS: frozenset[str] = frozenset(
    {
        "steps",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "val_every_num_steps",
        "save_every_num_steps",
        "log_every_n_steps",
    }
)


def check_resume_fit_args(
    checkpoint_args: dict[str, Any] | None, **current: Any
) -> None:
    """Report drift between a resumed checkpoint and the current call (D17).

    The current call's values always govern (resume never silently adopts
    checkpoint hyperparameters); the check exists so a drifted resume is a
    loud event instead of a silent one: whitelisted
    :data:`RESUME_OVERRIDABLE_KEYS` differences log at info, everything
    else (precision, accumulation, ...) logs a warning with both values.
    """
    if not checkpoint_args:
        return
    for key in sorted(checkpoint_args):
        checkpoint_value = checkpoint_args[key]
        current_value = current.get(key)
        if current_value == checkpoint_value:
            continue
        if key in RESUME_OVERRIDABLE_KEYS:
            logger.info(
                "Resuming with %s=%s (checkpoint was trained with %s).",
                key,
                current_value,
                checkpoint_value,
            )
        else:
            logger.warning(
                "Resume drift: %s=%s but the resumed checkpoint was trained "
                "with %s — the current call's value governs; verify this "
                "is intended.",
                key,
                current_value,
                checkpoint_value,
            )


class _InfiniteCycleIterator:
    """Cycles a dataloader forever, supporting mid-training re-creation.

    With a (distributed) sampler, every restart advances the sampler epoch so
    the shuffle changes across cycles — otherwise each epoch would repeat the
    exact same data order and partition.
    """

    def __init__(self, dataloader: DataLoader, sampler: Any = None) -> None:
        self.dataloader = dataloader
        self.sampler = sampler
        self._epoch = 0
        self._iterator = iter(dataloader)

    def _advance_epoch(self) -> None:
        if self.sampler is not None:
            self._epoch += 1
            self.sampler.set_epoch(self._epoch)

    def reset(self) -> None:
        self._advance_epoch()
        self._iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._advance_epoch()
            self._iterator = iter(self.dataloader)
            return next(self._iterator)


class TaskEngine(Engine):
    """Step-based engine for task training (finetuning).

    Extends the pretraining engine with: step semantics, gradient
    accumulation, transform/collate step hooks (with dataloader
    reinitialization on active-window changes), EMA via the model's
    ``on_train_batch_end``, periodic validation with best-checkpoint
    tracking, and last/best checkpoint + export writing.
    """

    def fit_task(
        self,
        train_model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        *,
        out_dir: Path,
        steps: int,
        gradient_accumulation_steps: int = 1,
        step_hooks: list[Any] | None = None,
        val_every_num_steps: int | None = None,
        save_every_num_steps: int = 1000,
        save_last: bool = True,
        save_best: bool = True,
        devices: int | str = "auto",
        precision: str = "auto",
        seed: int = 0,
        checkpoint: Path | None = None,
        resume_interrupted: bool = False,
        logger_fn: Any = None,
        log_every_n_steps: int = 50,
        normalize_args: dict | None = None,
        model_class_path: str = "",
        model_init_args: dict | None = None,
        debug_args: Any = None,
        visualize_examples: bool = True,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        callbacks: list[Any] | None = None,
        gradient_checkpointing: bool = False,
        parallel: str = "ddp",
        amp_check: bool = True,
    ) -> TaskFitResult:
        """Run the step loop and return the fitted artifacts.

        Early stopping: ``early_stopping_patience`` consecutive validations
        without an improvement greater than ``early_stopping_min_delta`` stop
        the run (requires ``val_dataloader`` and ``val_every_num_steps``).
        ``callbacks`` are duck-typed objects; the optional hooks are
        ``on_train_start(steps, out_dir)``, ``on_step_end(step, log_dict)``,
        ``on_validation_end(step, metrics)``, ``on_early_stop(step,
        best_metric, patience)`` and ``on_train_end(result)``. Hooks run on
        rank zero only; a callback exception is logged and never aborts
        training.
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if gradient_accumulation_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps must be >= 1, got "
                f"{gradient_accumulation_steps}."
            )
        if len(train_dataloader) == 0:
            raise ValueError(
                "Training dataloader is empty; check the dataset root, filters, "
                "and batch_size (a zero-length loader would silently train for "
                "zero steps)."
            )
        if val_dataloader is not None and len(val_dataloader) == 0:
            raise ValueError(
                "Validation dataloader is empty; an empty val set would record "
                "val_loss=0 and can save a bogus best checkpoint."
            )
        parallel = str(parallel).lower()
        if parallel not in ("ddp", "fsdp"):
            raise NotImplementedError(
                f"parallel='{parallel}' is not implemented; supported: "
                "ddp (default) and fsdp. DeepSpeed/ZeRO/elastic training "
                "need an external launcher (see docs/parallelism.md)."
            )
        out_dir = Path(out_dir)
        if early_stopping_patience is not None:
            if early_stopping_patience < 1:
                raise ValueError(
                    "early_stopping_patience must be >= 1, got "
                    f"{early_stopping_patience}."
                )
            if val_dataloader is None or not val_every_num_steps:
                raise ValueError(
                    "early_stopping_patience requires a val_dataloader and "
                    "val_every_num_steps; otherwise no intermediate metric "
                    "can trigger the stop."
                )
        if early_stopping_min_delta < 0:
            raise ValueError(
                f"early_stopping_min_delta must be >= 0, got "
                f"{early_stopping_min_delta}."
            )
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        from autovision.debug.debug_args import DebugArgs
        from autovision.debug.nan_capture import (
            NaNCaptureMonitor,
            NaNDetectedError,
        )
        from autovision.debug.underflow_overflow import DebugUnderflowOverflow

        debug_args = debug_args or DebugArgs()
        nan_monitor = None
        if debug_args.nan_capture is not None and debug_args.nan_capture.enabled:
            nan_monitor = NaNCaptureMonitor(
                train_model,
                debug_args.nan_capture.output_dir(out_dir),
                max_microbatches=debug_args.nan_capture.max_microbatches,
                model_class_path=model_class_path,
                model_init_args=model_init_args,
            )
        underflow_monitor = None
        if debug_args.underflow_overflow is not None and debug_args.underflow_overflow.enabled:
            ua = debug_args.underflow_overflow
            underflow_monitor = DebugUnderflowOverflow(
                train_model, out_dir=ua.output_dir(out_dir),
                max_frames_to_save=ua.max_frames_to_save,
                trace_batch_nums=ua.trace_batch_nums,
                abort_after_batch_num=ua.abort_after_batch_num,
            )
        if checkpoint is not None and resume_interrupted:
            raise ValueError(
                "'checkpoint' and 'resume_interrupted' cannot both be set."
            )
        seed_everything(seed)
        device = resolve_device(devices)
        _set_accelerator_device(device)
        if parallel == "fsdp":
            if world_size < 2:
                raise ValueError(
                    "parallel='fsdp' needs torchrun with WORLD_SIZE >= 2; "
                    "a single process has nothing to shard."
                )
            if device.type != "cuda":
                raise ValueError(
                    "parallel='fsdp' requires CUDA; torch FSDP refuses CPU "
                    f"parameters (resolved device: '{device}')."
                )
        train_model.to(device)
        if gradient_checkpointing:
            enable_fn = getattr(train_model, "enable_gradient_checkpointing", None)
            patched = enable_fn() if callable(enable_fn) else 0
            if patched:
                logger.info(
                    "Gradient checkpointing enabled on %d module(s).", patched
                )
            else:
                logger.warning(
                    "gradient_checkpointing=True but no checkpointable "
                    "blocks were found; continuing without it."
                )
        if device.type == "cuda":
            enable_cuda_math_modes(device)

        # ── distributed setup (torchrun) ──
        ddp_train_model: nn.Module = train_model
        fsdp_train_model: nn.Module | None = None
        train_sampler = None
        val_sampler = None
        if world_size > 1:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend=ddp_backend_for(device.type),
                    timeout=timedelta(seconds=Env.DDP_TIMEOUT_SEC.value),
                )
            if parallel == "fsdp":
                from autovision.runtime.fsdp import wrap_fsdp

                fsdp_train_model = wrap_fsdp(train_model, device)
                ddp_train_model = fsdp_train_model
            else:
                ddp_train_model = nn.parallel.DistributedDataParallel(
                    train_model,
                    device_ids=[device.index] if device.type in ("cuda", "npu") else None,
                    # The EMA state is per-rank by construction; buffers are not
                    # broadcast every step.
                    broadcast_buffers=False,
                    # Step-dependent branches (early-epoch heads, debug hooks)
                    # leave some parameters grad-free on individual steps; DDP
                    # then aborts with "expected to have finished reduction in
                    # the prior iteration". Baseline matches with
                    # ddp_find_unused_parameters_true.
                    # Opt out (stable step shapes) via AUTOVISION_DDP_FIND_UNUSED=0.
                    find_unused_parameters=Env.DDP_FIND_UNUSED.value,
                )
            _validate_ddp_sampler(train_dataloader)
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataloader.dataset, shuffle=True, drop_last=True
            )
            train_dataloader = self._with_sampler(train_dataloader, train_sampler)
            if val_dataloader is not None:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_dataloader.dataset, shuffle=False, drop_last=False
                )
                val_dataloader = self._with_sampler(val_dataloader, val_sampler)
            logger.info(
                "Task training in DDP mode: world_size=%d, per-rank batches=%d",
                world_size, len(train_dataloader),
            )

        optimizer, scheduler = train_model.get_optimizer(
            total_steps=steps,
            # Effective batch = per-device batch × world × accumulation; the
            # LR recipe scales with sqrt of this (missing the accumulation
            # factor would under-scale the LR for accumulated runs).
            global_batch_size=(train_dataloader.batch_size or 1)
            * gradient_accumulation_steps
            * max(1, int(os.getenv("WORLD_SIZE", "1"))),
            train_num_batches=len(train_dataloader),
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        # AMP gates (D16) run BEFORE the banner so it prints the precision
        # actually trained with (the probe may downgrade the request).
        precision = resolve_amp_precision(
            train_model, device=device, precision=precision, amp_check=amp_check
        )
        autocast_dtype = {"bf16-mixed": torch.bfloat16, "16-mixed": torch.float16}.get(precision)
        scaler = _build_grad_scaler(precision, device.type)
        if world_size == 1 or _is_rank_zero_internal():
            _log_fit_banner(
                train_model, optimizer, scheduler, steps,
                gradient_accumulation_steps, train_dataloader,
                val_dataloader, precision, world_size,
            )

        last_path = out_dir / "checkpoints" / "last.ckpt"
        best_metric: float | None = None
        no_improve = 0

        def _restore_last_state(
            *, validate_finite: bool = False
        ) -> tuple[int, dict[str, Any]]:
            """Reload every tracked state from ``last.ckpt`` (D32).

            Shared by ``resume_interrupted`` and the NaN-recovery handler so
            a mid-run rollback restores exactly what a resume would: model
            (incl. the EMA sidecar), optimizer, scheduler, scaler, RNG, the
            best-metric baseline (sidecar takes precedence), and the
            early-stopping counter. Returns ``(resumed_step, engine_state)``.
            ``validate_finite`` refuses a checkpoint that itself carries
            non-finite weights — the recovery path sets it; regular resumes
            keep their historical tolerance.
            """
            nonlocal best_metric, no_improve
            checkpoint_obj = Checkpoint.from_path(last_path, map_location=str(device))
            if validate_finite:
                # Same guard as the pretraining rollback: weights AND the
                # engine_state payload (optimizer/EMA sidecar/scaler) must
                # both be clean, or the rollback reinstalls the poison.
                _assert_state_dict_finite(checkpoint_obj.state_dict, last_path)
            engine_state = checkpoint_obj.autovision.engine_state
            if validate_finite:
                _assert_state_dict_finite(engine_state, last_path)
            if fsdp_train_model is not None:
                from autovision.runtime.fsdp import load_full_state_dict

                load_full_state_dict(
                    fsdp_train_model,
                    optimizer,
                    checkpoint_obj.state_dict,
                    engine_state.get("optimizer"),
                )
            else:
                train_model.load_state_dict(checkpoint_obj.state_dict)
                if "optimizer" in engine_state:
                    optimizer.load_state_dict(engine_state["optimizer"])
            if "scheduler" in engine_state:
                scheduler.load_state_dict(engine_state["scheduler"])
            if scaler is not None and "scaler" in engine_state:
                scaler.load_state_dict(engine_state["scaler"])
            _restore_rng_state(engine_state.get("rng"))
            resumed_step = int(engine_state.get("step", -1)) + 1
            # Restore the best-metric baseline: without it the first
            # post-resume validation resets tracking and a WORSE metric
            # overwrites best.ckpt (seen in the pilot2 resume — a
            # 0.1033 best was replaced by a 0.0978 dip). Old checkpoints
            # without the key fall back to fresh tracking. The
            # last.ckpt value lags its own validation (the final save
            # precedes it), so the sidecar written at every best
            # improvement takes precedence.
            best_metric = engine_state.get("best_metric")
            sidecar = out_dir / "checkpoints" / "best_metric.json"
            if sidecar.exists():
                try:
                    sidecar_value = float(
                        json.loads(sidecar.read_text())["best_metric"]
                    )
                except (OSError, ValueError, KeyError, TypeError):
                    sidecar_value = None
                if sidecar_value is not None and (
                    best_metric is None or sidecar_value > best_metric
                ):
                    best_metric = sidecar_value
            # Early-stopping state (D17): continue the patience counter
            # instead of resetting it. Old checkpoints without the key
            # fall back to fresh counting.
            no_improve = int(engine_state.get("no_improve", 0))
            if hasattr(train_model, "on_load_checkpoint"):
                train_model.on_load_checkpoint({"ema": engine_state.get("ema")})
            return resumed_step, engine_state

        start_step = 0
        if resume_interrupted:
            if last_path.exists():
                start_step, resume_state = _restore_last_state()
                # Fit-args drift check (D17): loud, never blocking — the
                # current call's values govern.
                check_resume_fit_args(
                    resume_state.get("fit_args"),
                    steps=steps,
                    precision=precision,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    val_every_num_steps=val_every_num_steps,
                    save_every_num_steps=save_every_num_steps,
                    log_every_n_steps=log_every_n_steps,
                )
                logger.info("Resumed task training at step %d.", start_step)
        elif checkpoint is not None:
            ckpt = Checkpoint.from_path(checkpoint, map_location="cpu")
            if fsdp_train_model is not None:
                from autovision.runtime.fsdp import load_full_state_dict

                load_full_state_dict(
                    fsdp_train_model, optimizer, ckpt.state_dict, None
                )
            else:
                train_model.load_train_state_dict = getattr(  # hook if available
                    train_model, "load_train_state_dict", train_model.load_state_dict
                )
                train_model.load_train_state_dict(ckpt.state_dict)  # type: ignore[misc]

        step_hooks = step_hooks or []
        infinite_loader = _InfiniteCycleIterator(train_dataloader, sampler=train_sampler)
        train_window = _TrainWindow()
        oom_attempts = 0
        nan_attempts = 0
        stopped_early = False
        last_completed_step = start_step - 1
        best_metrics: dict[str, float] = {}
        watch_key = getattr(train_model, "watch_metric", None) or "val_metric/map"
        result = TaskFitResult(out_dir=out_dir, steps_run=0)
        callbacks = list(callbacks or [])

        callback_registry = CallbackRegistry()
        callback_registry.adapt(callbacks or [])
        self.callbacks = callback_registry  # post-construction add_callback use

        def _invoke(name: str, *args: Any) -> None:
            # The registry keeps the duck contract and isolates failures
            # per handler (design doc D22).
            callback_registry.run(name, *args)

        is_rank_zero_run = world_size == 1 or _is_rank_zero_internal()
        if is_rank_zero_run:
            _invoke("on_train_start", steps, out_dir)

        last_full_state: dict[str, Any] | None = None

        def _collect_state() -> tuple[dict[str, Any], dict[str, Any]]:
            if fsdp_train_model is not None:
                from autovision.runtime.fsdp import collect_full_state_dict

                return collect_full_state_dict(
                    fsdp_train_model, optimizer, rank0_only=True
                )
            return train_model.state_dict(), optimizer.state_dict()

        def save_checkpoint(step: int, best_or_last: str) -> None:
            nonlocal last_full_state
            with otel_span("engine.checkpoint", {"step": step, "kind": best_or_last}):
                # D33: a NaN EMA must not enter the file — resync from live
                # weights first. FSDP shards the live model, so the per-
                # tensor comparison is skipped there (recorded deviation).
                _repair_ema_if_present(train_model, allow=fsdp_train_model is None)
                # FSDP gathers collectively on every rank; the other ranks
                # return empty dicts and must not write. Non-FSDP only
                # materializes the (potentially large) state on rank zero;
                # the others still join the failure barrier below.
                is_zero = world_size == 1 or _is_rank_zero_internal()
                if fsdp_train_model is not None or is_zero:
                    state, optim_state = _collect_state()
                else:
                    state, optim_state = {}, {}
                last_full_state = state
                write_ok = True
                write_error: Exception | None = None
                if is_zero:
                    try:
                        checkpoint_dict: dict = {"state_dict": state}
                        if hasattr(train_model, "on_save_checkpoint"):
                            # EMA is a plain object (not an FSDP-managed
                            # module), so this stays a full sidecar.
                            train_model.on_save_checkpoint(checkpoint_dict)
                        engine_state: dict = {
                            "optimizer": optim_state,
                            "scheduler": scheduler.state_dict(),
                            "rng": _capture_rng_state(),
                            "step": step,
                            # Closure read at call time: the best baseline
                            # rides in every checkpoint so a resume can
                            # restore it.
                            "best_metric": best_metric,
                            # Early-stopping state (D17): resuming with a
                            # zeroed patience counter would let a model that
                            # already exhausted its patience survive many
                            # more validations.
                            "no_improve": no_improve,
                            # Fit-args snapshot (D17): the resume path warns
                            # on out-of-whitelist drift against this.
                            "fit_args": {
                                "steps": steps,
                                "precision": precision,
                                "gradient_accumulation_steps":
                                    gradient_accumulation_steps,
                                "early_stopping_patience":
                                    early_stopping_patience,
                                "early_stopping_min_delta":
                                    early_stopping_min_delta,
                                "val_every_num_steps": val_every_num_steps,
                                "save_every_num_steps": save_every_num_steps,
                                "log_every_n_steps": log_every_n_steps,
                            },
                        }
                        if "ema" in checkpoint_dict:
                            engine_state["ema"] = checkpoint_dict["ema"]
                        if scaler is not None:
                            engine_state["scaler"] = scaler.state_dict()
                        metadata = CheckpointMetadata(
                            epoch=step,
                            global_step=step,
                            normalize_args=normalize_args,
                            engine_state=engine_state,
                        )
                        Checkpoint(
                            state_dict=_sanitize_state_dict_for_serialization(
                                checkpoint_dict["state_dict"]
                            ),
                            autovision=metadata,
                        ).save(out_dir / "checkpoints" / f"{best_or_last}.ckpt")
                        _invoke("on_checkpoint_save", step + 1, best_or_last)
                    except Exception as error:  # noqa: BLE001 - propagate via barrier
                        write_ok = False
                        write_error = error
                        logger.exception(
                            "rank zero failed to write the %s checkpoint.",
                            best_or_last,
                        )
                # All ranks participate, so a rank-zero disk/permission
                # failure aborts the job instead of hanging on the next
                # collective.
                _rank_zero_write_barrier(device, world_size, write_ok)
                if write_error is not None:
                    raise write_error

        def export_weights(
            step: int, best_or_last: str, state: dict[str, Any] | None = None
        ) -> Path:
            export_dir = out_dir / "exported_models"
            export_dir.mkdir(parents=True, exist_ok=True)
            path = export_dir / f"exported_{best_or_last}.pt"
            if fsdp_train_model is None:
                export_state = train_model.get_export_state_dict()
            elif getattr(train_model, "ema_model", None) is not None:
                # EMA is outside FSDP (plain object): its weights are full.
                export_state = {
                    f"ema_model.{key}": value
                    for key, value in train_model.ema_model.model.state_dict().items()
                }
            else:
                # Full FSDP state keys are ``model.*``; the task-model
                # loaders accept that prefix.
                export_state = state if state is not None else last_full_state or {}
            torch.save(
                {
                    "schema_version": 1,
                    "kind": "task_model",
                    "model_class_path": model_class_path,
                    "model_init_args": model_init_args or {},
                    "train_model": export_state,
                    "normalize_args": normalize_args,
                    "version": __import__("autovision", fromlist=["__version__"]).__version__,
                },
                path,
            )
            return path

        fit_span = start_span(
            "task.fit",
            {
                "device": str(device),
                "precision": precision,
                "world_size": world_size,
                "steps": steps,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
        )
        try:
            # Force train mode UP FRONT: HF `from_pretrained`-built task
            # models start in eval mode, and the only other train() call in
            # this loop sits at the END of the validation block — so every
            # step before the FIRST validation would otherwise run in eval
            # mode (no denoising groups, truncated aux losses, frozen BN
            # stats, no dropout). Pure-nn.Module constructors are unaffected
            # (train is the nn.Module default), which is why this only
            # surfaced on the HF-wrapped families (rfdetr).
            train_model.train()
            # D32: while (not for) so NaN recovery can rewind `step` and
            # re-run from the last checkpoint's step.
            step = start_step
            while step < steps:
                with otel_span("engine.step", {"step": step}):
                    for hook in step_hooks:
                        hook.set_step(step)
                    if any(
                        hook.requires_dataloader_reinitialization() for hook in step_hooks
                        if hasattr(hook, "requires_dataloader_reinitialization")
                    ):
                        infinite_loader.reset()
                        for hook in step_hooks:
                            if hasattr(hook, "mark_dataloader_as_reinitialized"):
                                hook.mark_dataloader_as_reinitialized()

                    # OOM recovery (D21): allocator exhaustion on the compute region
                    # halves the per-device batch and rebuilds the loader — single-process
                    # only (a mid-run batch change under DDP must be synchronized across
                    # ranks, which asymmetric catches cannot do safely). The OOMed step
                    # is skipped; optimizer/scheduler state is preserved.
                    corrupt_loss: torch.Tensor | None = None
                    try:
                        with otel_span("engine.dataload", {"step": step}):
                            batch = self._to_device(next(infinite_loader), device)
                        if nan_monitor is not None:
                            nan_monitor.capture_batch(batch)
                            nan_monitor.set_optimizer_state(optimizer.state_dict())
                        if (
                            visualize_examples
                            and step < 3
                            and _is_rank_zero_internal()
                            and "bboxes" in batch  # detection-specific visualization
                        ):
                            from autovision.visualize import save_detection_examples

                            save_detection_examples(
                            out_dir, batch, step,
                            normalize_args=(
                                normalize_args if isinstance(normalize_args, dict) else None
                            ),
                        )
                        if underflow_monitor is not None:
                            underflow_monitor.set_batch_number(step)
                            if underflow_monitor.aborted:
                                break
                        autocast = (
                            torch.autocast(device_type=device.type, dtype=autocast_dtype)
                            if autocast_dtype is not None
                            else nullcontext()
                        )
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_loss = 0.0
                        for accumulation_index in range(gradient_accumulation_steps):
                            batch_i = (
                                self._to_device(next(infinite_loader), device)
                                if accumulation_index > 0
                                else batch
                            )
                            # Skip gradient synchronization for all but the last
                            # microbatch of an accumulation window.
                            sync_context = (
                                ddp_train_model.no_sync()  # type: ignore[union-attr]
                                if world_size > 1
                                and accumulation_index < gradient_accumulation_steps - 1
                                else nullcontext()
                            )
                            with sync_context:
                                with autocast:
                                    # forward == training_step; DDP syncs through it.
                                    step_result = ddp_train_model(batch_i, step)
                            loss = step_result.loss / gradient_accumulation_steps
                            if corrupt_loss is None and not torch.isfinite(loss):
                                # Record the first poisoned microbatch (D32);
                                # the accumulation window runs to completion so
                                # every DDP rank posts the same collective
                                # sequence — the post-window verdict decides.
                                corrupt_loss = step_result.loss.detach()
                            if scaler is not None:
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()
                            # Kept on-device: one sync at logging time instead
                            # of one per microbatch.
                            accumulated_loss = accumulated_loss + step_result.loss.detach()
                        # D32: window-level corruption verdict. Single-process
                        # checks locally; under DDP a 4-byte MIN-reduce (the
                        # grad reduction already synchronized the ranks this
                        # step) flips every rank together, so all ranks roll
                        # back from the same checkpoint and stay aligned.
                        local_corrupted = corrupt_loss is not None
                        window_corrupted = local_corrupted
                        if world_size > 1:
                            window_corrupted = _sync_corruption_verdict(
                                device, world_size, window_corrupted
                            )
                        if window_corrupted:
                            if nan_monitor is not None:
                                # check_loss snapshots then raises
                                # NaNDetectedError; route that into the D32
                                # recovery instead of killing the run (with
                                # the snapshot already taken).
                                try:
                                    nan_monitor.check_loss(
                                        corrupt_loss
                                        if corrupt_loss is not None
                                        else step_result.loss
                                    )
                                except NaNDetectedError as captured:
                                    raise _TrainingCorruptedError(
                                        f"Non-finite loss at step {step}. "
                                        "Check data or lower lr."
                                    ) from captured
                            if local_corrupted:
                                raise _TrainingCorruptedError(
                                    f"Non-finite loss at step {step}. "
                                    "Check data or lower lr."
                                )
                            # Verdict flipped by a peer: keep the origin in
                            # the signal so the rollback log does not send a
                            # clean rank hunting for a NaN it never had.
                            raise _TrainingCorruptedError(
                                f"A peer rank reported a non-finite loss at "
                                f"step {step} (this rank's loss was finite); "
                                "rolling back jointly."
                            )
                        if scaler is not None:
                            # Unscale once after the accumulation window so the
                            # gradient checks and clip see real magnitudes.
                            scaler.unscale_(optimizer)
                        if nan_monitor is not None:
                            # Same routing for the gradient-level probe: a
                            # NaNDetectedError here used to abort the run; it
                            # now enters the same rollback path.
                            try:
                                nan_monitor.check_gradients(train_model)
                            except NaNDetectedError as captured:
                                raise _TrainingCorruptedError(
                                    f"Non-finite gradients at step {step} "
                                    "(nan_capture probe)."
                                ) from captured
                        clip_norm = train_model.clip_gradients(optimizer)
                        # AMP overflow is handled by GradScaler (skip + scale
                        # backoff); only the non-AMP path treats it as fatal.
                        if scaler is None:
                            _assert_finite_gradients(optimizer, clip_norm)
                        if scaler is not None:
                            scale_before = scaler.get_scale()
                            scaler.step(optimizer)
                            scaler.update()
                            stepped = scaler.get_scale() >= scale_before
                        else:
                            optimizer.step()
                            stepped = True
                        if stepped:
                            scheduler.step()
                    except _TrainingCorruptedError as corrupt_error:
                        # D32: roll back to last.ckpt and re-run from its
                        # step. Every rank received the same verdict, so all
                        # ranks restore identical state and stay aligned. The
                        # restored RNG reproduces the pre-corruption data
                        # order — a data-driven NaN reproduces and eventually
                        # exhausts the budget; a transient (AMP/allocator
                        # spike) recovers.
                        nan_attempts += 1
                        if nan_attempts > _NAN_RECOVERY_MAX_ATTEMPTS:
                            raise RuntimeError(
                                "Training failed: non-finite loss persisted "
                                f"across {_NAN_RECOVERY_MAX_ATTEMPTS} "
                                f"rollbacks to {last_path} (last detection at "
                                f"step {step}); check the data pipeline or "
                                "lower the learning rate."
                            ) from corrupt_error
                        if not last_path.exists():
                            raise RuntimeError(
                                f"Non-finite loss at step {step} and no "
                                "last.ckpt exists to roll back to (nothing "
                                "completed yet); fix the data or lower the "
                                "learning rate."
                            ) from corrupt_error
                        logger.warning(
                            "%s Recovery attempt %d/%d — rolling back to %s "
                            "and re-running from its step.",
                            corrupt_error, nan_attempts,
                            _NAN_RECOVERY_MAX_ATTEMPTS, last_path,
                        )
                        restored_step, _ = _restore_last_state(validate_finite=True)
                        step = restored_step
                        continue
                    except (torch.OutOfMemoryError, RuntimeError) as error:
                        if (
                            not _is_oom_error(error)
                            or oom_attempts >= _OOM_MAX_ATTEMPTS
                            # OOM recovery is single-process only: skipping
                            # the step here while peers post the D32
                            # corruption verdict (or the next grad
                            # all_reduce) would desync the collectives.
                            # Re-raising ends this rank; the finally block
                            # destroys the process group, so peers fail fast
                            # instead of hanging on a dead communicator.
                            or world_size > 1
                            or fsdp_train_model is not None
                            or not train_dataloader.batch_size
                            or train_dataloader.batch_size <= 1
                        ):
                            raise
                        oom_attempts += 1
                        if torch.cuda.is_available():
                            # Release cached segments so the halved-batch
                            # retry gets maximal allocator headroom; cheap
                            # when there is nothing cached.
                            torch.cuda.empty_cache()
                        halved = train_dataloader.batch_size // 2
                        logger.warning(
                            "OOM at step %d (attempt %d/%d); halving the "
                            "per-device batch to %d and rebuilding the "
                            "dataloader. The OOMed step is skipped; "
                            "optimizer/scheduler state kept. Set batch_size "
                            "explicitly or use batch_size='auto' to size "
                            "this before the run next time.",
                            step + 1, oom_attempts, _OOM_MAX_ATTEMPTS, halved,
                        )
                        train_dataloader = _clone_loader_with_batch_size(
                            train_dataloader, halved
                        )
                        infinite_loader = _InfiniteCycleIterator(
                            train_dataloader, sampler=train_sampler
                        )
                        optimizer.zero_grad(set_to_none=True)
                        # The while loop advances only at its bottom; the
                        # skipped step must increment explicitly here.
                        step += 1
                        continue
                if (
                    fsdp_train_model is not None
                    and getattr(train_model, "ema_model", None) is not None
                ):
                    # EMA keeps full parameter copies; FSDP's main model is
                    # sharded between forwards. Gather for the EMA update
                    # only (writeback=False), at the configured interval.
                    interval = max(1, Env.FSDP_EMA_INTERVAL.value)
                    if (step + 1) % interval == 0:
                        from torch.distributed.fsdp import (
                            FullyShardedDataParallel as FSDP,
                        )

                        with FSDP.summon_full_params(
                            fsdp_train_model, recurse=True, writeback=False
                        ):
                            train_model.on_train_batch_end()
                else:
                    train_model.on_train_batch_end()
                if is_rank_zero_run:
                    _invoke("on_step_end", step + 1, dict(step_result.log_dict))

                is_rank_zero = world_size == 1 or _is_rank_zero_internal()
                train_window.update(step_result.log_dict)
                if (
                    logger_fn is not None
                    and is_rank_zero
                    and (step + 1) % log_every_n_steps == 0
                ):
                    # Windowed means over [previous log, this one]: a single
                    # batch's loss swings by whole units at detection scale,
                    # so a spot sample every log_every_n_steps reads as
                    # noise (and the criterion components even more so).
                    means = train_window.means()
                    metrics = {"train_loss": means.get("loss", 0.0)}
                    metrics.update({
                        key: means[key] if key in means else value
                        for key, value in step_result.log_dict.items()
                    })
                    metrics["lr"] = max(
                        group["lr"] for group in optimizer.param_groups
                    )
                    backbone_lrs = [
                        group["lr"] for group in optimizer.param_groups
                        if str(group.get("name", "")).startswith("backbone")
                    ]
                    if backbone_lrs:
                        metrics["lr_backbone"] = backbone_lrs[0]
                    seconds_per_step = round(train_window.seconds_per_step(), 3)
                    metrics["seconds_per_step"] = seconds_per_step
                    _log_metrics_safely(logger_fn, metrics, step + 1)
                    logger.info(
                        "Step %d/%d | train_loss %.4f | lr %.2e%s | %.2f s/step",
                        step + 1, steps, metrics["train_loss"], metrics["lr"],
                        (
                            f" (backbone {metrics['lr_backbone']:.2e})"
                            if "lr_backbone" in metrics else ""
                        ),
                        seconds_per_step,
                    )
                    train_window.reset()

                is_last_step = step == steps - 1
                is_save_step = (
                    save_last
                    and save_every_num_steps > 0
                    and (step + 1) % save_every_num_steps == 0
                )
                if is_save_step or (is_last_step and save_last):
                    save_checkpoint(step, "last")
                    if is_rank_zero:
                        result.checkpoint_path = out_dir / "checkpoints" / "last.ckpt"
                        result.exported_model_path = export_weights(
                            step, "last", last_full_state
                        )
                        logger.info("Saved last checkpoint at step %d.", step + 1)

                # One validation ahead of schedule (D17 possible_stop): when
                # exactly one non-improving validation short of the patience
                # threshold, validating now instead of waiting for the next
                # scheduled point gives early stopping a timely decision.
                possible_stop = (
                    early_stopping_patience is not None
                    and early_stopping_patience >= 2
                    and no_improve == early_stopping_patience - 1
                )
                do_validation = (
                    val_dataloader is not None
                    and (
                        is_last_step
                        or possible_stop
                        or (
                            val_every_num_steps
                            and (step + 1) % val_every_num_steps == 0
                        )
                    )
                )
                if do_validation and val_dataloader is not None:
                    with otel_span("engine.validation", {"step": step}):
                        # D34: clear only when the caching allocator is
                        # hoarding (reserved share above threshold; MPS
                        # always, CPU never) — memory-saturated GPUs still
                        # clear every time, small runs skip the round trip.
                        _maybe_clear_accelerator_cache(device)
                        train_model.eval()
                        if val_sampler is not None:
                            val_sampler.set_epoch(step)
                        val_losses = []
                        # Validation calls the unwrapped model directly; FSDP
                        # keeps parameters sharded outside its own forward, so
                        # gather them for the eval pass (writeback=False).
                        if fsdp_train_model is not None:
                            from torch.distributed.fsdp import (
                                FullyShardedDataParallel as FSDP,
                            )

                            param_context = FSDP.summon_full_params(
                                fsdp_train_model,
                                recurse=True,
                                writeback=False,
                            )
                        else:
                            param_context = nullcontext()
                        with param_context, torch.no_grad():
                            for val_batch in val_dataloader:
                                val_batch = self._to_device(val_batch, device)
                                val_result = train_model.validation_step(val_batch, step)
                                val_losses.append(float(val_result.loss.detach()))
                        val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
                        if world_size > 1:
                            # Gather per-rank metric state; the metric is computed
                            # once on rank zero over the union of all predictions.
                            metric_state = train_model.export_metric_state("val")
                            gathered: list[Any] = [None] * world_size
                            torch.distributed.all_gather_object(gathered, metric_state)
                            if is_rank_zero:
                                # Clear the local accumulation first: rank zero's
                                # own shard arrives inside `gathered` and would
                                # otherwise be counted twice.
                                train_model.reset_metrics("val")
                                for rank_state in gathered:
                                    train_model.merge_metric_state("val", rank_state)
                            else:
                                train_model.reset_metrics("val")
                            loss_tensor = torch.tensor([val_loss], device=device)
                            torch.distributed.all_reduce(loss_tensor)
                            val_loss = float(loss_tensor.item()) / world_size
                        if not math.isfinite(val_loss):
                            # D32 boundary: a non-finite VALIDATION loss is a
                            # measurement-side fault (training windows verify
                            # weights/gradients every step), so it does not
                            # trigger rollback — replaying the same val shard
                            # would only reproduce it and discard clean
                            # training work. Warn loudly (rank zero only, per
                            # the logging convention); the NaN watch value
                            # fails the `improved` comparison, so best-metric
                            # tracking stays untouched and early stopping
                            # keeps its usual cadence.
                            if is_rank_zero:
                                logger.warning(
                                    "Non-finite validation loss at step %d; "
                                    "check the validation set or reporting "
                                    "pipeline (this round counts as no "
                                    "improvement).",
                                    step + 1,
                                )
                        val_metrics: dict[str, float] = (
                            train_model.compute_metrics("val") if is_rank_zero else {}
                        )
                        # D34: symmetric release on the way back to training,
                        # same threshold rule as the pre-validation clear.
                        _maybe_clear_accelerator_cache(device)
                    val_metrics["val_loss"] = val_loss
                    if world_size > 1:
                        # Broadcast rank zero's metrics so every rank makes
                        # the same best-model decision.
                        payload = [val_metrics]
                        torch.distributed.broadcast_object_list(payload, src=0)
                        val_metrics = payload[0]
                    if logger_fn is not None and is_rank_zero:
                        _log_metrics_safely(logger_fn, val_metrics, step + 1)
                    if is_rank_zero:
                        report = None
                        report_fn = getattr(train_model, "validation_report", None)
                        if callable(report_fn):
                            # The report is cosmetic — a renderer failure
                            # must never kill a training run.
                            try:
                                report = report_fn(val_metrics)
                            except Exception:
                                logger.debug(
                                    "validation_report failed", exc_info=True
                                )
                        if report:
                            logger.info(
                                "Validation at step %d:\n%s", step + 1, report
                            )
                        else:
                            watched = {
                                k: v for k, v in val_metrics.items()
                                if "loss" not in k
                            }
                            logger.info(
                                "Validation at step %d: %s", step + 1,
                                {k: round(v, 4) for k, v in watched.items()},
                            )
                    if is_rank_zero_run:
                        _invoke("on_validation_end", step + 1, dict(val_metrics))
                    watch_value = val_metrics.get(watch_key, -val_metrics["val_loss"])
                    if best_metric is None:
                        first_baseline = True
                        # A non-finite first evaluation is not a baseline:
                        # every later comparison with a NaN baseline fails,
                        # so accepting it here would freeze best tracking
                        # (and write a NaN best.ckpt) for the whole run.
                        improved = math.isfinite(watch_value)
                    else:
                        first_baseline = False
                        improved = (
                            watch_value > best_metric + early_stopping_min_delta
                        )
                    if improved:
                        best_metric = watch_value
                        best_metrics = dict(val_metrics)
                        no_improve = 0
                        export_ok = True
                        export_error: Exception | None = None
                        if save_best:
                            save_checkpoint(step, "best")
                        if is_rank_zero:
                            try:
                                if save_best:
                                    result.best_checkpoint_path = out_dir / "checkpoints" / "best.ckpt"
                                    result.best_exported_model_path = export_weights(
                                        step, "best", last_full_state
                                    )
                                logger.info(
                                    "Best %s %s %.4f (step %d)",
                                    watch_key,
                                    "baseline set to" if first_baseline
                                    else "improved to",
                                    watch_value, step + 1,
                                )
                                # Sidecar for the resume path: the last.ckpt
                                # engine_state lags its own validation (the
                                # final save precedes it), so the freshest
                                # baseline lives here, written at each improve.
                                sidecar = out_dir / "checkpoints" / "best_metric.json"
                                sidecar.parent.mkdir(parents=True, exist_ok=True)
                                sidecar.write_text(
                                    json.dumps(
                                        {"best_metric": watch_value, "step": step + 1}
                                    )
                                )
                            except Exception as error:  # noqa: BLE001
                                export_ok = False
                                export_error = error
                        _rank_zero_write_barrier(device, world_size, export_ok)
                        if export_error is not None:
                            raise export_error
                    else:
                        no_improve += 1
                        if (
                            early_stopping_patience is not None
                            and no_improve >= early_stopping_patience
                        ):
                            stopped_early = True
                            last_completed_step = step
                            if save_last:
                                save_checkpoint(step, "last")
                            if is_rank_zero:
                                logger.info(
                                    "Early stopping at step %d: %s did not "
                                    "improve for %d validations (best %.4f).",
                                    step + 1, watch_key, no_improve,
                                    best_metric if best_metric is not None else float("nan"),
                                )
                                # Persist the latest weights even when this is
                                # not a scheduled save step (unless the user
                                # disabled last-checkpoint retention).
                                if save_last:
                                    result.checkpoint_path = out_dir / "checkpoints" / "last.ckpt"
                                    result.exported_model_path = export_weights(
                                        step, "last", last_full_state
                                    )
                                _invoke(
                                    "on_early_stop",
                                    step + 1, best_metric, no_improve,
                                )
                            train_model.train()
                            break
                    train_model.train()
                # A clean step restores the NaN-recovery budget (D32): a
                # data-driven NaN reproduces at the same replayed step, so
                # consecutive attempts still exhaust the budget, while
                # scattered transients each get the full allowance —
                # including runs without a validation set, where the
                # previous validation-time reset never fired.
                nan_attempts = 0
                last_completed_step = step
                step += 1

        finally:
            end_span(fit_span)
            if underflow_monitor is not None:
                underflow_monitor.remove_hooks()
            if world_size > 1 and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        result.steps_run = max(0, last_completed_step - start_step + 1)
        result.best_metrics = best_metrics
        result.stopped_early = stopped_early
        result.stop_step = last_completed_step + 1 if stopped_early else None
        if is_rank_zero_run:
            _invoke("on_train_end", result)
        return result

    @staticmethod
    def _with_sampler(loader: DataLoader, sampler: Any) -> DataLoader:
        """Rebuild ``loader`` with a distributed sampler (same semantics)."""
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=loader.num_workers,
            worker_init_fn=loader.worker_init_fn,
            generator=loader.generator,
            drop_last=loader.drop_last,
            collate_fn=loader.collate_fn,
            timeout=loader.timeout,
            pin_memory=loader.pin_memory,
            persistent_workers=loader.persistent_workers,
        )
