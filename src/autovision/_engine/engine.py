# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Training engine: step loop, precision, checkpointing, optional DDP.

Deliberately thin and framework-free. The loop implements exactly what the
methods need: forward/backward per step, gradient clipping, LR scheduler
steps, NaN guards, atomic checkpointing per epoch, and resume. Multi-device
training works under ``torchrun`` (environment-initialized process group);
single-process runs need no setup.
"""

from __future__ import annotations

import logging
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from autovision._accelerator import (
    ddp_backend_for,
    npu_available,
    supports_amp,
)
from autovision._checkpoint import Checkpoint, CheckpointMetadata
from autovision._distributed import get_local_rank
from autovision._env import Env
from autovision._observability.otel import end_span, otel_span, start_span

__all__ = ["Engine", "FitResult"]

logger = logging.getLogger("autovision")


def _is_rank_zero_internal() -> bool:
    from autovision._distributed import is_global_rank_zero

    return is_global_rank_zero()


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch globally.

    Dataloader worker processes re-seed from the global torch RNG per worker
    (PyTorch default ``base_seed`` mechanism), so worker randomness follows
    the global seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(devices: int | str = "auto") -> torch.device:
    """Resolve the torch device from the devices specification.

    ``"auto"`` picks CUDA > Ascend NPU (torch_npu) > MPS > CPU (honoring
    LOCAL_RANK under torchrun); explicit specs (``"cpu"``, ``"cuda:1"``,
    ``"npu:0"``, ``int`` indices) pass through. NPU is software-side
    support only (design §12.4) — real-hardware validation is M4+.
    """
    if isinstance(devices, int):
        if torch.cuda.is_available():
            return torch.device("cuda", devices % torch.cuda.device_count())
        if npu_available():
            return torch.device("npu", devices % torch.npu.device_count())
        return torch.device("cpu")
    if isinstance(devices, str) and devices != "auto":
        return torch.device(devices)
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
    """Drives the training loop for a :class:`~autovision._methods.Method`."""

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
    ) -> FitResult:
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
        seed_everything(seed)
        device = resolve_device(devices)
        method.to(device)

        # ── distributed setup (torchrun) ──
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        ddp_method: nn.Module = method
        sampler = None
        if world_size > 1:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend=ddp_backend_for(device.type)
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
        optimizer, scheduler = method.configure_optimizers(
            total_steps=max_epochs * steps_per_epoch, num_epochs=max_epochs
        )
        method.total_steps = max_epochs * steps_per_epoch

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
                start_epoch, global_step = self._resume(last_path, method, optimizer, scheduler, device)
            else:
                logger.info("No last.ckpt found; starting from scratch.")
        elif checkpoint is not None:
            self._load_weights(checkpoint, method, device)

        if precision in ("bf16-mixed", "16-mixed") and not supports_amp(device.type):
            logger.warning("Mixed precision requested without CUDA/NPU; using fp32.")
            precision = "32-true"
        autocast_dtype = {
            "bf16-mixed": torch.bfloat16,
            "16-mixed": torch.float16,
        }.get(precision)

        epochs_run = 0
        run_metrics: dict[str, float] = {}
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
            for epoch in range(start_epoch, max_epochs):
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
                            # Method.forward == training_step; DDP wraps forward.
                            loss = ddp_method(batch, batch_idx)
                        if not torch.isfinite(loss):
                            raise RuntimeError(
                                f"Non-finite loss at epoch {epoch}, step {global_step}: "
                                f"{loss}. Check for data issues or lower the learning rate."
                            )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        method.clip_gradients(optimizer)
                        method.on_before_optimizer_step(optimizer)
                        optimizer.step()
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
                                logger_fn(metrics, global_step)
                        else:
                            method.consume_log_dict()  # keep step state clean on all ranks
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
                    if logger_fn is not None:
                        logger_fn(knn_metrics, global_step)

                # Checkpoint at the end of every epoch (rank zero only).
                if world_size == 1 or torch.distributed.get_rank() == 0:
                    with otel_span("engine.checkpoint", {"epoch": epoch}):
                        self.save_checkpoint(
                            out_dir / "checkpoints" / "last.ckpt",
                            method=method,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            global_step=global_step,
                            normalize_args=normalize_args,
                        )
        finally:
            end_span(fit_span)
            # Never leave a hung process group behind on error paths.
            if world_size > 1 and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

        return FitResult(
            out_dir=out_dir,
            epochs_run=epochs_run,
            global_step=global_step,
            checkpoint_path=out_dir / "checkpoints" / "last.ckpt",
            metrics=run_metrics,
        )

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
        path: Path, method: nn.Module, optimizer, scheduler, device: torch.device
    ) -> tuple[int, int]:
        ckpt = Checkpoint.from_path(path, map_location=str(device))
        raw = {"state_dict": dict(ckpt.state_dict)}
        method.on_load_checkpoint(raw)
        method.load_state_dict(raw["state_dict"])
        engine_state = ckpt.autovision.engine_state
        if "optimizer" in engine_state:
            optimizer.load_state_dict(engine_state["optimizer"])
        if "scheduler" in engine_state:
            scheduler.load_state_dict(engine_state["scheduler"])
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
    def save_checkpoint(
        path: Path,
        method: nn.Module,
        optimizer,
        scheduler,
        epoch: int,
        global_step: int,
        normalize_args: dict | None = None,
    ) -> None:
        checkpoint = {"state_dict": method.state_dict()}
        method.on_save_checkpoint(checkpoint)
        metadata = CheckpointMetadata(
            epoch=epoch,
            global_step=global_step,
            normalize_args=normalize_args,
            engine_state={"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        )
        Checkpoint(state_dict=checkpoint["state_dict"], autovision=metadata).save(path)


@dataclass
class TaskFitResult:
    out_dir: Path
    steps_run: int
    checkpoint_path: Path | None = None
    best_checkpoint_path: Path | None = None
    exported_model_path: Path | None = None
    best_exported_model_path: Path | None = None
    best_metrics: dict[str, float] = field(default_factory=dict)


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
    ) -> TaskFitResult:
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        from autovision._debug.debug_args import DebugArgs
        from autovision._debug.nan_capture import NaNCaptureMonitor
        from autovision._debug.underflow_overflow import DebugUnderflowOverflow

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
        train_model.to(device)
        if device.type == "cuda":
            # Ampere+ tensor cores for the fp32 remainder (EMA, optimizer,
            # non-autocast ops); autocast-covered bf16 ops are unaffected.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # ── distributed setup (torchrun) ──
        ddp_train_model: nn.Module = train_model
        train_sampler = None
        val_sampler = None
        if world_size > 1:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend=ddp_backend_for(device.type)
                )
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

        start_step = 0
        if resume_interrupted:
            last_path = out_dir / "checkpoints" / "last.ckpt"
            if last_path.exists():
                ckpt = Checkpoint.from_path(last_path, map_location=str(device))
                train_model.load_state_dict(ckpt.state_dict)
                engine_state = ckpt.autovision.engine_state
                if "optimizer" in engine_state:
                    optimizer.load_state_dict(engine_state["optimizer"])
                if "scheduler" in engine_state:
                    scheduler.load_state_dict(engine_state["scheduler"])
                start_step = engine_state.get("step", -1) + 1
                if hasattr(train_model, "on_load_checkpoint"):
                    train_model.on_load_checkpoint(
                        {"ema": engine_state.get("ema")}
                    )
                logger.info("Resumed task training at step %d.", start_step)
        elif checkpoint is not None:
            ckpt = Checkpoint.from_path(checkpoint, map_location="cpu")
            train_model.load_train_state_dict = getattr(  # hook if available
                train_model, "load_train_state_dict", train_model.load_state_dict
            )
            train_model.load_train_state_dict(ckpt.state_dict)  # type: ignore[misc]

        if precision in ("bf16-mixed", "16-mixed") and not supports_amp(device.type):
            precision = "32-true"
        autocast_dtype = {"bf16-mixed": torch.bfloat16, "16-mixed": torch.float16}.get(precision)

        step_hooks = step_hooks or []
        infinite_loader = _InfiniteCycleIterator(train_dataloader, sampler=train_sampler)
        best_metric: float | None = None
        best_metrics: dict[str, float] = {}
        watch_key = getattr(train_model, "watch_metric", None) or "val_metric/map"
        result = TaskFitResult(out_dir=out_dir, steps_run=0)

        def save_checkpoint(step: int, best_or_last: str) -> None:
            with otel_span("engine.checkpoint", {"step": step, "kind": best_or_last}):
                checkpoint_dict: dict = {"state_dict": train_model.state_dict()}
                if hasattr(train_model, "on_save_checkpoint"):
                    train_model.on_save_checkpoint(checkpoint_dict)
                engine_state: dict = {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                }
                if "ema" in checkpoint_dict:
                    engine_state["ema"] = checkpoint_dict["ema"]
                metadata = CheckpointMetadata(
                    epoch=step,
                    global_step=step,
                    normalize_args=normalize_args,
                    engine_state=engine_state,
                )
                Checkpoint(
                    state_dict=checkpoint_dict["state_dict"], autovision=metadata
                ).save(out_dir / "checkpoints" / f"{best_or_last}.ckpt")

        def export_model(step: int, best_or_last: str) -> Path:
            export_dir = out_dir / "exported_models"
            export_dir.mkdir(parents=True, exist_ok=True)
            path = export_dir / f"exported_{best_or_last}.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "kind": "task_model",
                    "model_class_path": model_class_path,
                    "model_init_args": model_init_args or {},
                    "train_model": train_model.get_export_state_dict(),
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
            for step in range(start_step, steps):
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
                        from autovision._visualize import save_detection_examples

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
                        if not torch.isfinite(loss):
                            if nan_monitor is not None:
                                nan_monitor.check_loss(step_result.loss)
                            raise RuntimeError(
                                f"Non-finite loss at step {step}. Check data or lower lr."
                            )
                        loss.backward()
                        # Kept on-device: one sync at logging time instead
                        # of one per microbatch.
                        accumulated_loss = accumulated_loss + step_result.loss.detach()
                    if nan_monitor is not None:
                        nan_monitor.check_gradients(train_model)
                    train_model.clip_gradients(optimizer)
                optimizer.step()
                scheduler.step()
                train_model.on_train_batch_end()

                is_rank_zero = world_size == 1 or _is_rank_zero_internal()
                if (
                    logger_fn is not None
                    and is_rank_zero
                    and (step + 1) % log_every_n_steps == 0
                ):
                    metrics = {
                        "train_loss": float(accumulated_loss)
                        / gradient_accumulation_steps
                    }
                    metrics.update({
                        key: float(value) if torch.is_tensor(value) else value
                        for key, value in step_result.log_dict.items()
                    })
                    logger_fn(metrics, step + 1)

                is_last_step = step == steps - 1
                is_save_step = (
                    save_last
                    and save_every_num_steps > 0
                    and (step + 1) % save_every_num_steps == 0
                )
                if (is_save_step or is_last_step) and is_rank_zero:
                    save_checkpoint(step, "last")
                    result.checkpoint_path = out_dir / "checkpoints" / "last.ckpt"
                    result.exported_model_path = export_model(step, "last")
                    logger.info("Saved last checkpoint at step %d.", step + 1)

                do_validation = (
                    val_dataloader is not None
                    and (
                        is_last_step
                        or (val_every_num_steps and (step + 1) % val_every_num_steps == 0)
                    )
                )
                if do_validation:
                    with otel_span("engine.validation", {"step": step}):
                        train_model.eval()
                        if val_sampler is not None:
                            val_sampler.set_epoch(step)
                        val_losses = []
                        with torch.no_grad():
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
                        val_metrics: dict[str, float] = (
                            train_model.compute_metrics("val") if is_rank_zero else {}
                        )
                    val_metrics["val_loss"] = val_loss
                    if world_size > 1:
                        # Broadcast rank zero's metrics so every rank makes
                        # the same best-model decision.
                        payload = [val_metrics]
                        torch.distributed.broadcast_object_list(payload, src=0)
                        val_metrics = payload[0]
                    if logger_fn is not None and is_rank_zero:
                        logger_fn(val_metrics, step + 1)
                    if is_rank_zero:
                        watched = {k: v for k, v in val_metrics.items() if "loss" not in k}
                        logger.info(
                            "Validation at step %d: %s", step + 1,
                            {k: round(v, 4) for k, v in watched.items()},
                        )
                    watch_value = val_metrics.get(watch_key, -val_metrics["val_loss"])
                    if best_metric is None or watch_value > best_metric:
                        best_metric = watch_value
                        best_metrics = dict(val_metrics)
                        if save_best and is_rank_zero:
                            save_checkpoint(step, "best")
                            result.best_checkpoint_path = out_dir / "checkpoints" / "best.ckpt"
                            result.best_exported_model_path = export_model(step, "best")
                    train_model.train()

        finally:
            end_span(fit_span)
            if underflow_monitor is not None:
                underflow_monitor.remove_hooks()
            if world_size > 1 and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        result.steps_run = steps - start_step
        result.best_metrics = best_metrics
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
            drop_last=loader.drop_last,
            collate_fn=loader.collate_fn,
            timeout=loader.timeout,
            pin_memory=loader.pin_memory,
            persistent_workers=loader.persistent_workers,
        )
