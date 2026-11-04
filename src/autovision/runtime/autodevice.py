# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Idle-GPU selection for shared servers (design doc D18).

``-1`` inside a device specification is a placeholder meaning "pick an
idle GPU automatically": :func:`resolve_device_spec` expands every
placeholder in a spec string (``"0,-1"`` -> ``"cuda:0,cuda:3"``), and
:func:`autovision.engine.resolve_device` accepts the placeholder forms
directly (``"-1"``, ``"cuda:-1"``) for single-process resolution.
Selection filters candidate GPUs by free-memory and idle-time fractions
and ranks survivors by ``(utilization, -free_memory, random tiebreak)``;
the random tiebreak keeps several processes that start simultaneously
from all landing on the same "first idle" card. Degradation is
two-staged: when no card passes the thresholds, the least-busy visible
cards fill the remaining placeholders (with a warning); when there are
fewer visible GPUs than placeholders, the configuration is impossible
and expansion raises instead of inventing ``cpu`` ranks — a loud error
beats a silently half-GPU job. pynvml imports lazily and any NVML
failure is soft: placeholders degrade to sequential picks with a
warning.

Multi-process note: torchrun maps ``LOCAL_RANK`` -> ``cuda:<rank>``, so
no in-process spec can redirect rank 1 to a non-adjacent card. Resolve
the spec before launching (e.g. with :func:`resolve_device_spec`) and
export ``CUDA_VISIBLE_DEVICES`` accordingly.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Sequence

from autovision.errors import AutoVisionError

__all__ = [
    "GPUStat",
    "list_gpu_stats",
    "resolve_device_spec",
    "select_idle_gpus",
]

logger = logging.getLogger(__name__)

#: A GPU must have at least this fraction of its memory free to count as idle.
DEFAULT_MIN_MEMORY_FRACTION = 0.2
#: A GPU must be at least this fraction idle (1 - utilization) to count.
DEFAULT_MIN_FREE_FRACTION = 0.2


@dataclass(frozen=True)
class GPUStat:
    """Per-GPU utilization/memory snapshot (fractions in ``[0, 1]``).

    ``index`` is the physical (NVML) index, which differs from the torch
    index when ``CUDA_VISIBLE_DEVICES`` restricts visibility.
    """

    index: int
    utilization: float
    memory_free_fraction: float


def _load_gpu_stats() -> list[GPUStat] | None:
    """Snapshot all physical GPUs, or ``None`` when NVML is unusable.

    pynvml imports inside this function (optional dependency); driver or
    binding failures degrade to ``None`` with a warning instead of raising,
    so callers can fall back to sequential selection.
    """
    try:
        import pynvml  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "Idle-GPU selection needs the NVIDIA Management Library "
            "bindings (pip install nvidia-ml-py); '-1' device placeholders "
            "fall back to sequential picks."
        )
        return None
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 - any driver error is soft
        logger.warning(
            "NVML initialization failed (%s); '-1' device placeholders "
            "fall back to sequential picks.",
            exc,
        )
        return None
    try:
        stats: list[GPUStat] = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            total = float(memory.total)
            stats.append(
                GPUStat(
                    index=index,
                    utilization=float(utilization.gpu) / 100.0,
                    memory_free_fraction=(
                        float(memory.free) / total if total > 0.0 else 0.0
                    ),
                )
            )
        return stats
    except Exception as exc:  # noqa: BLE001 - per-query driver error is soft
        logger.warning(
            "NVML query failed (%s); '-1' device placeholders fall back "
            "to sequential picks.",
            exc,
        )
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 - shutdown is best-effort
            pass


def list_gpu_stats() -> list[GPUStat]:
    """Snapshot all physical GPUs (empty when NVML is unavailable).

    Returns:
        One :class:`GPUStat` per physical GPU, in physical-index order.
        An empty list means NVML is missing/failed or no GPU exists.
    """
    stats = _load_gpu_stats()
    return stats if stats is not None else []


def _cuda_visible_physical_ids() -> list[int] | None:
    """Physical GPU ids exposed via ``CUDA_VISIBLE_DEVICES`` (``None`` = all).

    Ids are physical NVML indices; the list position is the torch index
    torch assigns to each visible device.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    ids: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if token.isdigit():
            ids.append(int(token))
    if raw.strip() and not ids:
        # UUID/MIG entries cannot be mapped to NVML indices here: selection
        # would run blind, so say so instead of silently seeing zero GPUs.
        logger.warning(
            "CUDA_VISIBLE_DEVICES holds no numeric indices (UUID/MIG form); "
            "idle-GPU selection cannot map them and will treat no GPU as "
            "visible."
        )
    return ids


def _torch_to_physical(torch_index: int) -> int | None:
    """Translate a torch GPU index to its physical (NVML) index."""
    visible = _cuda_visible_physical_ids()
    if visible is None:
        return torch_index
    if 0 <= torch_index < len(visible):
        return visible[torch_index]
    return None


def _physical_to_torch(physical_index: int) -> int | None:
    """Translate a physical (NVML) GPU index back to the torch index."""
    visible = _cuda_visible_physical_ids()
    if visible is None:
        return physical_index
    try:
        return visible.index(physical_index)
    except ValueError:
        return None


def select_idle_gpus(
    count: int,
    *,
    candidates: Sequence[int] | None = None,
    stats: Sequence[GPUStat] | None = None,
    min_memory_fraction: float = DEFAULT_MIN_MEMORY_FRACTION,
    min_free_fraction: float = DEFAULT_MIN_FREE_FRACTION,
    rng: random.Random | None = None,
    warn: bool = True,
) -> list[int]:
    """Pick ``count`` idle GPUs, best first (torch indices).

    Args:
        count: How many idle GPUs to pick.
        candidates: Torch indices to consider; defaults to all visible GPUs.
        stats: Injected snapshot (tests); defaults to :func:`list_gpu_stats`.
        min_memory_fraction: Free-memory fraction a GPU must exceed.
        min_free_fraction: Idle-time fraction (``1 - utilization``) required.
        rng: Injectable RNG for the tiebreak (tests); defaults to ``random``.
        warn: False silences the shortfall warning (stage-2 callers that
            emit their own, more specific message).

    Returns:
        Up to ``count`` torch indices ranked by
        ``(utilization, -free_memory, tiebreak)``. Fewer than ``count``
        (with a warning) when not enough GPUs pass the thresholds — the
        caller decides whether shrinking is acceptable.
    """
    snapshot = list(stats) if stats is not None else list_gpu_stats()
    if count <= 0 or not snapshot:
        return []
    if rng is None:
        rng = random.Random()

    visible = _cuda_visible_physical_ids()
    if candidates is None:
        if visible is not None:
            candidate_torch_ids = list(range(len(visible)))
        else:
            candidate_torch_ids = list(range(len(snapshot)))
    else:
        candidate_torch_ids = list(candidates)

    by_physical = {stat.index: stat for stat in snapshot}
    idle: list[tuple[float, float, float, int]] = []
    for torch_index in candidate_torch_ids:
        physical_index = _torch_to_physical(torch_index)
        stat = by_physical.get(physical_index) if physical_index is not None else None
        if stat is None:
            continue
        if stat.memory_free_fraction < min_memory_fraction:
            continue
        if (1.0 - stat.utilization) < min_free_fraction:
            continue
        idle.append(
            (
                stat.utilization,
                -stat.memory_free_fraction,
                rng.random(),
                torch_index,
            )
        )
    idle.sort()
    selected = [torch_index for _, _, _, torch_index in idle[:count]]
    if len(selected) < count and warn:
        logger.warning(
            "Only %d of %d requested idle GPU(s) passed the thresholds "
            "(memory at least %.0f%% free, idle time at least %.0f%%).",
            len(selected),
            count,
            min_memory_fraction * 100.0,
            min_free_fraction * 100.0,
        )
    return selected


def _spec_tokens(spec: str) -> list[str]:
    """Split a device spec into normalized (stripped) comma tokens."""
    return [token.strip() for token in spec.split(",") if token.strip()]


def _explicit_cuda_indices(tokens: Sequence[str]) -> list[int]:
    """CUDA indices already claimed by concrete spec tokens.

    Both bare ``2`` and ``cuda:2`` forms claim index 2; non-CUDA device
    tokens (``cpu``, ``mps``, ``npu:0``) claim nothing.
    """
    claimed: list[int] = []
    for token in tokens:
        text = token
        if text.startswith("cuda:"):
            text = text.removeprefix("cuda:")
        if text.isdigit():
            claimed.append(int(text))
    return claimed


def _sequential_placeholder_fill(
    tokens: list[str], placeholder_positions: list[int]
) -> list[str]:
    """Fill placeholders with unclaimed CUDA indices (NVML-unavailable path).

    Raises:
        AutoVisionError: When fewer unclaimed GPUs remain than placeholders
            — the spec cannot be satisfied on this host, and a ``cpu``
            rank would only turn that into a silently half-GPU job.
    """
    visible = _cuda_visible_physical_ids()
    if visible is not None:
        total = len(visible)
    else:
        import torch

        total = torch.cuda.device_count()
    claimed = set(_explicit_cuda_indices(tokens))
    pool = [index for index in range(total) if index not in claimed]
    if len(pool) < len(placeholder_positions):
        raise AutoVisionError(
            f"Cannot satisfy {len(placeholder_positions)} '-1' device "
            f"placeholder(s): only {len(pool)} unclaimed CUDA GPU(s) "
            f"visible on this host (claimed: {sorted(claimed)}). Reduce "
            "the placeholders or add GPUs."
        )
    filled = list(tokens)
    for position in placeholder_positions:
        filled[position] = f"cuda:{pool.pop(0)}"
        logger.warning(
            "Device placeholder filled sequentially with '%s' "
            "(no idle-GPU information available).",
            filled[position],
        )
    return filled


def resolve_device_spec(spec: str) -> str:
    """Expand ``-1`` idle-GPU placeholders into concrete device strings.

    Args:
        spec: Device specification. Recognized tokens: bare ``-1`` and
            ``cuda:-1`` (idle-GPU placeholders), bare ``2`` and
            ``cuda:2`` (CUDA indices), anything else passes through
            unchanged (``cpu``, ``mps``, ``npu:0``). A ``npu:-1`` token is
            rejected — NVML only covers NVIDIA GPUs.

    Returns:
        The spec with every placeholder replaced by ``cuda:<index>``.
        Degradation is two-staged: idle-threshold misses fall back to the
        least-busy visible cards (with a warning), and when idle
        information is unavailable entirely, sequential unclaimed picks
        are used — the process count implied by the token list never
        shrinks. Specs without placeholders are returned unchanged.

    Raises:
        AutoVisionError: If a non-CUDA accelerator placeholder is used,
            or if fewer usable CUDA GPUs are visible than the spec has
            placeholders (the configuration cannot be satisfied).
    """
    tokens = _spec_tokens(spec)
    for token in tokens:
        if token.endswith(":-1") and not token.startswith(("cuda:", "-1")):
            prefix = token.removesuffix(":-1")
            raise AutoVisionError(
                f"Idle-GPU placeholders ('-1') are only supported for CUDA "
                f"devices, got '{token}'; resolve '{prefix}' devices "
                "explicitly."
            )
    placeholder_positions = [
        position
        for position, token in enumerate(tokens)
        if token == "-1" or token == "cuda:-1"
    ]
    if not placeholder_positions:
        return spec
    import torch

    stats = _load_gpu_stats() if torch.cuda.device_count() > 0 else None
    if not stats:
        # Covers both NVML being unusable and torch without CUDA: the
        # expanded spec must stay runnable on THIS process's torch.
        filled = _sequential_placeholder_fill(tokens, placeholder_positions)
        return ",".join(
            f"cuda:{token}" if token.isdigit() else token for token in filled
        )
    claimed = set(_explicit_cuda_indices(tokens))
    candidates = []
    visible = _cuda_visible_physical_ids()
    for torch_index in (
        range(len(visible)) if visible is not None else range(len(stats))
    ):
        # claimed/candidates share the TORCH index namespace; translating
        # to physical here would break the exclusion under CVD (a spec's
        # explicit id could be "re-selected" by a placeholder).
        physical = _torch_to_physical(torch_index)
        if physical is not None and torch_index not in claimed:
            candidates.append(torch_index)
    selected = select_idle_gpus(
        len(placeholder_positions),
        candidates=candidates,
        stats=stats,
        warn=False,  # the stage-2 fallback warning covers the shortfall
    )
    if len(selected) < len(placeholder_positions):
        # Stage 2: no card passes the thresholds (e.g. the only GPU is
        # somewhat busy) — fill the rest with the least-busy visible cards
        # rather than inventing cpu ranks.
        logger.warning(
            "Only %d of %d requested idle GPU(s) passed the thresholds; "
            "falling back to the least-busy visible GPUs.",
            len(selected),
            len(placeholder_positions),
        )
        remaining_needed = len(placeholder_positions) - len(selected)
        remaining_candidates = [
            index for index in candidates if index not in selected
        ]
        selected = selected + select_idle_gpus(
            remaining_needed,
            candidates=remaining_candidates,
            stats=stats,
            min_memory_fraction=0.0,
            min_free_fraction=0.0,
            warn=False,  # the outer fallback warning already covers this
        )
    if len(selected) < len(placeholder_positions):
        raise AutoVisionError(
            f"Cannot satisfy {len(placeholder_positions)} '-1' device "
            f"placeholder(s): only {len(selected)} usable CUDA GPU(s) "
            "visible on this host. Reduce the number of placeholders "
            "('...,-1') or add GPUs."
        )
    filled = list(tokens)
    for position, torch_index in zip(placeholder_positions, selected):
        filled[position] = f"cuda:{torch_index}"
    # Bare-int tokens are new grammar on this path only; normalize them so
    # torch.device accepts every token of the expanded spec.
    normalized = [
        f"cuda:{token}" if token.isdigit() else token for token in filled
    ]
    return ",".join(normalized)
