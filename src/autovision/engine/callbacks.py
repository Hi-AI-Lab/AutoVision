# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Engine-level callback registry (design doc D22).

A named-event registry that both training loops can share, replacing the
per-loop duck-typed ``_invoke`` walk. Two registration styles coexist:

- explicit: ``registry.add("on_step_end", fn)`` — precise, composable;
- duck adaptation: :meth:`CallbackRegistry.adapt` scans objects for known
  hook names (the historical ``callbacks=`` contract), so existing callers
  keep working unchanged.

Isolation discipline (deliberately STRICTER than the upstream reference):
every handler runs inside its own try/except and a failure is logged as a
warning, never propagated — a broken callback must not kill a training
run. The event list is intentionally short; extend it only when a real
consumer appears, not for symmetry.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any

__all__ = ["CALLBACK_EVENTS", "CallbackRegistry"]

logger = logging.getLogger("autovision")

#: Events the engines fire, in lifecycle order. The hook signatures follow
#: the historical duck contract documented in ``TaskEngine.fit_task``.
CALLBACK_EVENTS: tuple[str, ...] = (
    "on_train_start",
    "on_step_end",
    "on_validation_end",
    "on_checkpoint_save",
    "on_early_stop",
    "on_train_end",
)


class CallbackRegistry:
    """Named-event handler registry with per-handler exception isolation."""

    def __init__(self, events: Sequence[str] = CALLBACK_EVENTS) -> None:
        self._events = tuple(events)
        self._handlers: dict[str, list[Callable[..., None]]] = {
            event: [] for event in self._events
        }

    # ── registration ─────────────────────────────────────────────────
    def add(self, event: str, handler: Callable[..., None]) -> None:
        """Append one handler for ``event`` (unknown events are rejected)."""
        self._require_event(event)
        self._handlers[event].append(handler)

    def set(self, event: str, handler: Callable[..., None]) -> None:
        """Replace all handlers for ``event`` with this one."""
        self._require_event(event)
        self._handlers[event] = [handler]

    def clear(self, event: str | None = None) -> None:
        """Drop handlers for one event, or every event when ``None``."""
        if event is None:
            for known in self._handlers:
                self._handlers[known] = []
            return
        self._require_event(event)
        self._handlers[event] = []

    def adapt(self, callbacks: Iterable[Any]) -> None:
        """Register the known hook methods found on duck-typed objects.

        An object that defines none of the known hooks is silently ignored;
        one that defines a hook not in ``CALLBACK_EVENTS`` is not magic —
        it simply never fires (the engines only dispatch known events).
        """
        for callback in callbacks:
            for event in self._events:
                hook = getattr(callback, event, None)
                if callable(hook):
                    self.add(event, hook)

    def _require_event(self, event: str) -> None:
        if event not in self._handlers:
            raise ValueError(
                f"Unknown callback event '{event}'; known: {list(self._events)}."
            )

    # ── dispatch ─────────────────────────────────────────────────────
    def run(self, event: str, *args: Any) -> None:
        """Fire ``event``; handler failures log a warning and never raise.

        Payload contracts: ``on_train_start(units, out_dir)`` (steps for
        ``fit_task``, epochs for ``Engine.fit``); ``on_train_end(result)``
        carries the loop's own result object. ``on_checkpoint_save`` is
        currently fired by ``fit_task`` only.
        """
        self._require_event(event)
        for handler in self._handlers.get(event, ()):
            try:
                handler(*args)
            except Exception:  # noqa: BLE001 - callbacks are non-fatal
                logger.warning(
                    "callback %s on %s failed; continuing.",
                    getattr(handler, "__qualname__", handler),
                    event,
                    exc_info=True,
                )

    def handlers(self, event: str) -> list[Callable[..., None]]:
        """Registered handlers for ``event`` (read-only copy, for tests)."""
        return list(self._handlers.get(event, ()))
