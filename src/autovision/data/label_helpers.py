# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Internal class id mapping: dataset ids -> contiguous ids from 0."""

from __future__ import annotations

__all__ = ["get_class_id_to_internal_class_id_mapping"]


def get_class_id_to_internal_class_id_mapping(
    class_ids: list[int],
) -> dict[int, int]:
    """Map the given (non-ignored) class ids to contiguous internal ids."""
    return {class_id: index for index, class_id in enumerate(sorted(class_ids))}
