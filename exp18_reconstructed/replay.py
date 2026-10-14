"""Controller-safe reference selection, separated from phase-aware replay sampling."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class Remembered(Protocol):
    experience_id: int
    admitted_update: int


def eligible_reference_items(items: Iterable[Remembered], update_ids: set[int], update_index: int, min_age: int) -> list[Remembered]:
    """Return recency-qualified references without task/phase information."""
    return [item for item in items
            if item.experience_id not in update_ids and update_index - item.admitted_update >= min_age]
