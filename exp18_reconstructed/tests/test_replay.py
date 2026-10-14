from dataclasses import dataclass

from exp18_reconstructed.replay import eligible_reference_items


@dataclass
class Item:
    experience_id: int
    admitted_update: int
    phase_index: int


def test_reference_selection_excludes_exact_current_ids_without_phase_selection():
    items = [Item(1, 1, 0), Item(2, 1, 3), Item(3, 5, 1)]
    selected = eligible_reference_items(items, {1}, update_index=6, min_age=1)
    assert [x.experience_id for x in selected] == [2, 3]
    # The result is determined by identity and recency: phase values are irrelevant.
    assert [x.phase_index for x in selected] == [3, 1]


def test_reference_selection_rejects_recent_items():
    items = [Item(7, 10, 0)]
    assert eligible_reference_items(items, set(), update_index=10, min_age=1) == []
