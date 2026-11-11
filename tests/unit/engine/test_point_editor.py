"""Point-handle persistence and frame-edit isolation without a display or model."""

from __future__ import annotations

import pytest

from boxmot.engine.ui.point_editor import PointEditor


def test_committing_retains_each_objects_handles_without_replaying_points() -> None:
    editor = PointEditor()
    first = editor.put(1, (10, 20), 0)
    second = editor.put(2, (30, 40), 1)
    third = editor.put(1, (50, 60), 1)

    assert first.id < second.id < third.id
    assert editor.visible(1) == editor.changed(1) == (first, third)
    assert editor.visible(2) == editor.changed(2) == (second,)
    editor.commit()

    assert editor.visible(1) == (first, third)
    assert editor.visible(2) == (second,)
    assert editor.changed(1) == editor.changed(2) == ()
    assert editor.drafts == {}


def test_move_and_relabel_keep_identity_and_submit_only_the_changed_handle() -> None:
    editor = PointEditor()
    first = editor.put(1, (10, 20), 0)
    untouched = editor.put(1, (30, 40), 0)
    other_object = editor.put(2, (50, 60), 1)
    editor.commit()

    moved = editor.put(1, (12, 24), 0, point_id=first.id)
    relabeled = editor.put(1, moved.position, 1, point_id=first.id)

    assert moved.id == relabeled.id == first.id
    assert editor.visible(1) == (relabeled, untouched)
    assert editor.changed(1) == (relabeled,)
    assert editor.saved[1][first.id] == first
    assert editor.visible(2) == (other_object,) and editor.changed(2) == ()
    editor.commit()
    assert editor.visible(1) == (relabeled, untouched) and editor.changed(1) == ()


def test_reverting_to_saved_position_and_label_cancels_inference_changes() -> None:
    editor = PointEditor()
    original = editor.put(1, (10, 20), 0)
    editor.commit()
    editor.put(1, (30, 40), 1, point_id=original.id)
    editor.put(1, original.position, original.label, point_id=original.id)

    assert editor.visible(1) == (original,)
    assert editor.changed(1) == () and editor.drafts == {}
    editor.delete(1, original.id)
    editor.put(1, original.position, original.label, point_id=original.id)
    assert editor.visible(1) == (original,) and editor.drafts == {}


def test_deleting_saved_handle_is_undoable_and_does_not_submit_a_point() -> None:
    editor = PointEditor()
    point = editor.put(1, (10, 20), 0)
    editor.commit()
    before_delete = editor.snapshot(1)
    editor.delete(1, point.id)

    assert editor.visible(1) == editor.changed(1) == ()
    assert editor.drafts[1] == {point.id: None}
    assert editor.saved[1][point.id] == point
    editor.restore(1, before_delete)
    assert editor.visible(1) == (point,) and editor.changed(1) == ()
    editor.delete(1, point.id)
    editor.commit()
    assert editor.visible(1) == () and editor.saved == {}


def test_deleting_new_handle_cancels_it_without_reusing_its_id() -> None:
    editor = PointEditor()
    point = editor.put(1, (10, 20), 0)
    editor.delete(1, point.id)
    editor.delete(1, point.id)

    assert editor.visible(1) == editor.changed(1) == ()
    assert editor.drafts == {}
    replacement = editor.put(2, (30, 40), 1)
    assert replacement.id > point.id


def test_undo_snapshots_are_owned_and_do_not_roll_back_identity_allocation() -> None:
    editor = PointEditor()
    first = editor.put(1, (10, 20), 0)
    snapshot = editor.snapshot(1)
    second = editor.put(1, (30, 40), 1)
    assert snapshot == {first.id: first}

    editor.restore(1, snapshot)
    snapshot.clear()
    assert editor.visible(1) == editor.changed(1) == (first,)
    third = editor.put(1, (50, 60), 1)
    assert third.id > second.id
    assert editor.changed(1) == (first, third)


def test_discard_restores_saved_points_and_deletions_across_objects() -> None:
    editor = PointEditor()
    first = editor.put(1, (10, 20), 0)
    second = editor.put(2, (30, 40), 1)
    editor.commit()
    editor.put(1, (12, 24), 1, point_id=first.id)
    editor.delete(2, second.id)
    abandoned = editor.put(3, (50, 60), 0)
    editor.discard()

    assert editor.visible(1) == (first,)
    assert editor.visible(2) == (second,)
    assert editor.visible(3) == () and editor.drafts == {}
    assert editor.put(3, (50, 60), 0).id > abandoned.id


def test_removed_objects_cannot_be_resurrected_by_their_pending_edits() -> None:
    editor = PointEditor()
    first = editor.put(1, (10, 20), 0)
    second = editor.put(2, (30, 40), 1)
    editor.commit()
    editor.put(1, (12, 24), 1, point_id=first.id)
    editor.put(3, (50, 60), 0)
    editor.commit(object_id for object_id in (1, 3))

    assert editor.saved == {2: {second.id: second}}
    assert editor.visible(1) == editor.visible(3) == ()
    assert editor.drafts == {}


def test_point_edits_and_deletions_do_not_cross_object_boundaries() -> None:
    editor = PointEditor()
    point = editor.put(1, (10, 20), 0)
    editor.commit()

    with pytest.raises(KeyError, match="does not belong to object 2"):
        editor.put(2, (30, 40), 1, point_id=point.id)
    editor.delete(2, point.id)
    assert editor.visible(1) == (point,)
    assert editor.visible(2) == () and editor.drafts == {}


@pytest.mark.parametrize("position,label", [((float("nan"), 0), 0), ((0, float("inf")), 1), ((0, 0), 2)])
def test_invalid_point_does_not_mutate_edits_or_consume_an_identity(position: tuple, label: int) -> None:
    editor = PointEditor()
    with pytest.raises(ValueError):
        editor.put(1, position, label)
    assert editor.saved == editor.drafts == {}
    assert editor.put(1, (10, 20), 0).id == 1


def test_new_editor_resets_sequence_state_without_affecting_the_previous_editor() -> None:
    previous = PointEditor()
    saved = previous.put(1, (10, 20), 0)
    previous.commit()
    previous.put(1, (30, 40), 1)
    previous.discard()
    current = PointEditor()

    assert current.visible(1) == current.changed(1) == ()
    assert current.saved == current.drafts == {}
    assert current.put(1, (50, 60), 1).id == 1
    assert previous.visible(1) == (saved,)
