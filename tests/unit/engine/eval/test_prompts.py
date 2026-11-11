from __future__ import annotations

from pathlib import Path

import pytest

from boxmot.engine.eval.prompts import PersonPrompt, load_first_visible_prompts


def _ground_truth(tmp_path: Path, *rows: str) -> Path:
    """Write a small MOT annotation file for prompt selection tests."""
    path = tmp_path / "gt.txt"
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def test_first_eligible_prompt_per_person_is_sorted_and_never_reseeded(tmp_path: Path) -> None:
    path = _ground_truth(
        tmp_path,
        "5,17,30,20,20,30,1,1,1",  # A later eligible observation appears first in the file.
        "1,17,10,20,20,30,1,1,0.5",  # Occluded at first appearance.
        "2,17,-1,20,20,30,1,1,1",  # Fully visible annotation but crossing the image plane.
        "3,17,10,20,20,30,1,1,1",
        "4,4,20,10,10,40,1,1,1",  # A new person arrives later.
        "1,9,0,0,20,30,1,1,1",  # Touching the border is still contained.
        "3,2,30,10,10,40,1,1,1",
        "3,2,30,10,10,40,1,1,1",  # Duplicate annotation does not duplicate the prompt.
        "6,17,40,20,20,30,1,1,1",  # A later return does not reseed the identity.
    )

    assert load_first_visible_prompts(
        path, image_sizes={frame: (100, 100) for frame in range(6)}, person_class_id=1
    ) == (
        PersonPrompt(0, 9, (0.0, 0.0, 20.0, 30.0)),
        PersonPrompt(2, 2, (30.0, 10.0, 40.0, 50.0)),
        PersonPrompt(2, 17, (10.0, 20.0, 30.0, 50.0)),
        PersonPrompt(3, 4, (20.0, 10.0, 30.0, 50.0)),
    )


def test_selection_uses_selected_frame_dimensions_class_and_mark(tmp_path: Path) -> None:
    path = _ground_truth(
        tmp_path,
        "1,0,0,0,100,100,1,2,1",  # Excluded frame must not consume this ID.
        "2,0,0,0,100,100,1,2,1",  # Outside this frame's smaller width.
        "3,0,0,0,80,100,1,2,1",  # Exactly matches both image boundaries.
        "3,-1,0,0,10,10,0,2,-1",  # Ignored region has no valid ID or visibility.
        "3,-1,0,0,10,10,1,1,-1",  # Other classes need not have person visibility.
        "3,4,0,0,10,10,-1,2,1",
        "3,5,0,0,10,10,0,2,1",
    )

    assert load_first_visible_prompts(path, image_sizes={1: (100, 80), 2: (100, 80)}, person_class_id=2) == (
        PersonPrompt(2, 0, (0.0, 0.0, 80.0, 100.0)),
    )


@pytest.mark.parametrize(
    "box",
    ["-1,0,10,10", "0,-1,10,10", "95,0,10,10", "0,95,10,10", "0,0,0,10", "0,0,10,-1"],
)
def test_ineligible_geometry_never_becomes_a_clipped_prompt(tmp_path: Path, box: str) -> None:
    path = _ground_truth(tmp_path, f"1,1,{box},1,1,1")

    assert load_first_visible_prompts(path, image_sizes={0: (100, 100)}, person_class_id=1) == ()


def test_visibility_threshold_is_inclusive(tmp_path: Path) -> None:
    path = _ground_truth(tmp_path, "1,1,0,0,10,10,1,1,0.8", "2,1,0,0,10,10,1,1,0.9")
    sizes = {0: (100, 100), 1: (100, 100)}

    assert load_first_visible_prompts(path, image_sizes=sizes, person_class_id=1) == ()
    assert load_first_visible_prompts(path, image_sizes=sizes, person_class_id=1, min_visibility=0.9) == (
        PersonPrompt(1, 1, (0.0, 0.0, 10.0, 10.0)),
    )


@pytest.mark.parametrize("visibility", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_visibility_threshold_is_rejected(tmp_path: Path, visibility: float) -> None:
    path = _ground_truth(tmp_path)

    with pytest.raises(ValueError, match="min_visibility.*finite.*\\[0, 1\\]"):
        load_first_visible_prompts(path, image_sizes={}, person_class_id=1, min_visibility=visibility)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("1,1,0,0,10,10,1,1", "including visibility"),
        ("1,1,broken,0,10,10,1,1,1", "could not convert string to float"),
        ("1,1,nan,0,10,10,1,1,1", "non-finite"),
        ("1,1,0,0,10,10,1,1,inf", "non-finite"),
        ("1.5,1,0,0,10,10,1,1,1", "frame ID must be an integer"),
        ("0,1,0,0,10,10,1,1,1", "frame ID must be an integer"),
        ("1,1.5,0,0,10,10,1,1,1", "object ID must be an integer"),
        ("1,-1,0,0,10,10,1,1,1", "object ID must be an integer"),
        ("1,9223372036854775808,0,0,10,10,1,1,1", "object ID must be within the nonnegative int64 range"),
        ("1,1,0,0,10,10,1,1,-0.1", "visibility must be within"),
        ("1,1,0,0,10,10,1,1,1.1", "visibility must be within"),
    ],
)
def test_invalid_annotations_have_path_and_line_context(tmp_path: Path, row: str, message: str) -> None:
    path = _ground_truth(tmp_path, "", row)

    with pytest.raises(ValueError, match=message) as error:
        load_first_visible_prompts(path, image_sizes={0: (100, 100)}, person_class_id=1)

    assert f"{path}:2:" in str(error.value)


def test_large_integer_object_ids_are_preserved(tmp_path: Path) -> None:
    path = _ground_truth(
        tmp_path,
        "1,9007199254740992,0,0,10,10,1,1,1",
        "1,9007199254740993,10,10,10,10,1,1,1",
    )

    prompts = load_first_visible_prompts(path, image_sizes={0: (100, 100)}, person_class_id=1)

    assert [prompt.object_id for prompt in prompts] == [9007199254740992, 9007199254740993]


def test_empty_annotations_have_no_prompts(tmp_path: Path) -> None:
    assert (
        load_first_visible_prompts(_ground_truth(tmp_path, "", "  "), image_sizes={0: (100, 100)}, person_class_id=1)
        == ()
    )
