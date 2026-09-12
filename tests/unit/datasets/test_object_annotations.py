"""Native object annotations retain fractional truncation and exact frame membership."""

from __future__ import annotations

from pathlib import Path

import pytest

from boxmot.datasets.readers.boxes3d import read_kitti_object_labels

_CAR = "Car 0.35 2 -0.2 10 20 40 60 1.5 2 4 5 6 20 0.25"
_IGNORE = "DontCare -1 -1 -10 0 0 4 3 -1 -1 -1 -1000 -1000 -1000 -10"


def test_object_labels_preserve_all_classes_metadata_and_empty_frames(tmp_path: Path) -> None:
    rows = (_CAR, _CAR.replace("Car", "Van"), _IGNORE)
    (tmp_path / "000000.txt").write_text("\n".join(rows) + "\n")
    (tmp_path / "000001.txt").write_text("")
    (tmp_path / "000002.txt").write_text(_CAR.replace("Car", "Pedestrian") + "\n")

    labels = read_kitti_object_labels(tmp_path, frame_count=3)

    assert labels.frame_rows == (rows, (), (_CAR.replace("Car", "Pedestrian"),))
    assert labels.frame_rows[0][0].split()[1] == "0.35"
    assert len(labels.source_sha256) == 64
    before = labels.source_sha256
    (tmp_path / "000000.txt").write_text("\n".join(rows).replace("0.35", "0.36") + "\n")
    assert read_kitti_object_labels(tmp_path, frame_count=3).source_sha256 != before


@pytest.mark.parametrize(
    "row,message",
    (
        ("0 7 " + _CAR, "15 KITTI object label fields"),
        (_CAR + " 0.9", "15 KITTI object label fields"),
        (_CAR.replace("0.35", "2"), "truncation must be a fraction"),
        (_CAR.replace("0.35", "-1"), "truncation must be a fraction"),
        (_CAR.replace("0.35", "nan"), "numeric fields must be finite"),
        (_CAR.replace("0.35 2", "0.35 1.5"), "occlusion must be an integer"),
        (_CAR.replace("0.35 2", "0.35 0.0"), "occlusion must be an integer"),
        (_CAR.replace("0.35 2", "0.35 0e0"), "occlusion must be an integer"),
        (_CAR.replace("0.35 2", "0.35 4"), "occlusion must be an integer"),
        (_CAR.replace("Car", "Cár"), "ASCII letters or underscores"),
        (_CAR.replace("Car", "Person-sitting"), "ASCII letters or underscores"),
        (_CAR.replace("Car", "C" * 255), "ASCII letters or underscores"),
        (_CAR.replace("Car ", "Car\N{NO-BREAK SPACE}"), "native ASCII text"),
        (_CAR.replace("10 20 40 60", "1_0 20 40 60"), "ASCII decimal notation"),
        (_CAR.replace("10 20 40 60", "10 20 10 60"), "image bounds"),
        (_CAR.replace("1.5 2 4", "0 2 4"), "3D dimensions must be positive"),
        (_CAR.replace("Car", "Van").replace("0.35", "2"), "truncation must be a fraction"),
    ),
)
def test_invalid_object_labels_name_the_source(tmp_path: Path, row: str, message: str) -> None:
    (tmp_path / "000000.txt").write_text(row + "\n")
    with pytest.raises(ValueError, match=f"000000.txt:1:.*{message}"):
        read_kitti_object_labels(tmp_path, frame_count=1)


@pytest.mark.parametrize("names", (("000000.txt",), ("0.txt", "1.txt"), ("000000.txt", "000002.txt")))
def test_object_labels_require_exact_contiguous_six_digit_frame_files(tmp_path: Path, names: tuple[str, ...]) -> None:
    for name in names:
        (tmp_path / name).write_text(_CAR + "\n")
    with pytest.raises(ValueError, match="align to every sequence image"):
        read_kitti_object_labels(tmp_path, frame_count=2)


@pytest.mark.parametrize("frame_count", (0, -1, True, 1.5, 1_000_001))
def test_object_labels_require_a_valid_timeline(tmp_path: Path, frame_count: int) -> None:
    with pytest.raises(ValueError, match="positive frame count"):
        read_kitti_object_labels(tmp_path, frame_count=frame_count)
