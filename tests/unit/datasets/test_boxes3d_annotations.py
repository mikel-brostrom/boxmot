"""Identity-bearing 3D annotations remain distinct from detector outputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from boxmot.datasets.readers.boxes3d import read_kitti_tracking_labels

_CLASSES = {
    "vehicle": {"id": 7, "evaluation": "target"},
    "walker": {"id": 12, "evaluation": "target"},
    "untracked": {"id": 99, "evaluation": "ignore"},
}


def _row(frame: int = 0, identity: int = 3, label: str = "Vehicle") -> str:
    """Use unequal dimensions and coordinates to expose field-order mistakes."""
    return f"{frame} {identity} {label} 0 0 0 0 0 4 3 1.5 2 4 5 6 20 0.25\n"


def test_tracking_annotations_reorder_boxes_and_preserve_identity_gaps(tmp_path: Path) -> None:
    path = tmp_path / "drive.txt"
    path.write_text(_row(2, 0) + _row(0, 0) + _row(0, 3, "Pedestrian"))

    labels = read_kitti_tracking_labels(
        path, frame_count=3, classes=_CLASSES, options={"class_map": {"Pedestrian": "walker"}}
    )

    assert labels.frame_indices.tolist() == [2, 0, 0]
    assert labels.track_ids.tolist() == [0, 0, 3]
    assert labels.class_ids.tolist() == [7, 7, 12]
    np.testing.assert_array_equal(labels.boxes, [[5, 6, 20, 0.25, 4, 2, 1.5]] * 3)
    assert labels.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert labels.row_count == 3


def test_explicitly_ignored_annotations_may_have_no_3d_geometry(tmp_path: Path) -> None:
    path = tmp_path / "drive.txt"
    ignored = "0 -1 DontCare -1 -1 -10 0 0 4 3 -1 -1 -1 -1000 -1000 -1000 -10\n"
    path.write_text(ignored + ignored.replace("DontCare", "Untracked") + _row())

    labels = read_kitti_tracking_labels(path, frame_count=1, classes=_CLASSES, options={"ignore_classes": ["DontCare"]})

    assert labels.row_count == 3
    assert labels.class_ids.tolist() == [7]
    assert labels.track_ids.tolist() == [3]


@pytest.mark.parametrize(
    "row,message",
    [
        (_row().replace("0 3 Vehicle", "0.5 3 Vehicle"), "invalid literal"),
        (_row().replace("0 3 Vehicle", "0 3.5 Vehicle"), "invalid literal"),
        (_row(-1), "frame index"),
        (_row(3), "frame index"),
        (_row(identity=-1), "nonnegative int64"),
        (_row(identity=2**63), "nonnegative int64"),
        (_row(label="Unknown"), "unsupported 3D annotation class"),
        (_row().replace("1.5 2 4", "0 2 4"), "positive dimensions"),
        (_row().replace("1.5 2 4", "1e-100 2 4"), "positive in float32"),
        (_row().replace("5 6 20", "nan 6 20"), "must be finite"),
        (_row().replace("5 6 20", "1e308 6 20"), "finite float32"),
        (_row().strip() + " 0.9\n", "17 KITTI tracking label fields"),
        ("Vehicle -1 -1 0 0 0 4 3 1.5 2 4 5 6 20 0.25 0.9\n", "17 KITTI tracking label fields"),
        (_row() + _row(), "repeated class/identity"),
    ],
)
def test_tracking_annotation_errors_name_the_source(tmp_path: Path, row: str, message: str) -> None:
    path = tmp_path / "drive.txt"
    path.write_text(row)

    with pytest.raises(ValueError, match=f"drive.txt:[12]:.*{message}"):
        read_kitti_tracking_labels(path, frame_count=3, classes=_CLASSES)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"score_transform": "odds"}, "Unsupported kitti-tracking-labels options"),
        ({"coordinate_frame": "world"}, "coordinate_frame must be 'camera'"),
        ({"class_map": {"Car": "missing"}}, "Unknown configured 3D class name"),
        ({"class_map": {"Car": 123}}, "not a configured class ID"),
        ({"ignore_classes": "DontCare"}, "must be a list"),
    ],
)
def test_annotation_options_reject_detection_scores_and_undefined_classes(
    tmp_path: Path, options: dict[str, object], message: str
) -> None:
    path = tmp_path / "drive.txt"
    path.write_text(_row())

    with pytest.raises(ValueError, match=message):
        read_kitti_tracking_labels(path, frame_count=1, classes=_CLASSES, options=options)


def test_empty_ground_truth_has_canonical_empty_arrays(tmp_path: Path) -> None:
    path = tmp_path / "drive.txt"
    path.write_text("\n")

    labels = read_kitti_tracking_labels(path, frame_count=3, classes=_CLASSES)

    assert labels.boxes.shape == (0, 7)
    assert labels.track_ids.dtype == np.int64
    assert labels.row_count == 0
