"""Native 2D tracking labels preserve evaluator metadata without demanding 3D data."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from boxmot.datasets.readers import boxes2d
from boxmot.datasets.readers.boxes2d import read_kitti_tracking_labels_2d

_CAR = "0 7 Car 0 0 -10 20 30 80 90 -1 -1 -1 -1000 -1000 -1000 -10"
_DONTCARE = "0 -1 DontCare -1 -1 -10 100 30 150 90 -1 -1 -1 -1000 -1000 -1000 -10"


def test_tracking_annotations_preserve_native_ignored_rows_and_missing_spatial_geometry(tmp_path: Path) -> None:
    rows = (
        _CAR,
        "0 8 Van 2 3 -10 -10 30 80 90 -1 -1 -1 -1000 -1000 -1000 -10",
        "2 9 Person 1 2 -10 20 30 80 90 -1 -1 -1 -1000 -1000 -1000 -10",
        _DONTCARE,
        _DONTCARE.replace("100 30 150 90", "200 30 250 90"),
    )
    path = tmp_path / "0000.txt"
    payload = ("\ufeff\n" + "\n\n".join(rows) + "\n").encode("utf-8")
    path.write_bytes(payload)

    annotations = read_kitti_tracking_labels_2d(path, frame_count=3)

    assert annotations.source_rows == rows
    assert annotations.source_sha256 == hashlib.sha256(payload).hexdigest()


def test_empty_annotation_file_describes_frames_without_objects(tmp_path: Path) -> None:
    path = tmp_path / "0000.txt"
    path.write_text("\n", encoding="utf-8")

    annotations = read_kitti_tracking_labels_2d(path, frame_count=2)

    assert annotations.source_rows == ()


@pytest.mark.parametrize(
    "column,value,message",
    [
        (0, "-1", "frame index"),
        (0, "3", "frame index"),
        (0, "0.0", "must be integers"),
        (1, "-1", "track identities"),
        (1, str(1 << 63), "track identities"),
        (1, "2.5", "must be integers"),
        (2, "Cär", "native ASCII"),
        (3, "0.2", "must be integers"),
        (3, "-1", "tracking truncation"),
        (3, "3", "tracking truncation"),
        (4, "4", "tracking occlusion"),
        (4, "-1", "tracking occlusion"),
        (5, "NaN", "finite ASCII decimals"),
        (6, "1_000", "finite ASCII decimals"),
        (8, "20", "positive width and height"),
        (9, "29", "positive width and height"),
        (10, "inf", "finite ASCII decimals"),
    ],
)
def test_invalid_image_annotations_fail_with_source_location(
    tmp_path: Path, column: int, value: str, message: str
) -> None:
    fields = _CAR.split()
    fields[column] = value
    path = tmp_path / "labels.txt"
    path.write_text("\n" + " ".join(fields), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"Invalid 2D tracking ground truth at .*labels.txt:2: .*{message}"):
        read_kitti_tracking_labels_2d(path, frame_count=3)


@pytest.mark.parametrize("row", [_CAR + " 0.9", " ".join(_CAR.split()[2:])])
def test_detections_and_object_labels_cannot_substitute_for_tracking_ground_truth(tmp_path: Path, row: str) -> None:
    path = tmp_path / "labels.txt"
    path.write_text(row, encoding="utf-8")

    with pytest.raises(ValueError, match="17 KITTI tracking label fields"):
        read_kitti_tracking_labels_2d(path, frame_count=1)


def test_repeated_identity_within_frame_is_rejected_across_classes(tmp_path: Path) -> None:
    path = tmp_path / "labels.txt"
    path.write_text(_CAR + "\n" + _CAR.replace("Car", "Pedestrian"), encoding="utf-8")

    with pytest.raises(ValueError, match="repeated track identity in frame 0"):
        read_kitti_tracking_labels_2d(path, frame_count=1)


@pytest.mark.parametrize("frame_count", [True, 0, -1, 1.5])
def test_frame_count_must_define_an_image_timeline(tmp_path: Path, frame_count: object) -> None:
    with pytest.raises(ValueError, match="positive integer frame count"):
        read_kitti_tracking_labels_2d(tmp_path / "labels.txt", frame_count=frame_count)


def test_cached_annotations_preserve_raw_metadata_without_reparsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.txt"
    path.write_text(_CAR + "\n" + _DONTCARE + "\n", encoding="utf-8")
    expected = read_kitti_tracking_labels_2d(path, frame_count=2)
    cache_root = tmp_path / "cache"
    cold = read_kitti_tracking_labels_2d(path, frame_count=2, cache_inputs=True, cache_root=cache_root)

    def unexpected(*_args, **_kwargs):
        pytest.fail("Warm annotation caches must not reparse native KITTI rows.")

    monkeypatch.setattr(boxes2d, "_read_kitti_tracking_labels_2d", unexpected)
    warm = read_kitti_tracking_labels_2d(path, frame_count=2, cache_inputs=True, cache_root=cache_root)

    assert cold == warm == expected
    assert len(list(cache_root.glob("*/values.npy"))) == 1


def test_cached_annotations_revalidate_source_frame_count_and_cache_mutations(tmp_path: Path) -> None:
    path = tmp_path / "labels.txt"
    path.write_text(_CAR + "\n", encoding="utf-8")
    cache_root = tmp_path / "cache"
    initial = read_kitti_tracking_labels_2d(path, frame_count=2, cache_inputs=True, cache_root=cache_root)
    next(cache_root.glob("*/values.npy")).write_bytes(b"interrupted or corrupt cache")

    rebuilt = read_kitti_tracking_labels_2d(path, frame_count=2, cache_inputs=True, cache_root=cache_root)

    assert rebuilt == initial
    path.write_text(_CAR + "\n" + _DONTCARE.replace("0 -1", "1 -1") + "\n", encoding="utf-8")
    changed = read_kitti_tracking_labels_2d(path, frame_count=2, cache_inputs=True, cache_root=cache_root)
    assert changed.source_rows == (_CAR, _DONTCARE.replace("0 -1", "1 -1"))
    assert changed.source_sha256 != initial.source_sha256
    with pytest.raises(ValueError, match="frame index must be between 0 and 0"):
        read_kitti_tracking_labels_2d(path, frame_count=1, cache_inputs=True, cache_root=cache_root)
    path.write_text(_CAR + "\n" + _DONTCARE.replace("100 30 150 90", "100 30 50 90") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive width and height"):
        read_kitti_tracking_labels_2d(path, frame_count=2, cache_inputs=True, cache_root=cache_root)
