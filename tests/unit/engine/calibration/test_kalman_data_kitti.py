"""2D KF calibration consumes the same native KITTI image timeline as evaluation."""

from __future__ import annotations

import hashlib
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from boxmot.engine.calibration import kalman_data
from boxmot.engine.calibration.kalman_data import load_calibration_data
from tests.unit.engine.calibration.test_kalman_data import _fixture


def _kitti_fixture(tmp_path: Path) -> Namespace:
    """Reuse canonical detector outputs with sampled GT and neighboring large IDs."""
    args = _fixture(tmp_path)
    path = args.gt_folder / "native-labels.txt"
    rows = []
    timestamps = {0: 0.0, 1: 0.1, 4: 0.4, 6: 0.5}
    for frame in range(8):
        left = 20.0 + timestamps.get(frame, 100.0)
        for identity, label, truncation, occlusion, offset in (
            ((1 << 53) + 1, "Car", 0, 0, 0),
            ((1 << 53) + 2, "Car", 0, 2, 100),
            (3, "Pedestrian", 0, 2, 0),
            (4, "Car", 1, 0, 0),
            (5, "Pedestrian", 0, 3, 0),
            (6, "Van", 0, 0, 0),
            (7, "Person", 0, 0, 0),
            (-1, "DontCare", -1, -1, 0),
        ):
            if frame == 4 and identity == (1 << 53) + 1:
                continue
            rows.append(
                f"{frame} {identity} {label} {truncation} {occlusion} -10 "
                f"{left + offset} 20 {left + offset + 20} 60 -1 -1 -1 -1000 -1000 -1000 -10"
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    args.remapped_class_ids = [1, 2]
    args.remapped_class_names = ["car", "pedestrian"]
    args.evaluation_config = {
        "layout": "sequence",
        "box_type": "aabb",
        "classes": {"car": {"id": 1, "evaluation": "target"}, "pedestrian": {"id": 2, "evaluation": "target"}},
        "annotation_layout": "kitti_tracking",
        "kitti_gt_sequences": {
            "seq": {"path": str(path), "frame_count": 8, "frames": [(0, 0), (1, 1), (2, 4), (3, 6)]}
        },
    }
    return args


def test_calibration_preserves_sampled_timing_visibility_and_distinct_large_identities(tmp_path: Path) -> None:
    args = _kitti_fixture(tmp_path)

    data = load_calibration_data(args)

    cars = [track for track in data.tracks if track.class_id == 1]
    pedestrians = [track for track in data.tracks if track.class_id == 2]
    assert len(cars) == 2
    assert len({track.track_id for track in cars}) == 2
    assert len(pedestrians) == 1
    near_car = next(track for track in cars if track.gt_boxes[0, 0] < 100)
    far_car = next(track for track in cars if track.gt_boxes[0, 0] > 100)
    np.testing.assert_array_equal(near_car.frame_indices, [0, 1, 3])
    np.testing.assert_allclose(near_car.timestamps_s, [0.0, 0.1, 0.5])
    np.testing.assert_allclose(near_car.gt_boxes, near_car.detection_boxes, atol=1e-5)
    np.testing.assert_array_equal(far_car.frame_indices, [0, 1, 2, 3])
    assert np.isnan(far_car.detection_boxes).all()
    assert np.isnan(pedestrians[0].detection_boxes[2]).all()
    assert data.statistics["ground_truth_rows"] == 31
    assert data.statistics["ground_truth"] == 11
    assert data.statistics["filtered_ground_truth"] == 20
    assert data.statistics["matched"] == 6
    path = Path(args.evaluation_config["kitti_gt_sequences"]["seq"]["path"])
    assert data.ground_truth_sources == (
        {"sequence_id": "seq", "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    )


def test_cached_kitti_calibration_reuses_labels_without_losing_trajectories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _kitti_fixture(tmp_path)
    plain = load_calibration_data(args)
    args.cache_inputs = True
    cold = load_calibration_data(args)

    def unexpected(*_args, **_kwargs):
        pytest.fail("Warm KITTI calibration must reuse parsed annotation arrays.")

    monkeypatch.setattr(kalman_data, "_read_kitti_ground_truth", unexpected)
    warm = load_calibration_data(args)
    for actual in (cold, warm):
        assert actual.statistics == plain.statistics
        assert actual.ground_truth_sources == plain.ground_truth_sources
        for track, expected in zip(actual.tracks, plain.tracks, strict=True):
            assert (track.class_id, track.track_id) == (expected.class_id, expected.track_id)
            for field in ("frame_indices", "timestamps_s", "gt_boxes", "detection_boxes", "scores"):
                np.testing.assert_array_equal(getattr(track, field), getattr(expected, field))


def test_cached_kitti_calibration_keys_frame_mapping_and_exact_annotation_bytes(tmp_path: Path) -> None:
    args = _kitti_fixture(tmp_path)
    args.cache_inputs = True
    first = load_calibration_data(args)
    specification = args.evaluation_config["kitti_gt_sequences"]["seq"]
    specification["frames"][-1] = (3, 7)

    resampled = load_calibration_data(args)

    assert resampled.ground_truth_sources == first.ground_truth_sources
    assert resampled.statistics["matched"] == first.statistics["matched"] - 2
    assert all(track.gt_boxes[-1, 0] >= 100 for track in resampled.tracks)
    assert [(track.class_id, track.track_id) for track in resampled.tracks] == [
        (track.class_id, track.track_id) for track in first.tracks
    ]
    path = Path(specification["path"])
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    changed = load_calibration_data(args)
    assert changed.ground_truth_sources != resampled.ground_truth_sources
    assert changed.statistics == resampled.statistics


def test_kitti_calibration_respects_requested_target_classes(tmp_path: Path) -> None:
    args = _kitti_fixture(tmp_path)
    args.classes = [2]

    data = load_calibration_data(args)

    assert [track.class_id for track in data.tracks] == [2]
    assert data.statistics["matched"] == 3


@pytest.mark.parametrize(
    "frames",
    [
        [(0, 0), (1, 1), (2, 4)],
        [(0, 0), (1, 1), (2, 4), (3, 4)],
        [(0, 0), (1, 1), (2, 4), (3, 8)],
        [(0, 0), (1, 1), (2, 4), (4, 6)],
        [(0, 0), (1, 4), (2, 1), (3, 6)],
    ],
)
def test_kitti_calibration_rejects_misaligned_source_timeline(tmp_path: Path, frames: list[tuple[int, int]]) -> None:
    args = _kitti_fixture(tmp_path)
    args.evaluation_config["kitti_gt_sequences"]["seq"]["frames"] = frames

    with pytest.raises(ValueError, match="aligned, increasing source frames"):
        load_calibration_data(args)


def test_kitti_calibration_rejects_spatial_or_oriented_geometry(tmp_path: Path) -> None:
    args = _kitti_fixture(tmp_path)
    args.geometry = "obb"

    with pytest.raises(ValueError, match="KITTI tracking calibration requires AABB"):
        load_calibration_data(args)
