from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from boxmot.engine.calibration import ground_truth_noise
from boxmot.trackers.common.geometry.obb import xywha_to_corners


def _mmot_row(frame_id: int, track_id: int, box: np.ndarray, class_id: int) -> list[float]:
    corners = xywha_to_corners(np.asarray(box, dtype=np.float32)).tolist()
    return [frame_id, track_id, *corners, -1, class_id, 1]


def _write_mmot_annotations(
    root: Path,
    *,
    sequence_name: str = "sequence",
    frames: int = 5,
) -> tuple[Path, Path]:
    sequence_root = root / "sequences"
    annotations_root = root / "annotations"
    (sequence_root / sequence_name).mkdir(parents=True)
    annotations_root.mkdir()

    rows: list[list[float]] = []
    track_id = 1
    for class_id, class_x in ((0, 100.0), (7, 400.0)):
        for object_index in range(3):
            object_y = 100.0 + (object_index * 100.0)
            for frame_id in range(1, frames + 1):
                time = float(frame_id - 1)
                box = np.array(
                    [
                        class_x + (3.0 * time),
                        object_y + (0.5 * time),
                        40.0 + object_index,
                        14.0 + class_id * 0.1,
                        0.15 + (0.012 * time),
                    ],
                    dtype=float,
                )
                rows.append(_mmot_row(frame_id, track_id, box, class_id))
            track_id += 1

    np.savetxt(annotations_root / f"{sequence_name}.txt", np.asarray(rows), delimiter=",")
    return sequence_root, annotations_root


def test_mmot_corner_conversion_preserves_metadata_and_uses_enclosing_aabb() -> None:
    boxes = np.array(
        [
            [100.0, 80.0, 40.0, 12.0, 0.3],
            [250.0, 160.0, 18.0, 50.0, -1.2],
        ],
        dtype=float,
    )
    raw = np.asarray(
        [
            _mmot_row(1, 1, boxes[0], 0),
            _mmot_row(1, 2, boxes[1], 7),
        ]
    )

    converted = ground_truth_noise._obb_to_cxywh(raw)

    assert converted.shape == (2, 9)
    assert converted[:, :2].astype(int).tolist() == [[1, 1], [1, 2]]
    assert converted[:, 7].astype(int).tolist() == [0, 7]
    for converted_row, box in zip(converted, boxes):
        corners = xywha_to_corners(box).reshape(4, 2)
        expected = np.array(
            [
                corners[:, 0].min(),
                corners[:, 1].min(),
                np.ptp(corners[:, 0]),
                np.ptp(corners[:, 1]),
            ]
        )
        np.testing.assert_allclose(converted_row[2:6], expected, rtol=1e-5, atol=1e-4)


def test_load_gt_data_reads_external_mmot_annotations(tmp_path: Path) -> None:
    sequence_root, annotations_root = _write_mmot_annotations(tmp_path, frames=3)

    loaded = ground_truth_noise.load_gt_data(sequence_root / "sequence", annotations_root)

    assert loaded.shape == (18, 9)
    assert set(loaded[:, 7].astype(int)) == {0, 7}
    assert np.all(loaded[:, 4:6] > 0)


def test_mmot_tracks_feed_the_current_four_dimensional_kf_calibration(tmp_path: Path) -> None:
    sequence_root, annotations_root = _write_mmot_annotations(tmp_path)

    tracks, widths, heights = ground_truth_noise.build_tracks_from_sequence(
        sequence_root / "sequence",
        annotations_dir=annotations_root,
        min_detections=5,
    )

    assert len(tracks) == 6
    assert {class_id for _measurements, _states, class_id in tracks} == {0, 7}
    assert all(measurements.shape == (5, 4) for measurements, _states, _class_id in tracks)
    assert all(states.shape == (5, 8) for _measurements, states, _class_id in tracks)
    assert widths.shape == heights.shape == (30,)
    assert np.all(widths > 0)
    assert np.all(heights > 0)


def test_process_noise_is_zero_for_constant_velocity_tracks() -> None:
    measurements = np.array(
        [
            [0.0, 0.0, 10.0, 5.0],
            [1.0, 0.5, 10.0, 5.0],
            [2.0, 1.0, 10.0, 5.0],
            [3.0, 1.5, 10.0, 5.0],
        ]
    )
    velocity = np.vstack([np.zeros(4), np.diff(measurements, axis=0)])
    states = np.hstack([measurements, velocity])

    q_position, q_velocity = ground_truth_noise._estimate_process_noise([(measurements, states, 0)])

    np.testing.assert_allclose(q_position, np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(q_velocity, np.zeros(4), atol=1e-12)


def test_kf_parameterization_metadata_remains_explicit() -> None:
    assert ground_truth_noise.tracker_kf_type("botsort") == "xywh"
    assert ground_truth_noise.tracker_kf_type("sfsort") is None
    assert ground_truth_noise._measurement_labels("xywh") == ["cx", "cy", "w", "h"]
    assert ground_truth_noise._get_dim_x("xysr") == 7
    assert ground_truth_noise._get_dim_z("xysr") == 4


def test_kf_calibration_has_no_positional_detection_cache_path() -> None:
    parameters = inspect.signature(ground_truth_noise.estimate_kf_noise).parameters

    assert "dets_root" not in parameters
    assert "iou_threshold" not in parameters
    assert not hasattr(ground_truth_noise, "estimate_R_from_detections")
    assert not hasattr(ground_truth_noise, "run_kf_tuning")
