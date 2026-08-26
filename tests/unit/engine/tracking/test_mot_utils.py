from types import SimpleNamespace

import numpy as np
import pytest

from boxmot.engine.tracking.mot import (
    convert_to_mmot_obb_format,
    convert_to_mot_format,
    format_frame_tagged_tracks_for_mot,
    write_mot_results,
    xywha_to_corners,
)
from boxmot.engine.tracking.runtime import TrackerRuntime
from boxmot.trackers.results import TrackResults


def test_xywha_to_corners_canonicalizes_equivalent_obb_forms():
    base = np.array([640.0, 512.0, 320.0, 160.0, 0.45], dtype=np.float32)
    equivalent = np.array([640.0, 512.0, 160.0, 320.0, 0.45 + (np.pi / 2.0)], dtype=np.float32)

    np.testing.assert_allclose(xywha_to_corners(base), xywha_to_corners(equivalent), atol=1e-4)


def test_convert_to_mmot_obb_format_matches_equivalent_obb_forms():
    base = np.array([[640.0, 512.0, 320.0, 160.0, 0.45, 3.0, 0.9, 4.0, 7.0]], dtype=np.float32)
    equivalent = np.array([[640.0, 512.0, 160.0, 320.0, 0.45 + (np.pi / 2.0), 3.0, 0.9, 4.0, 7.0]], dtype=np.float32)

    np.testing.assert_allclose(
        convert_to_mmot_obb_format(base, frame_idx=12),
        convert_to_mmot_obb_format(equivalent, frame_idx=12),
        atol=1e-4,
    )


def test_track_results_save_mot_matches_canonical_aabb_rounding(tmp_path):
    tracks = np.array([[-1.6, 2.6, 11.7, 22.8, 3, 0.9, 4, 7]], dtype=np.float32)
    output_path = tmp_path / "tracks.txt"

    TrackResults(tracks).save_mot(output_path, frame_id=5)

    saved = np.loadtxt(output_path, delimiter=",", ndmin=2)
    expected = convert_to_mot_format(tracks, frame_idx=5)
    np.testing.assert_allclose(saved, expected, atol=1e-6)


def test_frame_tagged_export_rejects_fractional_frame_ids():
    row = np.array([[1.000001, 10, 20, 30, 40, 7, 0.9, 2, -1]], dtype=np.float64)

    with pytest.raises(ValueError, match="finite integers"):
        format_frame_tagged_tracks_for_mot(row)


def test_mot_writer_rejects_noncanonical_obb_rows(tmp_path):
    raw_frame_tagged_obb = np.zeros((1, 10), dtype=np.float32)

    with pytest.raises(ValueError, match="9-column AABB or 13-column MMOT"):
        write_mot_results(tmp_path / "tracks.txt", raw_frame_tagged_obb)


def test_empty_aabb_and_obb_exports_preserve_canonical_widths():
    aabb = TrackResults(np.empty((0, 8), dtype=np.float32))
    obb = TrackResults(np.empty((0, 9), dtype=np.float32))

    assert convert_to_mot_format(aabb, frame_idx=3).shape == (0, 9)
    assert convert_to_mmot_obb_format(obb, frame_idx=3).shape == (0, 13)
    assert TrackerRuntime.format_for_mot(aabb, frame_idx=3).shape == (0, 9)
    assert TrackerRuntime.format_for_mot(obb, frame_idx=3).shape == (0, 13)
    assert format_frame_tagged_tracks_for_mot(np.empty((0, 9), dtype=np.float32)).shape == (0, 9)
    assert format_frame_tagged_tracks_for_mot(np.empty((0, 10), dtype=np.float32)).shape == (0, 13)


def test_aabb_object_and_array_exports_use_the_same_frame_index():
    rows = np.array([[10, 20, 30, 50, 7, 0.9, 2, 4]], dtype=np.float32)

    class Boxes(SimpleNamespace):
        def __len__(self):
            return len(self.id)

    boxes = Boxes(
        xyxy=rows[:, :4],
        id=rows[:, 4],
        conf=rows[:, 5],
        cls=rows[:, 6],
        det_ind=rows[:, 7],
    )

    np.testing.assert_array_equal(
        convert_to_mot_format(SimpleNamespace(boxes=boxes), frame_idx=12),
        convert_to_mot_format(rows, frame_idx=12),
    )


def test_cross_mode_exporters_reject_tracker_rows():
    with pytest.raises(ValueError, match="AABB MOT export"):
        convert_to_mot_format(np.empty((0, 9), dtype=np.float32), frame_idx=1)
    with pytest.raises(ValueError, match="OBB MMOT export"):
        convert_to_mmot_obb_format(np.empty((0, 8), dtype=np.float32), frame_idx=1)
