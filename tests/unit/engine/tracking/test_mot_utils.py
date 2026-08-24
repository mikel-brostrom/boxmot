import numpy as np
import pytest

from boxmot.engine.tracking.mot import (
    convert_to_mmot_obb_format,
    convert_to_mot_format,
    format_frame_tagged_tracks_for_mot,
    write_mot_results,
    xywha_to_corners,
)
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
