from __future__ import annotations

import numpy as np
import pytest

from boxmot.native.trackers import bytetrack as native_module
from boxmot.trackers.bbox.bytetrack import ByteTrack


@pytest.mark.parametrize("is_obb", [False, True], ids=["aabb", "obb"])
def test_native_bytetrack_kalman_prediction_matches_python(is_obb):
    """Use the prior box, as Python does, to scale prediction process noise."""
    cfg = {
        "min_conf": 0.1,
        "track_thresh": 0.6,
        "match_thresh": 0.9,
        "track_buffer": 30,
        "frame_rate": 30,
    }
    python_tracker = ByteTrack(**cfg)
    library = native_module._ByteTrackLiveLibrary(native_module.ensure_bytetrack_cpp_library())
    native_tracker = native_module.NativeByteTrackTracker(cfg, library=library)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    geometry_cols = 5 if is_obb else 4

    try:
        for frame_id in range(1, 9):
            if is_obb:
                detections = np.array(
                    [
                        [
                            50 + 3 * frame_id,
                            60 + frame_id,
                            30 + 0.5 * frame_id,
                            20 + 0.2 * frame_id,
                            0.1 * frame_id,
                            0.95,
                            0,
                        ]
                    ],
                    dtype=np.float32,
                )
            else:
                detections = np.array(
                    [
                        [
                            40 + 3 * frame_id,
                            50 + frame_id,
                            70 + 4 * frame_id,
                            70 + 1.5 * frame_id,
                            0.95,
                            0,
                        ]
                    ],
                    dtype=np.float32,
                )

            python_output = np.asarray(python_tracker.update(detections, image))
            native_output = np.asarray(native_tracker.update(detections))

            assert python_output.shape == native_output.shape == (1, geometry_cols + 4)
            np.testing.assert_allclose(
                native_output[:, :geometry_cols],
                python_output[:, :geometry_cols],
                atol=1e-5,
            )
            # Numeric ID labels may use different starting offsets without
            # changing tracking metrics; confidence, class, and det index must
            # still be identical.
            np.testing.assert_array_equal(
                native_output[:, geometry_cols + 1 :],
                python_output[:, geometry_cols + 1 :],
            )
    finally:
        native_tracker.close()
