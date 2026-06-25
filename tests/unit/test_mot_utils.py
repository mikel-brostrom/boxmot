import numpy as np

from boxmot.engine.tracking.mot import convert_to_mmot_obb_format, xywha_to_corners


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
