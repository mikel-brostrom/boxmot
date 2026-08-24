import numpy as np
import pytest

from boxmot.trackers.common.motion import MotionModelKind, create_motion_model


@pytest.mark.parametrize(
    ("kind", "is_obb", "dim_x", "dim_z"),
    (
        (MotionModelKind.XYAH, False, 8, 4),
        (MotionModelKind.XYAH, True, 10, 5),
        (MotionModelKind.XYWH, False, 8, 4),
        (MotionModelKind.XYWH, True, 10, 5),
        (MotionModelKind.XYSR, False, 7, 4),
        (MotionModelKind.XYSR, True, 9, 5),
        (MotionModelKind.XYHR, False, 8, 4),
        (MotionModelKind.XYHR, True, 10, 5),
        (MotionModelKind.XYSCR, False, 9, 5),
    ),
)
def test_motion_model_adapter_dimensions_and_filter(kind, is_obb, dim_x, dim_z):
    adapter = create_motion_model(kind, is_obb=is_obb, max_obs=7)
    measurement = np.ones(dim_z, dtype=float)

    kf = adapter.create_filter(measurement if kind is MotionModelKind.XYHR else None)

    assert adapter.dim_x == dim_x
    assert adapter.dim_z == dim_z
    assert adapter.is_obb is is_obb
    assert kf.dim_x == dim_x
    assert kf.dim_z == dim_z


@pytest.mark.parametrize(
    "kind",
    (
        MotionModelKind.XYAH,
        MotionModelKind.XYWH,
        MotionModelKind.XYSR,
        MotionModelKind.XYHR,
        MotionModelKind.XYSCR,
    ),
)
def test_motion_model_adapter_roundtrip_aabb(kind):
    adapter = create_motion_model(kind, is_obb=False)
    box = np.array([10.0, 20.0, 30.0, 70.0, 0.8], dtype=float)

    state = adapter.to_measurement(box, column=False)
    decoded = adapter.to_box(state, score=1.0 if kind is MotionModelKind.XYSCR else None)

    np.testing.assert_allclose(decoded[0, :4], box[:4], atol=1e-5)


@pytest.mark.parametrize(
    "kind",
    (
        MotionModelKind.XYAH,
        MotionModelKind.XYWH,
        MotionModelKind.XYSR,
        MotionModelKind.XYHR,
    ),
)
def test_motion_model_adapter_roundtrip_obb(kind):
    adapter = create_motion_model(kind, is_obb=True)
    box = np.array([32.0, 24.0, 20.0, 10.0, 0.25], dtype=float)

    state = adapter.to_measurement(box, column=False)
    decoded = adapter.to_box(state)

    np.testing.assert_allclose(decoded[0], box, atol=1e-5)


def test_xyscr_motion_model_rejects_obb():
    with pytest.raises(ValueError, match="XYSCR does not support OBB"):
        create_motion_model(MotionModelKind.XYSCR, is_obb=True)
