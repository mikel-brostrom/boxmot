"""Calibrated YAML carries portable tracker and filter compatibility metadata."""

from __future__ import annotations

import pytest
import yaml

from boxmot.engine.calibration import kalman_sensor
from boxmot.engine.calibration.kalman import calibrate_kalman
from boxmot.engine.config.trackers import resolve_tracker_options
from boxmot.engine.eval.eagermot_kitti import load_kitti_profiles
from boxmot.trackers import create_tracker
from boxmot.trackers.common.config import load_tracker_config
from boxmot.trackers.common.motion.kalman_filters.config import KalmanConfig
from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig
from boxmot.trackers.common.motion.kalman_filters.profile import calibration_profile_signature
from tests.unit.engine.calibration.test_kalman import _args, _data, _load_fixture
from tests.unit.engine.calibration.test_kalman_sensor import _data as _sensor_data
from tests.unit.engine.calibration.test_kalman_sensor import _dataset


def test_moved_calibrated_profile_validates_factory_geometry_without_sidecar(monkeypatch, tmp_path):
    _load_fixture(monkeypatch, _data())
    result = calibrate_kalman(_args(tmp_path), output_dir=tmp_path)
    moved = tmp_path / "portable.yaml"
    result.config_path.rename(moved)
    result.report_path.unlink()
    options = load_tracker_config("bytetrack", moved)
    assert options["calibration.geometry"] == "aabb"
    assert options["calibration.filter"] == "xyah"
    assert options["calibration.dataset"] == "fixture"
    tracker = create_tracker("bytetrack", geometry="aabb", **options)
    assert tracker.is_obb is False
    with pytest.raises(ValueError, match="profile geometry=.*incompatible"):
        create_tracker("bytetrack", geometry="obb", **options)
    with pytest.raises(ValueError, match="profile tracker=.*incompatible"):
        create_tracker("botsort", geometry="aabb", **options)
    with pytest.raises(ValueError, match="Python tracker backend"):
        create_tracker("bytetrack", backend="cpp", **options)


def test_eval_options_reject_calibrated_geometry_change_before_replay(monkeypatch, tmp_path):
    _load_fixture(monkeypatch, _data(geometry="obb"))
    result = calibrate_kalman(_args(tmp_path, geometry="obb"), output_dir=tmp_path)
    with pytest.raises(ValueError, match="profile geometry=.*incompatible"):
        resolve_tracker_options(
            _args(tmp_path, geometry="aabb", tracker_config=result.config_path), include_defaults=True
        )
    options = resolve_tracker_options(
        _args(tmp_path, geometry="obb", tracker_config=result.config_path), include_defaults=True
    )
    assert options["calibration.geometry"] == "obb"


@pytest.mark.parametrize(
    "overrides,field",
    [
        ({"kalman.noise.reference_dt_s": 0.5}, "reference_dt_s"),
        ({"kalman.variable_dt": False, "kalman.noise.time_unit": "frames"}, "variable_dt"),
    ],
)
def test_saved_profile_binds_resolved_time_basis_in_factory_and_eval(monkeypatch, tmp_path, overrides, field):
    _load_fixture(monkeypatch, _data())
    result = calibrate_kalman(_args(tmp_path, variable_dt=True), output_dir=tmp_path)
    saved = load_tracker_config("bytetrack", result.config_path)
    assert saved["calibration.variable_dt"] is True
    assert saved["calibration.time_unit"] == "seconds"
    assert saved["calibration.reference_dt_s"] == pytest.approx(1 / 30)
    with pytest.raises(ValueError, match=f"profile {field}=.*incompatible"):
        create_tracker("bytetrack", **{**saved, **overrides})
    with pytest.raises(ValueError, match=f"profile {field}=.*incompatible"):
        resolve_tracker_options(
            _args(tmp_path, variable_dt=None, tracker_config=result.config_path), overrides, include_defaults=True
        )


def test_partial_manual_settings_resolve_signature_timing_without_false_rejection():
    signature = calibration_profile_signature("bytetrack", "aabb", {"kalman.variable_dt": True})
    assert signature["time_unit"] == "seconds"
    assert signature["reference_dt_s"] == pytest.approx(1 / 30)
    tracker = create_tracker("bytetrack", kalman=KalmanConfig(variable_dt=True), calibration=signature)
    assert tracker.kalman_noise_config.time_unit == "seconds"
    manual = create_tracker(
        "bytetrack", kalman=KalmanConfig(variable_dt=True, noise=KalmanNoiseConfig(reference_dt_s=0.5))
    )
    assert manual.kalman_noise_config.reference_dt_s == 0.5


@pytest.mark.parametrize("field,value", [("filter", "xywh"), ("state_dimensions", 9), ("measurement_dimensions", True)])
def test_factory_rejects_inconsistent_calibration_signature(field, value):
    signature = calibration_profile_signature("bytetrack", "aabb", {})
    signature[field] = value
    with pytest.raises(ValueError, match=f"profile {field}=.*incompatible"):
        create_tracker("bytetrack", calibration=signature)


def test_manual_noise_settings_do_not_claim_a_calibration_geometry(tmp_path):
    path = tmp_path / "manual.yaml"
    path.write_text("kalman: {noise: {measurement_noise_scale: 0.25}}\n")
    options = load_tracker_config("bytetrack", path)
    for geometry in ("aabb", "obb"):
        tracker = create_tracker("bytetrack", geometry=geometry, **options)
        assert tracker.kalman_noise_config.measurement_noise_scale == 0.25


def test_sensor_profile_binds_angular_state_and_class_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(kalman_sensor, "load_sensor_calibration_data", lambda *a, **kw: _sensor_data())
    profiles = load_kitti_profiles()
    profiles[2]["kalman.is_angular"] = True
    result = kalman_sensor.calibrate_sensor_kalman(_dataset(tmp_path), profiles, output_dir=tmp_path)
    saved = load_kitti_profiles(result.config_path)
    assert saved[1]["calibration.state_dimensions"] == 10
    assert saved[2]["calibration.state_dimensions"] == 11
    assert saved[2]["calibration.is_angular"] is True
    assert saved[2]["calibration.variable_dt"] is False
    assert saved[2]["calibration.time_unit"] == "frames"
    create_tracker("eagermot", **saved[2])
    with pytest.raises(ValueError, match="profile filter=.*incompatible"):
        create_tracker("eagermot", **{**saved[2], "kalman.is_angular": False})

    authored = yaml.safe_load(result.config_path.read_text())
    authored["pedestrian"]["kalman"]["is_angular"] = False
    result.config_path.write_text(yaml.safe_dump(authored))
    with pytest.raises(ValueError, match="profile filter=.*incompatible"):
        load_kitti_profiles(result.config_path)
    authored["pedestrian"]["kalman"]["is_angular"] = True
    authored["car"], authored["pedestrian"] = authored["pedestrian"], authored["car"]
    result.config_path.write_text(yaml.safe_dump(authored))
    with pytest.raises(ValueError, match="does not match class"):
        load_kitti_profiles(result.config_path)
