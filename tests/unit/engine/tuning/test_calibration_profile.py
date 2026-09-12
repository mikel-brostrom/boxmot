"""Resuming calibrated tuning restores fixed priors without calibration or search."""

from __future__ import annotations

import builtins
import json
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from boxmot.engine.config.trackers import resolve_tracker_options
from boxmot.engine.tuning.calibration_profile import CALIBRATED_KF_OPTIONS, load_tuning_calibration
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_NOISE_OPTIONS


def _saved_run(tmp_path: Path, *, tracker: str = "botsort") -> tuple[Namespace, Path, dict, dict]:
    """Create the two actual serialized artifacts of a calibrated tuning run."""
    args = Namespace(
        tracker=tracker,
        tracker_backend="python",
        tracker_config=None,
        variable_dt=True,
        geometry="aabb",
        dataset_id="fixture",
        split="train",
        build_path=tmp_path / "build",
        per_class=False,
        tracker_class_ids=(1, 3),
        sequence_names=None,
        seq_info={"seq-a": 10, "seq-b": 12},
    )
    config = resolve_tracker_options(args, include_defaults=True, stamp_timing=True)
    config.update({name: (index + 1) / 10.0 for index, name in enumerate(KALMAN_NOISE_OPTIONS)})
    fixed = {name: config[name] for name in CALIBRATED_KF_OPTIONS if name in config}
    report = {
        "version": 2,
        "status": "complete",
        "method": "supervised_covariance_moments",
        "tracker": tracker,
        "geometry": "aabb",
        "dataset": "fixture",
        "split": "train",
        "build": str(args.build_path),
        "per_class": False,
        "class_ids": [1, 3],
        "sequences": ["seq-a", "seq-b"],
        "tuning": {"fixed_options": fixed},
    }
    directory = tmp_path / "tuning"
    calibration = directory / "kf-tuning"
    calibration.mkdir(parents=True)
    (calibration / "calibrated.yaml").write_text(yaml.safe_dump({"tracker": tracker, **config}))
    (calibration / "calibration.json").write_text(json.dumps(report))
    args.variable_dt = None
    return args, directory, config, report


def _write_report(directory: Path, report: dict) -> None:
    """Replace only report metadata to exercise resume guards."""
    (directory / "kf-tuning/calibration.json").write_text(json.dumps(report))


def test_no_calibration_artifacts_returns_none(tmp_path):
    assert load_tuning_calibration(Namespace(), tmp_path) is None


def test_calibrated_resume_cannot_switch_to_a_native_backend(tmp_path):
    args, directory, _, _ = _saved_run(tmp_path)
    args.tracker_backend = "cpp"
    with pytest.raises(ValueError, match="Python tracker backend"):
        load_tuning_calibration(args, directory)


def test_seconds_profile_restores_without_a_timing_flag_or_heavy_dependencies(tmp_path, monkeypatch):
    args, directory, saved, report = _saved_run(tmp_path)
    before = vars(args).copy()
    original_import = builtins.__import__

    def reject_search_or_calibration(name, *args, **kwargs):
        assert name.split(".")[0] not in {"ray", "optuna"}
        assert name not in {"boxmot.engine.calibration.kalman", "boxmot.engine.calibration.kalman_data"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_search_or_calibration)
    config, fixed = load_tuning_calibration(args, directory)
    assert config == saved
    assert config["variable_dt"] is True
    assert config["kf_time_unit"] == "seconds"
    assert fixed == report["tuning"]["fixed_options"]
    assert vars(args) == before


def test_non_kf_programmatic_override_can_change_the_tracker_baseline(tmp_path):
    args, directory, saved, _ = _saved_run(tmp_path)
    config, fixed = load_tuning_calibration(args, directory, overrides={"track_high_thresh": 0.77})
    assert config["track_high_thresh"] == 0.77
    assert all(config[name] == saved[name] == value for name, value in fixed.items())


def test_explicit_tracker_config_can_change_non_kf_settings(tmp_path):
    args, directory, saved, _ = _saved_run(tmp_path)
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(yaml.safe_dump({**saved, "track_high_thresh": 0.88}))
    args.tracker_config = explicit
    config, _ = load_tuning_calibration(args, directory)
    assert config["track_high_thresh"] == 0.88


def test_explicit_tracker_config_cannot_reset_calibrated_scales_to_defaults(tmp_path):
    args, directory, _, _ = _saved_run(tmp_path)
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(yaml.safe_dump({"variable_dt": True, "track_high_thresh": 0.88}))
    args.tracker_config = explicit
    with pytest.raises(ValueError, match="Cannot change fixed KF calibration"):
        load_tuning_calibration(args, directory)


@pytest.mark.parametrize("name", KALMAN_NOISE_OPTIONS)
def test_changed_calibrated_scale_override_is_rejected(tmp_path, name):
    args, directory, saved, _ = _saved_run(tmp_path)
    with pytest.raises(ValueError, match=name):
        load_tuning_calibration(args, directory, overrides={name: saved[name] * 2})


def test_explicit_timing_flag_uses_normal_time_unit_validation(tmp_path):
    args, directory, _, _ = _saved_run(tmp_path)
    args.variable_dt = False
    with pytest.raises(ValueError, match="kf_time_unit"):
        load_tuning_calibration(args, directory)


@pytest.mark.parametrize("override", [{"kf_reference_dt_s": 0.1}, {"variable_dt": False, "kf_time_unit": "frames"}])
def test_changed_calibrated_time_basis_is_rejected(tmp_path, override):
    args, directory, _, _ = _saved_run(tmp_path)
    with pytest.raises(ValueError, match="Cannot change fixed KF calibration"):
        load_tuning_calibration(args, directory, overrides=override)


@pytest.mark.parametrize("tracker", ["boosttrack", "occluboost"])
def test_adaptive_kalman_mode_stays_fixed_after_calibration(tmp_path, tracker):
    args, directory, saved, _ = _saved_run(tmp_path, tracker=tracker)
    _, fixed = load_tuning_calibration(args, directory)
    name = "adaptive_kf"
    assert name in fixed
    with pytest.raises(ValueError, match=name):
        load_tuning_calibration(args, directory, overrides={name: not saved[name]})


@pytest.mark.parametrize(
    "key,value",
    [
        ("tracker", "ocsort"),
        ("geometry", "obb"),
        ("dataset", "different"),
        ("split", "test"),
        ("build", "/different/build"),
        ("per_class", True),
        ("class_ids", [1]),
        ("sequences", ["seq-a"]),
    ],
)
def test_changed_calibration_metadata_rejects_resume(tmp_path, key, value):
    args, directory, _, report = _saved_run(tmp_path)
    report[key] = value
    _write_report(directory, report)
    with pytest.raises(ValueError, match=key):
        load_tuning_calibration(args, directory)


def test_equivalent_sequence_class_order_and_resolved_build_path_are_accepted(tmp_path):
    args, directory, _, report = _saved_run(tmp_path)
    report.update(class_ids=[3, 1], sequences=["seq-b", "seq-a"], build=str(args.build_path / ".." / "build"))
    _write_report(directory, report)
    assert load_tuning_calibration(args, directory) is not None


@pytest.mark.parametrize("missing", ["calibrated.yaml", "calibration.json"])
def test_incomplete_artifact_pair_is_rejected(tmp_path, missing):
    args, directory, _, _ = _saved_run(tmp_path)
    (directory / "kf-tuning" / missing).unlink()
    with pytest.raises(ValueError, match="incomplete"):
        load_tuning_calibration(args, directory)


@pytest.mark.parametrize("status", ["running", "error", None])
def test_incomplete_report_is_rejected(tmp_path, status):
    args, directory, _, report = _saved_run(tmp_path)
    report["status"] = status
    _write_report(directory, report)
    with pytest.raises(ValueError, match="not complete"):
        load_tuning_calibration(args, directory)


@pytest.mark.parametrize("malformation", ["missing_marker", "missing_prior", "extra_prior", "disagreeing_prior"])
def test_missing_or_inconsistent_fixed_marker_is_rejected(tmp_path, malformation):
    args, directory, _, report = _saved_run(tmp_path)
    if malformation == "missing_marker":
        report.pop("tuning")
    elif malformation == "missing_prior":
        report["tuning"]["fixed_options"].pop("kf_measurement_noise_scale")
    elif malformation == "extra_prior":
        report["tuning"]["fixed_options"]["unknown_setting"] = 1.0
    else:
        report["tuning"]["fixed_options"]["kf_measurement_noise_scale"] = 100.0
    _write_report(directory, report)
    with pytest.raises(ValueError, match="fixed_options|disagrees"):
        load_tuning_calibration(args, directory)


def test_profile_missing_explicit_calibrated_values_is_rejected(tmp_path):
    args, directory, _, _ = _saved_run(tmp_path)
    (directory / "kf-tuning/calibrated.yaml").write_text("tracker: botsort\n")
    with pytest.raises(ValueError, match="profile is incomplete"):
        load_tuning_calibration(args, directory)
