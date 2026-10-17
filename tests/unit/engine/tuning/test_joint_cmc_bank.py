"""Joint camera/KF choices remain bound through export, replay and resume."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from boxmot.components.resolution import component_options
from boxmot.engine.calibration import cmc, kalman, kalman_selection
from boxmot.engine.config.cmc import CMC_ANALYSIS_RULE, SequenceCMCPolicy, resolve_cmc_policy
from boxmot.engine.config.trackers import resolve_tracker_options
from boxmot.engine.tuning.calibration_bank import create_calibration_bank, load_calibration_bank
from boxmot.trackers.common.config import load_tracker_config, nest_tracker_options
from boxmot.trackers.common.motion.cmc.integration import resolve_cmc_min_conf
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES
from boxmot.trackers.common.motion.kalman_filters.profile import calibration_cmc_signature
from boxmot.trackers.common.specs import TrackerSpec
from tests.unit.engine.calibration.test_kalman import _args, _data, _load_fixture


def _joint_bank(monkeypatch, tmp_path, *, selected_cmc="auto", profile="measurement", tracker="bytetrack"):
    """Keep fitting and serialization real; isolate the camera and ranking inputs."""
    data = _data()
    _load_fixture(monkeypatch, data)
    warps = {"seq1": np.tile(np.eye(2, 3), (6, 1, 1))}
    fit_calls = []
    original_fit = kalman.calibrate_kalman

    def fit(*args, **kwargs):
        fit_calls.append(kwargs.get("camera_warps"))
        return original_fit(*args, **kwargs)

    def camera(args, baseline, *, output_dir, **kwargs):
        directory = output_dir
        directory.mkdir(parents=True)
        report, archive = directory / "analysis.json", directory / "warps.npz"
        report.write_text(json.dumps({"status": "complete", "mask_min_conf": baseline["cmc_min_conf"]}))
        np.savez(archive, **warps)
        return SimpleNamespace(
            warps=warps, policy=SequenceCMCPolicy(sequences=(("seq1", True),)), report_path=report, warps_path=archive
        )

    def select(data, **kwargs):
        assert kwargs["camera_candidates"]["auto"] is warps
        assert kwargs["camera_candidates"]["off"] is None
        candidate = {
            "method": "grouped_cross_validated_box_profiles",
            "selected_type": "xyah",
            "selected_profile": profile,
            "candidates": {"xyah": {"status": "valid", "selected_profile": profile, "mean_iou": 0.8}},
        }
        return {
            **candidate,
            "method": "grouped_cross_validated_cmc_box_profiles",
            "selected_cmc": selected_cmc,
            "cmc_candidates": {"off": candidate, "auto": candidate},
        }

    monkeypatch.setattr(cmc, "prepare_cmc_candidates", camera)
    monkeypatch.setattr(kalman, "calibrate_kalman", fit)
    monkeypatch.setattr(kalman_selection, "select_kalman_model", select)
    args = _args(tmp_path, tracker=tracker, tracker_class_ids=(0,))
    baseline = resolve_tracker_options(args, {"kalman.type": "xyah"}, include_defaults=True, stamp_timing=True)
    bank = create_calibration_bank(
        args, tmp_path, runtime_config=baseline, kinds=("xyah",), select_best=True, select_cmc=True
    )
    assert fit_calls == [warps if selected_cmc == "auto" else None]
    return args, bank


@pytest.mark.parametrize("selected_cmc", ["off", "auto"])
@pytest.mark.parametrize("profile", ["default", "motion", "measurement", "calibrated"])
@pytest.mark.parametrize("tracker", sorted(KALMAN_TRACKER_NAMES))
def test_joint_profile_roundtrip_and_automatic_policy_replay(monkeypatch, tmp_path, selected_cmc, profile, tracker):
    args, bank = _joint_bank(monkeypatch, tmp_path, selected_cmc=selected_cmc, profile=profile, tracker=tracker)
    assert bank.selected_cmc == selected_cmc
    config = bank.profiles["xyah"]
    defaults = load_tracker_config(tracker)
    threshold = resolve_cmc_min_conf(defaults)
    assert config["cmc_min_conf"] == threshold
    assert load_calibration_bank(args, tmp_path) == bank
    spec = TrackerSpec(tracker, options=component_options(config))
    policy = resolve_cmc_policy(args, spec)
    assert config["calibration.cmc_rule"] == CMC_ANALYSIS_RULE
    assert config["calibration.cmc_policy"] == selected_cmc
    assert all(config[name] == value for name, value in calibration_cmc_signature(defaults).items())
    assert policy == ("auto" if selected_cmc == "auto" else SequenceCMCPolicy())
    # Subsequent detector-threshold tuning does not change the camera mask.
    detector_threshold = "min_conf" if "min_conf" in defaults else "det_thresh"
    restored = load_calibration_bank(args, tmp_path, overrides={detector_threshold: 0.3})
    assert restored.profiles["xyah"][detector_threshold] == 0.3
    assert restored.profiles["xyah"]["cmc_min_conf"] == threshold


@pytest.mark.parametrize("tracker", sorted(KALMAN_TRACKER_NAMES))
def test_joint_resume_with_sparse_original_yaml_restores_generated_cmc_options(monkeypatch, tmp_path, tracker):
    """Default-on/ECC trackers must resume their generated off/SOF setup, not defaults."""
    args, bank = _joint_bank(monkeypatch, tmp_path, tracker=tracker)
    args.tracker_config = tmp_path / "original.yaml"
    args.tracker_config.write_text(yaml.safe_dump({"tracker": tracker, "kalman": {"type": "xyah"}}))
    assert load_calibration_bank(args, tmp_path) == bank


@pytest.mark.parametrize("artifact", ["cmc-policy.yaml", "cmc/analysis.json", "cmc/warps.npz"])
def test_resume_rejects_changed_camera_artifacts(monkeypatch, tmp_path, artifact):
    args, _ = _joint_bank(monkeypatch, tmp_path)
    path = tmp_path / "kf-tuning" / artifact
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="CMC .* artifact"):
        load_calibration_bank(args, tmp_path)


def test_resume_rejects_changed_camera_mask_threshold(monkeypatch, tmp_path):
    args, _ = _joint_bank(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="Cannot change KF calibration inputs"):
        load_calibration_bank(args, tmp_path, overrides={"cmc_min_conf": 0.4})


@pytest.mark.parametrize("overrides", [{"sof_target_width": 128}, {"sof_ransac_reproj_threshold": 1.5}])
def test_occluboost_joint_profile_binds_estimator_on_resume_and_eval(monkeypatch, tmp_path, overrides):
    """Changing the camera estimator also changes the evidence used to fit noise."""
    args, bank = _joint_bank(monkeypatch, tmp_path, tracker="occluboost")
    profile = bank.profiles["xyah"]
    assert profile["calibration.cmc_sof_target_width"] == 192
    assert profile["calibration.cmc_sof_ransac_reproj_threshold"] == 3.0
    with pytest.raises(ValueError, match="Cannot change KF calibration inputs.*sof_"):
        load_calibration_bank(args, tmp_path, overrides=overrides)
    spec = TrackerSpec("occluboost", options=component_options({**profile, **overrides}))
    with pytest.raises(ValueError, match="SOF estimator settings differ"):
        resolve_cmc_policy(args, spec)


@pytest.mark.parametrize("selected_cmc", ["off", "auto"])
@pytest.mark.parametrize("authored_mask", ["omitted", None, 0.4])
def test_resume_with_original_tracker_yaml_restores_frozen_camera_mask(
    monkeypatch, tmp_path, selected_cmc, authored_mask
):
    """Repeating an unchanged starting YAML must not undo generated CMC settings."""
    args, bank = _joint_bank(monkeypatch, tmp_path, selected_cmc=selected_cmc)
    original = {
        name: value
        for name, value in bank.baseline_config.items()
        if name not in {"use_cmc", "cmc_method", "cmc_min_conf"}
    }
    if authored_mask != "omitted":
        original["cmc_min_conf"] = authored_mask
    args.tracker_config = tmp_path / "original.yaml"
    args.tracker_config.write_text(yaml.safe_dump(nest_tracker_options(original)))
    if authored_mask == 0.4:
        with pytest.raises(ValueError, match="Cannot change KF calibration inputs.*cmc_min_conf"):
            load_calibration_bank(args, tmp_path)
    else:
        assert load_calibration_bank(args, tmp_path) == bank


def test_evaluation_cannot_silently_override_joint_policy(monkeypatch, tmp_path):
    args, bank = _joint_bank(monkeypatch, tmp_path)
    spec = TrackerSpec("bytetrack", options=component_options(bank.profiles["xyah"]))
    args.cmc_policy = "off"
    with pytest.raises(ValueError, match="differs from this calibrated KF profile"):
        resolve_cmc_policy(args, spec)


def test_exported_camera_rule_does_not_require_training_sequence_artifacts(monkeypatch, tmp_path):
    """Evaluation on another build must analyze it instead of loading training decisions."""
    args, bank = _joint_bank(monkeypatch, tmp_path)
    (tmp_path / "kf-tuning" / "cmc-policy.yaml").unlink()
    args.sequence_names = ("unseen-evaluation-sequence",)
    spec = TrackerSpec("bytetrack", options=component_options(bank.profiles["xyah"]))
    assert resolve_cmc_policy(args, spec) == "auto"
    # Resume still requires the original fitting evidence for audit/reproducibility.
    args.sequence_names = ("seq1",)
    with pytest.raises(ValueError, match="CMC policy artifact"):
        load_calibration_bank(args, tmp_path)


@pytest.mark.parametrize(
    "changed,value,message",
    [
        ("cmc_min_conf", 0.9, "mask threshold differs"),
        ("cmc_method", "ecc", "CMC method differs"),
        ("calibration.cmc_rule", "sof-motion-v3", "supported automatic/off decision rule"),
        ("calibration.cmc_rule", "different-rule", "supported automatic/off decision rule"),
    ],
)
def test_portable_profile_rejects_changed_camera_behavior(monkeypatch, tmp_path, changed, value, message):
    """New sequences are allowed; the camera behavior underlying fitted noise stays fixed."""
    args, bank = _joint_bank(monkeypatch, tmp_path)
    options = {**bank.profiles["xyah"], changed: value}
    spec = TrackerSpec("bytetrack", options=component_options(options))
    with pytest.raises(ValueError, match=message):
        resolve_cmc_policy(args, spec)
