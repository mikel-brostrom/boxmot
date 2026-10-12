"""Persist the independently selected filter and restore it without new fitting."""

import hashlib
import json

import numpy as np
import pytest
import yaml

from boxmot.engine.calibration import kalman_selection
from boxmot.engine.config.trackers import resolve_tracker_options
from boxmot.engine.tuning.calibration_bank import create_calibration_bank, load_calibration_bank
from boxmot.trackers.common.config import load_tracker_config, nest_tracker_options
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES
from tests.unit.engine.calibration.test_kalman import _args, _data, _load_fixture


def _bank(
    monkeypatch,
    tmp_path,
    *,
    selected_profile="calibrated",
    invalid_baseline=False,
    tracker="bytetrack",
    per_class=False,
    varied_confidence=False,
):
    """Use real calibration profiles while isolating selection persistence."""
    data = _data()
    if varied_confidence:
        for track in data.tracks:
            track.scores[:] = np.linspace(0.1, 0.99, len(track.scores))
    inputs = _load_fixture(monkeypatch, data)
    selections = []

    def select(data, **kwargs):
        selections.append((data, kwargs))
        return {
            "method": "grouped_cross_validated_box_profiles",
            "selected_type": "xywh",
            "selected_profile": selected_profile,
            "candidates": {
                "xyah": {
                    "mean_iou": None if invalid_baseline else 0.6,
                    "selected_profile": None if invalid_baseline else "default",
                    "status": "invalid" if invalid_baseline else "valid",
                },
                "xywh": {"mean_iou": 0.8, "selected_profile": selected_profile, "status": "valid"},
            },
            "folds": [{"training_sequences": ["first"], "validation_sequences": ["second"]}],
        }

    monkeypatch.setattr(kalman_selection, "select_kalman_model", select)
    args = _args(tmp_path, tracker=tracker, per_class=per_class, tracker_class_ids=(0,))
    baseline = resolve_tracker_options(args, {"kalman.type": "xyah"}, include_defaults=True, stamp_timing=True)
    bank = create_calibration_bank(args, tmp_path, runtime_config=baseline, kinds=("xyah", "xywh"), select_best=True)
    return args, bank, inputs, selections


def test_selection_is_saved_with_profiles_and_restored_without_refitting(monkeypatch, tmp_path):
    args, bank, inputs, selections = _bank(monkeypatch, tmp_path)
    assert len(inputs) == len(selections) == 1
    assert selections[0][1]["baseline_type"] == "xyah"
    assert set(selections[0][1]["configs"]) == {"xyah", "xywh"}
    assert bank.baseline_type == "xyah"
    assert bank.selected_type == "xywh"
    assert bank.selected_profile == "calibrated"
    assert bank.selection_path == tmp_path / "kf-tuning" / "selection.json"
    report = json.loads(bank.selection_path.read_text())
    assert report["status"] == "complete"
    assert report["selected_type"] == "xywh"
    assert report["selected_config"] == "xywh/calibrated.yaml"
    assert report["filter_types"] == ["xyah", "xywh"]
    assert report["build"] == str(args.build_path.resolve())
    assert load_calibration_bank(args, tmp_path) == bank
    assert len(inputs) == len(selections) == 1
    overridden = load_calibration_bank(args, tmp_path, overrides={"match_thresh": 0.8})
    assert overridden.selected_type == "xywh"
    assert overridden.profiles["xywh"]["match_thresh"] == 0.8


@pytest.mark.parametrize("tracker", sorted(KALMAN_TRACKER_NAMES))
def test_kf_only_selection_freezes_cmc_off_without_changing_tracker_defaults(monkeypatch, tmp_path, tracker):
    """An uncompensated KF comparison must export and resume uncompensated tracking."""
    defaults = load_tracker_config(tracker)
    args, bank, inputs, selections = _bank(monkeypatch, tmp_path, tracker=tracker)
    assert load_tracker_config(tracker) == defaults
    assert bank.selected_cmc is None
    assert bank.baseline_config["use_cmc"] is False
    assert all(config["use_cmc"] is False for config in selections[0][1]["configs"].values())
    for kind, config in bank.profiles.items():
        assert config["use_cmc"] is False
        report = json.loads((tmp_path / "kf-tuning" / kind / "calibration.json").read_text())
        assert report["tuning"]["fixed_options"]["use_cmc"] is False
    assert load_calibration_bank(args, tmp_path) == bank
    args.tracker_config = tmp_path / "original.yaml"
    args.tracker_config.write_text(yaml.safe_dump({"tracker": tracker, "kalman": {"type": "xyah"}}))
    assert load_calibration_bank(args, tmp_path) == bank
    with pytest.raises(ValueError, match="Cannot change KF calibration inputs.*use_cmc"):
        load_calibration_bank(args, tmp_path, overrides={"use_cmc": True})
    assert len(inputs) == len(selections) == 1


def test_selection_plots_run_after_the_complete_bank_is_published(monkeypatch, tmp_path):
    """Image generation consumes durable reports and does not run again on resume."""
    from boxmot.engine.tuning import analysis

    rendered = []

    def render(directory):
        bank_dir = directory / "kf-tuning"
        assert json.loads((bank_dir / "filters.json").read_text())["status"] == "complete"
        assert json.loads((bank_dir / "selection.json").read_text())["selected_type"] == "xywh"
        rendered.append(directory)
        return ()

    monkeypatch.setattr(analysis, "generate_selection_analysis", render)
    args, bank, _, _ = _bank(monkeypatch, tmp_path)
    assert rendered == [tmp_path]
    assert load_calibration_bank(args, tmp_path) == bank
    assert rendered == [tmp_path]


def test_default_winner_saves_and_restores_original_noise(monkeypatch, tmp_path):
    from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_DIAGONAL_OPTIONS, KALMAN_NOISE_OPTIONS

    args, bank, inputs, selections = _bank(monkeypatch, tmp_path, selected_profile="default")
    assert bank.selected_profile == "default"
    config = bank.profiles[bank.selected_type]
    assert all(config[name] == 1.0 for name in KALMAN_NOISE_OPTIONS)
    assert not set(config).intersection(KALMAN_DIAGONAL_OPTIONS)
    assert config["calibration.profile"] == "default"
    report = json.loads(bank.selection_path.read_text())
    assert report["selected_config"] == "xywh/default.yaml"
    assert (tmp_path / "kf-tuning/xywh/calibrated.yaml").is_file()
    assert load_calibration_bank(args, tmp_path) == bank
    assert len(inputs) == len(selections) == 1


@pytest.mark.parametrize("profile", ["motion", "measurement"])
def test_partial_winner_preserves_rejected_covariances_and_reloads(monkeypatch, tmp_path, profile):
    """A partial winner must keep only its validated covariance group."""
    from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_DIAGONAL_OPTIONS, KALMAN_NOISE_OPTIONS

    args, bank, inputs, selections = _bank(monkeypatch, tmp_path, selected_profile=profile)
    active = bank.profiles["xywh"]
    fitted = load_tracker_config("bytetrack", tmp_path / "kf-tuning/xywh/calibrated.yaml")
    assert bank.selected_profile == active["calibration.profile"] == profile
    assert json.loads(bank.selection_path.read_text())["selected_config"] == f"xywh/{profile}.yaml"
    for scalar, diagonal in zip(KALMAN_NOISE_OPTIONS, KALMAN_DIAGONAL_OPTIONS, strict=True):
        keep = ("measurement" in scalar) == (profile == "measurement")
        if keep:
            assert active[scalar] == fitted[scalar]
            assert active[diagonal] == fitted[diagonal]
        else:
            assert active[scalar] == 1.0
            assert diagonal not in active
    assert load_calibration_bank(args, tmp_path) == bank
    assert len(inputs) == len(selections) == 1


def test_partial_recipe_corruption_is_rejected_even_with_updated_digests(monkeypatch, tmp_path):
    """A saved motion profile cannot silently activate rejected measurement noise."""
    args, bank, _, _ = _bank(monkeypatch, tmp_path, selected_profile="motion")
    directory = tmp_path / "kf-tuning"
    path = directory / "xywh/motion.yaml"
    active = load_tracker_config("bytetrack", path)
    active["kalman.noise.measurement_noise_scale"] = 3.0
    path.write_text(yaml.safe_dump({"tracker": "bytetrack", **nest_tracker_options(active)}))
    index = json.loads(bank.index_path.read_text())
    index["profiles"]["xywh"]["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    bank.index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="motion noise.*recipe"):
        load_calibration_bank(args, tmp_path)


@pytest.mark.parametrize("profile", ["default", "motion", "measurement", "calibrated"])
def test_learned_confidence_curve_follows_selected_group_and_class_on_reload(monkeypatch, tmp_path, profile):
    """A saved class curve follows R and cannot leak into a motion-only winner."""
    args, bank, _, _ = _bank(
        monkeypatch,
        tmp_path,
        selected_profile=profile,
        tracker="strongsort",
        per_class=True,
        varied_confidence=True,
    )
    floor = "kalman.noise.measurement_confidence_floor"
    child_floor = "kalman.noise.by_class.0.measurement_confidence_floor"
    raw = load_tracker_config("strongsort", tmp_path / "kf-tuning/xywh/calibrated.yaml")
    assert raw[floor][0] == pytest.approx(1.0)
    active = bank.profiles["xywh"]
    for key in (floor, child_floor):
        if profile in {"measurement", "calibrated"}:
            assert active[key] == raw[key]
        else:
            assert key not in active
    assert load_calibration_bank(args, tmp_path) == bank


@pytest.mark.parametrize("profile", ["motion", "measurement"])
def test_partial_winner_requires_successful_full_data_fit(monkeypatch, tmp_path, profile):
    """A failed refit cannot silently replace the selected partial recipe."""
    from boxmot.engine.calibration import kalman

    def invalid_fit(*args, **kwargs):
        raise ValueError("unrepresentable covariance")

    monkeypatch.setattr(kalman, "calibrate_kalman", invalid_fit)
    with pytest.raises(ValueError, match="unrepresentable covariance"):
        _bank(monkeypatch, tmp_path, selected_profile=profile)
    assert not (tmp_path / "kf-tuning/filters.json").exists()


def test_rejected_full_data_fit_cannot_prevent_default_winner(monkeypatch, tmp_path):
    from boxmot.engine.calibration import kalman

    def invalid_fit(*args, **kwargs):
        raise ValueError("unrepresentable covariance")

    monkeypatch.setattr(kalman, "calibrate_kalman", invalid_fit)
    args, bank, _, _ = _bank(monkeypatch, tmp_path, selected_profile="default")
    assert bank.selected_profile == "default"
    assert load_calibration_bank(args, tmp_path) == bank
    report = json.loads((tmp_path / "kf-tuning/xywh/calibration.json").read_text())
    assert report["calibration_error"] == "unrepresentable covariance"
    assert not (tmp_path / "kf-tuning/xywh/calibrated.yaml").exists()


def test_invalid_candidate_does_not_prevent_saving_valid_winner(monkeypatch, tmp_path):
    args, bank, _, _ = _bank(monkeypatch, tmp_path, invalid_baseline=True)
    assert set(bank.profiles) == {"xywh"}
    assert bank.baseline_type == "xyah"
    assert load_calibration_bank(args, tmp_path) == bank
    assert not (tmp_path / "kf-tuning/xyah").exists()


@pytest.mark.parametrize(
    "change", ["missing", "edited", "unbound", "wrong_kind", "wrong_method", "wrong_profile", "invalid_candidate"]
)
def test_selection_corruption_cannot_silently_restore_joint_search(monkeypatch, tmp_path, change):
    args, bank, _, _ = _bank(monkeypatch, tmp_path)
    index = json.loads(bank.index_path.read_text())
    if change == "missing":
        bank.selection_path.unlink()
    elif change == "edited":
        bank.selection_path.write_text(bank.selection_path.read_text() + "\n")
    elif change == "unbound":
        del index["selection"]
        bank.index_path.write_text(json.dumps(index))
    else:
        report = json.loads(bank.selection_path.read_text())
        if change == "invalid_candidate":
            report["candidates"]["xywh"]["status"] = "invalid"
        else:
            field = {"wrong_kind": "selected_type", "wrong_method": "method", "wrong_profile": "selected_profile"}[
                change
            ]
            report[field] = "xyah"
        bank.selection_path.write_text(json.dumps(report))
        index["selection"]["report_sha256"] = hashlib.sha256(bank.selection_path.read_bytes()).hexdigest()
        bank.index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="KF selection"):
        load_calibration_bank(args, tmp_path)


def test_selection_failure_publishes_no_completed_bank(monkeypatch, tmp_path):
    _load_fixture(monkeypatch, _data())
    args = _args(tmp_path)
    baseline = resolve_tracker_options(args, include_defaults=True, stamp_timing=True)

    def select(*args, **kwargs):
        raise ValueError("insufficient held-out data")

    monkeypatch.setattr(kalman_selection, "select_kalman_model", select)
    with pytest.raises(ValueError, match="insufficient held-out"):
        create_calibration_bank(args, tmp_path, runtime_config=baseline, kinds=("xyah", "xywh"), select_best=True)
    assert not (tmp_path / "kf-tuning" / "filters.json").exists()
    assert not (tmp_path / "kf-tuning" / "selection.json").exists()
