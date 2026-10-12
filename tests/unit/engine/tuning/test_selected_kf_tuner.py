"""Selected Kalman profiles remain fixed through ordinary tuning and resume."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from boxmot.engine.calibration import kalman, kalman_selection
from boxmot.engine.calibration.kalman import calibrate_kalman
from boxmot.engine.tuning.tuner import Tuner
from tests.unit.engine.calibration.test_kalman import _data, _load_fixture
from tests.unit.engine.tuning.test_calibrated_tuner import fake_tuning as fake_tuning


@pytest.mark.parametrize("search_alg", ["optuna", "random"])
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("profile", ["default", "calibrated"])
def test_selected_kf_reaches_trials_and_resumes_without_refitting(
    fake_tuning: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    search_alg: str, geometry: str, profile: str,
) -> None:
    """Exercise real fitting, profile persistence and trial dispatch without Ray workers."""
    loads = _load_fixture(monkeypatch, _data(geometry=geometry))
    calls = []

    def fit(*args, **kwargs):
        calls.append(kwargs["profile_name"])
        return calibrate_kalman(*args, **kwargs)

    def select(data, *, configs, **kwargs):
        assert "camera_candidates" not in kwargs
        assert all(config["use_cmc"] is False for config in configs.values())
        return {
            "method": "grouped_cross_validated_box_profiles",
            "selected_type": "xywh",
            "selected_profile": profile,
            "candidates": {
                kind: {"status": "valid", "selected_profile": profile, "mean_iou": 0.8}
                for kind in configs
            },
        }

    monkeypatch.setattr(kalman, "calibrate_kalman", fit)
    monkeypatch.setattr(kalman_selection, "select_kalman_model", select)
    options = dict(
        tracker="bytetrack", calibrate_kf=False, variable_dt=False,
        tracker_class_ids=(0,), sequence_names=("seq1",), seq_info={"seq1": 6},
        geometry=geometry, search_alg=search_alg, _build_validated=True,
    )
    driver = Tuner(fake_tuning.args(**options, select_kf=True))
    _, directory, *_ = driver.fit()
    assert set(calls) == {"xyah", "xywh", "xysr", "xyhr", "xyscr"}
    assert len(calls) == 5 and len(loads) == 1
    selection = json.loads((directory / "kf-tuning" / "selection.json").read_text())
    assert selection["baseline_type"] == ("xywh" if geometry == "obb" else "xyah")
    assert selection["selected_type"] == "xywh"
    assert selection["selected_profile"] == profile
    captured = fake_tuning.captured
    fixed = driver._calibrated_fixed_options
    assert fixed["kalman.type"] == "xywh"
    assert fixed["use_cmc"] is False
    for config in captured["trial_configs"]:
        assert {key: config[key] for key in fixed} == fixed
        assert config["calibration.filter"] == "xywh"

    def forbidden(*args, **kwargs):
        pytest.fail("Resuming selection must reuse its saved profiles.")

    monkeypatch.setattr(kalman, "calibrate_kalman", forbidden)
    monkeypatch.setattr(kalman_selection, "select_kalman_model", forbidden)
    captured["restore_enabled"] = True
    resumed = Tuner(fake_tuning.args(**options, resume_tune=directory))
    resumed.fit()
    assert resumed._runtime_config == driver._runtime_config
    assert len(captured["trial_configs"]) == 4
    assert all({key: config[key] for key in fixed} == fixed for config in captured["trial_configs"])
