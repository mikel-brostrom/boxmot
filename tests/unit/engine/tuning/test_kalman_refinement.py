"""Search selected global/class covariance scales without changing their basis."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from boxmot.engine.tuning.backends.hyperopt_backend import yaml_to_hyperopt_space
from boxmot.engine.tuning.backends.optuna_backend import yaml_to_optuna_define_space
from boxmot.engine.tuning.kalman_refinement import prepare_kalman_refinement, refine_kalman_schema
from boxmot.engine.tuning.postprocessing import write_trial_yaml
from boxmot.engine.tuning.search_space import (
    conditional_yaml_tree,
    flatten_yaml_config,
    normalize_trial_config,
    yaml_to_tune_space,
)
from boxmot.trackers.common.config import flatten_tracker_options, load_tracker_defaults


def test_grouped_schema_preserves_conditional_children_across_search_backends() -> None:
    schema = {
        "enabled": {
            "type": "choice",
            "default": True,
            "options": [True],
            "activates": {
                "kalman_noise": {"process_position_scale": {"type": "loguniform", "default": 2.0, "range": [0.5, 8.0]}}
            },
        },
        "kalman_noise": {"time_unit": {"default": "frames"}},
    }
    key = "kalman_noise.process_position_scale"
    flat = flatten_yaml_config(schema)
    parents, children, mapping = conditional_yaml_tree(schema)
    assert key in flat and key in parents["enabled"]
    assert children == {key} and mapping == {key: "enabled"}
    ray = SimpleNamespace(choice=lambda choices: choices[0], loguniform=lambda low, high: (low, high))
    assert yaml_to_tune_space(schema, ray) == {"enabled": True, key: (0.5, 8.0)}
    optuna = pytest.importorskip("optuna")
    trial = optuna.trial.FixedTrial({"enabled": True, key: 3.0})
    yaml_to_optuna_define_space(schema)(trial)
    assert trial.params == {"enabled": True, key: 3.0}
    stochastic = pytest.importorskip("hyperopt.pyll.stochastic")
    sampled = normalize_trial_config(stochastic.sample(yaml_to_hyperopt_space(schema), rng=np.random.default_rng(4)))
    assert sampled["enabled"] is True
    assert 0.5 <= sampled[key] <= 8.0
    assert "kalman_noise.time_unit" not in sampled


def test_refinement_uses_each_class_prior_and_omits_unused_global_fallback() -> None:
    baseline = {
        **load_tracker_defaults("botsort"),
        "kalman_noise.by_class.1.process_velocity_scale": 2.0,
        "kalman_noise.by_class.3.process_velocity_scale": 12.0,
        "kalman_noise.by_class.3.measurement_noise_scale": 7.0,
        "kalman_noise.by_class.5.process_velocity_scale": 99.0,
    }
    schema = refine_kalman_schema(
        {"kalman_noise": {"process_velocity_scale": {"default": 1.0}}},
        baseline,
        ("process_velocity_scale",),
        class_ids=(1, 3),
    )
    assert schema["kalman_noise.process_velocity_scale"] == {"default": 1.0}
    assert schema["kalman_noise.by_class.1.process_velocity_scale"]["range"] == [0.5, 8.0]
    assert schema["kalman_noise.by_class.3.process_velocity_scale"]["range"] == [3.0, 48.0]
    assert "kalman_noise.by_class.5.process_velocity_scale" not in schema
    with_fallback = refine_kalman_schema({}, baseline, ("process_velocity_scale",), class_ids=(1, 2, 3))
    assert with_fallback["kalman_noise.process_velocity_scale"]["range"] == [0.25, 4.0]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_refinement_rejects_invalid_log_scale_baseline(value: float) -> None:
    with pytest.raises(ValueError, match="positive baseline"):
        refine_kalman_schema({}, {"kalman_noise.process_position_scale": value}, ("process_position_scale",))


def test_resume_requires_same_refinement_selection_and_all_class_baselines(tmp_path: Path) -> None:
    baseline = {**load_tracker_defaults("botsort"), "kalman_noise.by_class.1.process_velocity_scale": 2.0}
    args = SimpleNamespace(tune_kf=("process_velocity_scale",), tracker_class_ids=(1,), resume_tune=None)
    keys = prepare_kalman_refinement(args, baseline, tmp_path)
    assert keys == ("kalman_noise.by_class.1.process_velocity_scale",)
    args.resume_tune = tmp_path
    assert prepare_kalman_refinement(args, baseline, tmp_path) == keys
    for field in (
        "kalman_noise.process_position_scale",
        "kalman_noise.by_class.1.process_velocity_scale",
        "kalman_noise.reference_dt_s",
    ):
        with pytest.raises(ValueError, match="same baseline"):
            prepare_kalman_refinement(args, {**baseline, field: 123.0}, tmp_path)
    args.tune_kf = ()
    with pytest.raises(ValueError, match="same --tune-kf selection"):
        prepare_kalman_refinement(args, baseline, tmp_path)


def test_resume_cannot_add_refinement_or_accept_malformed_metadata(tmp_path: Path) -> None:
    args = SimpleNamespace(tune_kf=("process_position_scale",), resume_tune=tmp_path)
    baseline = load_tracker_defaults("botsort")
    with pytest.raises(ValueError, match="Cannot add --tune-kf"):
        prepare_kalman_refinement(args, baseline, tmp_path)
    (tmp_path / "kf-refinement.json").write_text("[", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        prepare_kalman_refinement(args, baseline, tmp_path)
    (tmp_path / "kf-refinement.json").write_text(
        json.dumps({"fields": ["kalman_noise.process_position_scale"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="same baseline"):
        prepare_kalman_refinement(args, baseline, tmp_path)


def test_best_yaml_nests_grouped_noise_and_preserves_class_settings(tmp_path: Path) -> None:
    baseline = {
        "kalman_noise.process_position_scale": 2.0,
        "kalman_noise.time_unit": "seconds",
        "kalman_noise.by_class.1.process_position_scale": 4.0,
        "kalman_noise.by_class.1.reference_dt_s": 0.05,
        "variable_dt": True,
        "calibration.tracker": "botsort",
        "calibration.geometry": "aabb",
    }
    path = tmp_path / "best.yaml"
    write_trial_yaml({}, {"kalman_noise.by_class.1.process_position_scale": 8.0}, path, base_config=baseline)
    saved = yaml.safe_load(path.read_text())
    assert not any("." in key for key in saved)
    assert saved["kalman_noise"]["by_class"]["1"]["process_position_scale"] == 8.0
    assert saved["calibration"] == {"tracker": "botsort", "geometry": "aabb"}
    assert flatten_tracker_options(saved) == {**baseline, "kalman_noise.by_class.1.process_position_scale": 8.0}
