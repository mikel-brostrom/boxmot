"""KF-only search, saved configuration reuse, and final evaluation wiring."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from boxmot.engine.eval import evaluator
from boxmot.engine.eval.results import ValidationResult
from boxmot.engine.tuning.kalman import kalman_search_bounds, tune_kalman, validate_kf_tuning
from boxmot.motion.kalman_filters.noise import KALMAN_NOISE_OPTIONS
from boxmot.trackers.config import load_tracker_config


def _args(tmp_path, **overrides):
    return SimpleNamespace(
        **{
            "tracker": "bytetrack",
            "tracker_backend": "python",
            "geometry": "aabb",
            "dataset_id": "fixture",
            "split": "ablation",
            "build_path": tmp_path / "build",
            "sequence_names": ("seq1",),
            "kf_trials": 3,
            "variable_dt": False,
            "_build_validated": True,
            **overrides,
        }
    )


def _fake_replay(monkeypatch, scores):
    calls = []

    def replay(args, **kwargs):
        config = dict(evaluator._tracker_options(args, kwargs.get("evolve_config")))
        score = scores[len(calls)]
        calls.append((args, config, kwargs))
        return ValidationResult(
            benchmark="fixture",
            raw={"HOTA": score},
            summary_label="single_class",
            summary={"HOTA": score},
            exp_dir=kwargs["output_dir"],
            args=args,
        )

    monkeypatch.setattr(evaluator, "run_eval", replay)
    return calls


@pytest.mark.parametrize("variable_dt", [False, True])
def test_search_selects_best_and_holds_timing_and_association_fixed(monkeypatch, tmp_path, variable_dt):
    args = _args(
        tmp_path,
        variable_dt=variable_dt,
        asso_func="giou",
        per_class=True,
        tracker_class_ids=(0,),
        tracker_class_names=((0, "pedestrian"),),
    )
    calls = _fake_replay(monkeypatch, [60.0, 65.0, 55.0])
    progress = []
    result = tune_kalman(args, output_dir=tmp_path, progress=progress.append)
    saved = load_tracker_config("bytetrack", result.config_path)
    assert result.baseline_hota == 60.0
    assert result.best_hota == 65.0
    assert saved == calls[1][1]
    assert saved["variable_dt"] is variable_dt
    assert saved["asso_func"] == "giou"
    assert {key: calls[0][1][key] for key in KALMAN_NOISE_OPTIONS} == dict.fromkeys(KALMAN_NOISE_OPTIONS, 1.0)
    constants = {key: value for key, value in calls[0][1].items() if key not in KALMAN_NOISE_OPTIONS}
    for trial_args, config, options in calls:
        assert {key: value for key, value in config.items() if key not in KALMAN_NOISE_OPTIONS} == constants
        assert trial_args is not args
        assert options["setup"] is False
        assert trial_args.compare_trackeval is False
    assert len({entry[2]["output_dir"] for entry in calls}) == 3
    report = json.loads(result.report_path.read_text())
    assert report["best_trial"] == 1
    assert report["status"] == "complete"
    assert report["score_scope"] == "tuned_on_selected_split"
    assert report["sequences"] == ["seq1"]
    assert report["per_class"] is True
    assert report["class_ids"] == [0]
    assert report["class_names"] == {"0": "pedestrian"}
    assert "tuning scores" in result.description
    assert len(progress) == 4
    assert not hasattr(args, "exp_dir")


def test_baseline_retained_if_every_candidate_is_worse_or_tied(monkeypatch, tmp_path):
    calls = _fake_replay(monkeypatch, [60.0, 59.0, 60.0])
    result = tune_kalman(_args(tmp_path), output_dir=tmp_path)
    assert load_tracker_config("bytetrack", result.config_path) == calls[0][1]
    assert json.loads(result.report_path.read_text())["best_trial"] == 0


def test_custom_baseline_scales_and_implicit_timing_are_preserved(monkeypatch, tmp_path):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "tracker": "bytetrack",
                "variable_dt": True,
                "kf_process_position_scale": 0.0001,
                "kf_measurement_noise_scale": 2.0,
                "kf_initial_velocity_scale": 900.0,
            }
        )
    )
    calls = _fake_replay(monkeypatch, [61.0])
    result = tune_kalman(
        _args(tmp_path, kf_trials=1, variable_dt=None, tracker_config=config_path), output_dir=tmp_path
    )
    saved = load_tracker_config("bytetrack", result.config_path)
    assert saved == calls[0][1]
    assert saved["variable_dt"] is True
    assert saved["kf_process_position_scale"] == 0.0001
    assert saved["kf_initial_velocity_scale"] == 900.0
    report = json.loads(result.report_path.read_text())
    assert report["search_bounds"]["kf_process_position_scale"][0] == 0.0001


def test_search_seed_repeats_candidates(monkeypatch, tmp_path):
    candidates = []
    for index in range(2):
        calls = _fake_replay(monkeypatch, [50.0] * 7)
        tune_kalman(_args(tmp_path, kf_trials=7), output_dir=tmp_path / str(index))
        candidates.append([config for _, config, _ in calls])
    assert candidates[0] == candidates[1]


def test_noise_search_uses_tracker_yaml_and_excludes_time():
    from boxmot.engine.tuning.search_space import load_yaml_config

    bounds = kalman_search_bounds("bytetrack")
    schema = load_yaml_config("bytetrack")
    assert set(bounds) == set(KALMAN_NOISE_OPTIONS)
    assert bounds == {name: tuple(schema[name]["range"]) for name in KALMAN_NOISE_OPTIONS}


@pytest.mark.parametrize("mode,override", [("seconds", False), ("frames", True)])
def test_calibrated_units_cannot_be_overridden(monkeypatch, tmp_path, mode, override):
    path = tmp_path / "calibrated.yaml"
    path.write_text(yaml.safe_dump({"variable_dt": mode == "seconds", "kf_time_unit": mode, "kf_reference_dt_s": 0.04}))
    calls = _fake_replay(monkeypatch, [])
    with pytest.raises(ValueError, match="conflicts with variable_dt"):
        tune_kalman(_args(tmp_path, tracker_config=path, variable_dt=override), output_dir=tmp_path)
    assert calls == []


def test_unfiltered_replay_records_all_resolved_sequences(monkeypatch, tmp_path):
    _fake_replay(monkeypatch, [60.0])
    result = tune_kalman(
        _args(tmp_path, kf_trials=1, sequence_names=None, seq_info={"seq1": 50, "seq2": 100}), output_dir=tmp_path
    )
    assert json.loads(result.report_path.read_text())["sequences"] == ["seq1", "seq2"]


@pytest.mark.parametrize("tracker,backend", [("sam2mot", "python"), ("botsort", "cpp")])
def test_unsupported_tuning_rejected_before_replay(tracker, backend):
    with pytest.raises(ValueError, match="Python tracker with a Kalman filter"):
        validate_kf_tuning(tracker, backend)


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_trial_budget_rejected(tmp_path, count):
    with pytest.raises(ValueError, match="positive integer"):
        tune_kalman(_args(tmp_path, kf_trials=count), output_dir=tmp_path)
    assert not (tmp_path / "kf-tuning").exists()


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_invalid_score_cannot_be_saved_as_best(monkeypatch, tmp_path, score):
    _fake_replay(monkeypatch, [score])
    with pytest.raises(ValueError, match="finite HOTA"):
        tune_kalman(_args(tmp_path, kf_trials=1), output_dir=tmp_path)
    assert not (tmp_path / "kf-tuning" / "best.yaml").exists()


def test_eval_main_replays_saved_winner_after_search(monkeypatch, tmp_path):
    calls = _fake_replay(monkeypatch, [60.0, 64.0, 61.0, 64.0])
    monkeypatch.setattr(evaluator, "eval_setup", lambda *args, **kwargs: None)
    args = _args(tmp_path, kf_tuning=True, project=tmp_path, name="eval", experiment_id="fixture", show=True, save=True)
    result = evaluator.main(args)
    assert len(calls) == 4
    assert calls[-1][1] == calls[1][1]
    assert "evolve_config" not in calls[-1][2]
    assert all(not trial_args.show and not trial_args.save for trial_args, _, _ in calls[:-1])
    assert calls[-1][0].show is True
    assert calls[-1][0].save is True
    assert result.exp_dir == tmp_path / "fixture" / "eval"
    report_path = result.exp_dir / "kf-tuning" / "trials.json"
    report = json.loads(report_path.read_text())
    assert report["final_summary"] == {"HOTA": 64.0}
    assert Path(report["final_output_dir"]) == result.exp_dir


def test_eval_without_tuning_runs_once(monkeypatch, tmp_path):
    calls = []

    def replay(args, **kwargs):
        calls.append(kwargs)
        return ValidationResult("fixture", {}, "", {}, exp_dir=tmp_path, args=args)

    monkeypatch.setattr(evaluator, "run_eval", replay)
    evaluator.main(_args(tmp_path))
    assert len(calls) == 1
    assert not (tmp_path / "kf-tuning").exists()
