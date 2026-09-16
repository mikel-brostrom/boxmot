"""Calibrate once, then keep the selected Kalman model fixed during Ray trials."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

import boxmot.engine.calibration.kalman as kalman_module
import boxmot.engine.tuning.tuner as tuner_module
from boxmot.engine.calibration.kalman import calibrate_kalman
from boxmot.engine.config.trackers import resolve_tracker_options
from boxmot.engine.tuning.search_space import flatten_yaml_config, load_yaml_config
from boxmot.trackers.common.config import load_tracker_config, nest_tracker_options
from boxmot.trackers.common.mask_guidance import MASK_GUIDANCE_OPTIONS
from boxmot.trackers.common.motion.kalman_filters.noise import DEFAULT_REFERENCE_DT_S, KALMAN_NOISE_OPTIONS
from tests.unit.engine.calibration.test_kalman import _data, _load_fixture


@dataclass(frozen=True)
class _Domain:
    """A fake Ray distribution that retains its identity until trial execution."""

    value: object


class _SuggestionTrial:
    """Execute the real Optuna define-by-run function without an Optuna job."""

    def __init__(self) -> None:
        self.params: dict[str, object] = {}

    def suggest_float(self, name: str, low: float, high: float, **kwargs: object) -> float:
        del kwargs
        self.params[name] = (low + high) / 2.0
        return self.params[name]

    def suggest_int(self, name: str, low: int, high: int, **kwargs: object) -> int:
        del high, kwargs
        self.params[name] = low
        return low

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        self.params[name] = choices[0]
        return choices[0]


@pytest.fixture
def fake_tuning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Exercise driver orchestration and actual search-space conversion cheaply."""
    captured = {
        "events": [],
        "calibrations": [],
        "trial_configs": [],
        "saved_results": [],
        "restore_enabled": False,
    }
    pipeline = MagicMock()
    pipeline.__enter__.return_value = pipeline
    monkeypatch.setattr(tuner_module.TuneWorkflowReporter, "pipeline", lambda *args, **kwargs: pipeline)
    monkeypatch.setattr(tuner_module, "set_tune_progress_workflow", lambda *args: None)
    monkeypatch.setattr(tuner_module.Tuner, "_configure_warning_filters", lambda self: None)
    monkeypatch.setattr(tuner_module, "_require_tuning_requirements", lambda: None)
    monkeypatch.setattr(tuner_module.Tuner, "_inject_callback_into_restored", lambda *args: None)

    def setup(args: SimpleNamespace, pipeline: object = None) -> None:
        del pipeline
        captured["events"].append("eval_setup")
        args.project = (tmp_path / "runs").resolve()

    monkeypatch.setattr(tuner_module, "eval_setup", setup)

    def calibrate(
        args: SimpleNamespace,
        *,
        output_dir: Path,
        progress: object = None,
        tracker_options: dict | None = None,
    ) -> kalman_module.KalmanCalibrationResult:
        del progress
        captured["events"].append("calibrate")
        baseline = resolve_tracker_options(args, tracker_options, include_defaults=True, stamp_timing=True)
        captured["calibrations"].append(dict(baseline))
        fitted = captured.get(
            "fitted_values", {name: float(index + 2) for index, name in enumerate(KALMAN_NOISE_OPTIONS)}
        )
        config = {"tracker": args.tracker, **baseline, **fitted}
        directory = output_dir / "kf-tuning"
        directory.mkdir(parents=True, exist_ok=True)
        config_path = directory / "calibrated.yaml"
        report_path = directory / "calibration.json"
        config_path.write_text(yaml.safe_dump(nest_tracker_options(config)), encoding="utf-8")
        report_path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "status": "complete",
                    "method": "supervised_covariance_moments",
                    "tracker": args.tracker,
                    "geometry": args.geometry,
                    "dataset": args.dataset_id,
                    "split": args.split,
                    "build": str(args.build_path),
                    "sequences": list(args.sequence_names),
                    "per_class": args.per_class,
                    "class_ids": list(args.tracker_class_ids),
                    "timing": {
                        key: config[key]
                        for key in ("kalman.variable_dt", "kalman.noise.time_unit", "kalman.noise.reference_dt_s")
                    },
                    "baseline_config": baseline,
                }
            ),
            encoding="utf-8",
        )
        captured["calibrated_config"] = config
        captured["config_path"] = config_path
        captured["report_path"] = report_path
        return kalman_module.KalmanCalibrationResult(config_path, report_path, 20, 15, KALMAN_NOISE_OPTIONS)

    monkeypatch.setattr(kalman_module, "calibrate_kalman", calibrate)

    def save_results(*args: object, base_config: dict, **kwargs: object) -> None:
        del args, kwargs
        captured["postprocess_base"] = dict(base_config)

    monkeypatch.setattr(tuner_module, "save_all_results", save_results)

    class Objective:
        def __init__(self, args: SimpleNamespace) -> None:
            captured["trial_args"] = args

        def __call__(self, config: dict) -> dict:
            captured["trial_configs"].append(dict(config))
            return {"HOTA": 50.0}

        def close(self) -> None:
            pass

    monkeypatch.setattr(tuner_module, "TrackerObjective", Objective)

    class RunConfig:
        def __init__(self, storage_path: str, name: str, callbacks: object = None, **kwargs: object) -> None:
            del callbacks, kwargs
            self.storage_path, self.name = storage_path, name

    class RayTuner:
        def __init__(self, trainable: object, param_space: dict, tune_config: object, run_config: RunConfig) -> None:
            assert tune_config.reuse_actors is True
            captured["events"].append("ray_tuner")
            captured["param_space"] = dict(param_space)
            captured["tune_dir"] = Path(run_config.storage_path) / run_config.name
            self.trainable = trainable
            self.restored = False

        @staticmethod
        def can_restore(path: str) -> bool:
            captured["restore_path"] = Path(path)
            return captured["restore_enabled"]

        @classmethod
        def restore(cls, path: str, *, trainable: object, **kwargs: object) -> RayTuner:
            del path, kwargs
            captured["events"].append("restore")
            restored = object.__new__(cls)
            restored.trainable = trainable
            restored.restored = True
            return restored

        def fit(self) -> list:
            captured["events"].append("fit")
            if self.restored:
                configs = [dict(result.config) for result in captured["saved_results"]]
            else:
                configs = []
                for _ in range(2):
                    suggestion = _SuggestionTrial()
                    if "search_space" in captured:
                        captured["search_space"](suggestion)
                    config = {
                        key: value.value if isinstance(value, _Domain) else value
                        for key, value in captured["param_space"].items()
                    }
                    configs.append({**config, **suggestion.params})
            actor = self.trainable()
            actor.setup(configs[0])
            try:
                for config in configs:
                    assert actor.reset_config(config)
                    assert actor.step()["done"] is True
            finally:
                actor.cleanup()
            captured["saved_results"] = [SimpleNamespace(config=config) for config in configs]
            return []

        def get_results(self) -> list:
            return captured["saved_results"]

    class OptunaSearch:
        def __init__(self, *, space: object, **kwargs: object) -> None:
            captured["search_space"] = space
            captured["search_kwargs"] = kwargs

    fake_tune = SimpleNamespace(
        Tuner=RayTuner,
        Trainable=object,
        TuneConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        with_resources=lambda function, resources: function,
        uniform=lambda low, high: _Domain((low + high) / 2.0),
        loguniform=lambda low, high: _Domain((low * high) ** 0.5),
        randint=lambda low, high: _Domain(low),
        qrandint=lambda low, high, step: _Domain(low),
        choice=lambda choices: _Domain(choices[0]),
    )
    fake_ray = SimpleNamespace(
        _private=SimpleNamespace(ray_constants=SimpleNamespace(RAY_ENABLE_UV_RUN_RUNTIME_ENV=True)),
        tune=fake_tune,
        is_initialized=lambda: False,
        init=lambda **kwargs: captured["events"].append("ray_init"),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(
        sys.modules,
        "ray.tune",
        SimpleNamespace(
            RunConfig=RunConfig,
            CheckpointConfig=lambda **kwargs: None,
            FailureConfig=lambda **kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "ray.tune.search",
        SimpleNamespace(ConcurrencyLimiter=lambda searcher, max_concurrent: searcher),
    )
    monkeypatch.setitem(sys.modules, "ray.tune.search.optuna", SimpleNamespace(OptunaSearch=OptunaSearch))

    def args(**overrides: object) -> SimpleNamespace:
        return SimpleNamespace(
            **{
                "tracker": "botsort",
                "tracker_backend": "python",
                "tracker_config": None,
                "variable_dt": True,
                "calibrate_kf": True,
                "resume_tune": None,
                "search_alg": "optuna",
                "dataset_id": "mot17-mini",
                "geometry": "aabb",
                "split": "ablation",
                "build_path": tmp_path / "cache",
                "sequence_names": ("seq",),
                "seq_info": {"seq": 6},
                "per_class": False,
                "tracker_class_ids": (1,),
                "project": tmp_path / "runs",
                "sequence_workers": 1,
                "n_trials": 2,
                "verbose": False,
                "maximize": ("HOTA",),
                "minimize": (),
                "objectives": ("HOTA",),
                **overrides,
            }
        )

    return SimpleNamespace(captured=captured, args=args)


@pytest.mark.parametrize("search_alg", ["optuna", "random"])
def test_kf_is_calibrated_once_before_ray_and_frozen_in_every_trial(
    fake_tuning: SimpleNamespace, search_alg: str
) -> None:
    args = fake_tuning.args(search_alg=search_alg)
    captured = fake_tuning.captured
    tuner = tuner_module.Tuner(args, baseline_config={"kalman.noise.reference_dt_s": 0.05})

    tuner.fit()

    assert captured["events"].count("calibrate") == 1
    assert captured["events"].index("eval_setup") < captured["events"].index("calibrate")
    assert captured["events"].index("calibrate") < captured["events"].index("ray_init")
    frozen = {
        **{name: float(index + 2) for index, name in enumerate(KALMAN_NOISE_OPTIONS)},
        "kalman.variable_dt": True,
        "kalman.noise.time_unit": "seconds",
        "kalman.noise.reference_dt_s": 0.05,
    }
    for key, expected in frozen.items():
        assert captured["param_space"][key] == expected
        assert captured["postprocess_base"][key] == expected
        assert all(config[key] == expected for config in captured["trial_configs"])
        assert flatten_yaml_config(tuner._yaml_cfg)[key] == {"default": expected}
    assert len(captured["trial_configs"]) == 2
    assert Path(args.tracker_config) == captured["config_path"]
    report = json.loads(captured["report_path"].read_text())
    assert report["tuning"]["fixed_options"] == frozen
    assert "type" in flatten_yaml_config(tuner._yaml_cfg)["match_thresh"]


@pytest.mark.parametrize("search_alg", ["optuna", "random"])
@pytest.mark.parametrize("guided", [False, True])
def test_driver_conditions_mask_search_before_dispatching_trials(
    fake_tuning: SimpleNamespace, tmp_path: Path, search_alg: str, guided: bool
) -> None:
    """Exercise the driver, real schema converters and reused trial actor together."""
    from boxmot.trackers.common.mask_guidance import MASK_GUIDANCE_OPTIONS

    checkpoint = tmp_path / "edgetam.pt"
    checkpoint.write_bytes(b"local checkpoint content")
    args = fake_tuning.args(
        tracker="bytetrack",
        search_alg=search_alg,
        calibrate_kf=False,
        edgetam=guided,
        mask_guidance_weights=checkpoint if guided else None,
        mask_guidance_max_objects=7 if guided else None,
        device="cpu",
    )
    driver = tuner_module.Tuner(args)
    driver.fit()
    captured = fake_tuning.captured
    assert len(captured["trial_configs"]) == 2
    for config in captured["trial_configs"]:
        if guided:
            assert config["asso_func"] == "iou"
            assert config["edgetam.max_objects"] == 7
            assert 0 <= config["edgetam.min_coverage"] <= 1
            assert 0 <= config["edgetam.min_fill"] <= 1
            assert 0 < config["edgetam.prompt_overlap"] <= 1
        else:
            assert not set(MASK_GUIDANCE_OPTIONS).intersection(config)
    assert captured["trial_args"].mask_guidance_weights == (checkpoint if guided else None)
    assert driver._yaml_cfg["asso_func"] == (
        {"default": "iou"} if guided else load_yaml_config("bytetrack")["asso_func"]
    )


def test_occluboost_guidance_runs_real_calibration_before_tuning(
    fake_tuning: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fit from saved detection/GT samples before dispatching guided trials."""
    calls = _load_fixture(monkeypatch, _data())
    captured = fake_tuning.captured
    checkpoint = tmp_path / "edgetam.pt"
    checkpoint.write_bytes(b"checkpoint identity only; no inference during calibration")

    def calibrate(*args: object, **kwargs: object) -> kalman_module.KalmanCalibrationResult:
        captured["events"].append("calibrate")
        result = calibrate_kalman(*args, **kwargs)
        captured["calibration"] = result
        return result

    monkeypatch.setattr(kalman_module, "calibrate_kalman", calibrate)
    args = fake_tuning.args(
        tracker="occluboost",
        variable_dt=False,
        edgetam=True,
        mask_guidance_weights=checkpoint,
        mask_guidance_max_objects=64,
        device="cpu",
        _build_validated=True,
    )
    driver = tuner_module.Tuner(args)
    driver.fit()

    assert len(calls) == 1
    assert calls[0].build_path == args.build_path
    assert captured["events"].count("calibrate") == 1
    assert captured["events"].index("calibrate") < captured["events"].index("ray_init")
    calibration = captured["calibration"]
    assert calibration.matched_detections == 12
    assert set(calibration.fitted_parameters) == set(KALMAN_NOISE_OPTIONS)
    saved = load_tracker_config("occluboost", calibration.config_path)
    defaults = load_tracker_config("occluboost")
    assert {key: saved[key] for key in MASK_GUIDANCE_OPTIONS} == {
        **{key: defaults[key] for key in MASK_GUIDANCE_OPTIONS},
        "edgetam.max_objects": args.mask_guidance_max_objects,
    }
    report = json.loads(calibration.report_path.read_text())
    assert not set(MASK_GUIDANCE_OPTIONS).intersection(report["tuning"]["fixed_options"])
    assert len(captured["trial_configs"]) == 2
    for config in captured["trial_configs"]:
        assert config["asso_func"] == "iou"
        assert config["edgetam.max_objects"] == 64
        assert {key: config[key] for key in KALMAN_NOISE_OPTIONS} == {
            key: saved[key] for key in KALMAN_NOISE_OPTIONS
        }
    schema = flatten_yaml_config(driver._yaml_cfg)
    assert schema["edgetam.max_objects"] == {"default": 64}
    assert "type" in schema["edgetam.min_coverage"]


@pytest.mark.parametrize(
    "tracker, overrides",
    [
        ("ocsort", {"kalman.noise.process_velocity_scale": 2.5}),
        ("deepocsort", {"kalman.noise.process_velocity_scale": 3.5}),
        ("boosttrack", {"kalman.adaptive_kf": True}),
        ("occluboost", {"kalman.adaptive_kf": False}),
    ],
)
@pytest.mark.parametrize("calibrate_kf", [False, True])
def test_kalman_options_stay_fixed_with_or_without_calibration(
    fake_tuning: SimpleNamespace, tracker: str, overrides: dict, calibrate_kf: bool
) -> None:
    tuner = tuner_module.Tuner(fake_tuning.args(tracker=tracker, calibrate_kf=calibrate_kf), baseline_config=overrides)

    tuner.fit()

    for key, expected in overrides.items():
        if calibrate_kf:
            assert fake_tuning.captured["calibrations"][0][key] == expected
        resolved = fake_tuning.captured["calibrated_config"][key] if calibrate_kf else expected
        assert set(flatten_yaml_config(tuner._yaml_cfg)[key]) == {"default"}
        assert fake_tuning.captured["param_space"][key] == resolved
        assert fake_tuning.captured["postprocess_base"][key] == resolved
        assert all(config[key] == resolved for config in fake_tuning.captured["trial_configs"])

    removed = {"Q_xy_scaling", "Q_s_scaling", "Q_a_scaling"}
    assert not removed.intersection(fake_tuning.captured["postprocess_base"])
    assert all(not removed.intersection(config) for config in fake_tuning.captured["trial_configs"])
    if calibrate_kf:
        saved_config = yaml.safe_load(fake_tuning.captured["config_path"].read_text())
        report = json.loads(fake_tuning.captured["report_path"].read_text())
        assert not removed.intersection(saved_config)
        assert not removed.intersection(report["tuning"]["fixed_options"])


@pytest.mark.parametrize("search_alg", ["optuna", "random"])
@pytest.mark.parametrize("use_profile", [False, True])
def test_plain_tuning_preserves_kf_defaults_or_scalar_profile_while_searching_tracker_settings(
    fake_tuning: SimpleNamespace, tmp_path: Path, search_alg: str, use_profile: bool
) -> None:
    profile = None
    expected = {
        **dict.fromkeys(KALMAN_NOISE_OPTIONS, 1.0),
        "kalman.variable_dt": True,
        "kalman.noise.time_unit": "seconds",
        "kalman.noise.reference_dt_s": DEFAULT_REFERENCE_DT_S,
    }
    if use_profile:
        expected.update(dict(zip(KALMAN_NOISE_OPTIONS, (1e-6, 240.0, 0.003, 550.0, 1.3), strict=True)))
        expected["kalman.noise.reference_dt_s"] = 0.05
        profile = tmp_path / "calibrated.yaml"
        profile.write_text(yaml.safe_dump({"tracker": "botsort", **expected}), encoding="utf-8")
    args = fake_tuning.args(
        calibrate_kf=False,
        search_alg=search_alg,
        tracker_config=profile,
        variable_dt=None if use_profile else True,
    )
    tuner = tuner_module.Tuner(args)

    tuner.fit()

    assert fake_tuning.captured["calibrations"] == []
    schema = flatten_yaml_config(tuner._yaml_cfg)
    assert all(set(schema[key]) == {"default"} for key in expected)
    for key, value in expected.items():
        assert fake_tuning.captured["param_space"][key] == value
        assert fake_tuning.captured["postprocess_base"][key] == value
        assert all(config[key] == value for config in fake_tuning.captured["trial_configs"])
    if search_alg == "optuna":
        trial = _SuggestionTrial()
        fake_tuning.captured["search_space"](trial)
        assert not set(expected).intersection(trial.params)
        assert "match_thresh" in trial.params
    else:
        assert isinstance(fake_tuning.captured["param_space"]["match_thresh"], _Domain)


def test_programmatic_baseline_and_custom_config_are_preserved_during_calibration(
    fake_tuning: SimpleNamespace, tmp_path: Path
) -> None:
    original_config = tmp_path / "tracker.yaml"
    original_config.write_text(yaml.safe_dump({"tracker": "botsort", "track_buffer": 61, "match_thresh": 0.8}))
    args = fake_tuning.args(tracker_config=original_config)
    baseline = {"match_thresh": 0.55, "use_cmc": False, "kalman.noise.measurement_noise_scale": 8.0}

    tuner_module.Tuner(args, baseline_config=baseline).fit()

    observed = fake_tuning.captured["calibrations"][0]
    assert observed["track_buffer"] == 61
    assert observed["match_thresh"] == 0.55
    assert observed["use_cmc"] is False
    assert observed["kalman.noise.measurement_noise_scale"] == 8.0
    assert fake_tuning.captured["postprocess_base"]["match_thresh"] == 0.55
    assert baseline == {"match_thresh": 0.55, "use_cmc": False, "kalman.noise.measurement_noise_scale": 8.0}
    assert yaml.safe_load(original_config.read_text())["match_thresh"] == 0.8


def test_calibration_and_resume_are_rejected_before_any_job_starts(fake_tuning: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="resume"):
        tuner_module.Tuner(fake_tuning.args(resume_tune="botsort_1")).fit()

    assert fake_tuning.captured["events"] == []


def test_resuming_calibrated_run_loads_saved_units_and_frozen_values_without_refitting(
    fake_tuning: SimpleNamespace,
) -> None:
    captured = fake_tuning.captured
    first = tuner_module.Tuner(fake_tuning.args(), baseline_config={"kalman.noise.reference_dt_s": 0.05})
    _, tune_dir, _, _ = first.fit()
    captured["restore_enabled"] = True
    resumed_args = fake_tuning.args(calibrate_kf=False, resume_tune=tune_dir, variable_dt=None)
    resumed = tuner_module.Tuner(resumed_args)

    resumed.fit()

    assert captured["events"].count("calibrate") == 1
    assert captured["events"].count("restore") == 1
    assert Path(resumed_args.tracker_config) == captured["config_path"]
    for key in (*KALMAN_NOISE_OPTIONS, "kalman.variable_dt", "kalman.noise.time_unit", "kalman.noise.reference_dt_s"):
        expected = captured["calibrated_config"][key]
        assert resumed._runtime_config[key] == expected
        assert flatten_yaml_config(resumed._yaml_cfg)[key] == {"default": expected}
        assert all(config[key] == expected for config in captured["trial_configs"])


@pytest.mark.parametrize("eval_masks", [False, True])
def test_tuning_persists_evaluation_mode_and_resumes_with_matching_trial_geometry(
    fake_tuning: SimpleNamespace, eval_masks: bool
) -> None:
    captured = fake_tuning.captured
    args = fake_tuning.args(
        calibrate_kf=False,
        eval_masks=eval_masks,
        evaluation_config={"layout": "sequence", "annotation_layout": "mots_png"},
    )
    _, tune_dir, _, _ = tuner_module.Tuner(args).fit()

    assert json.loads((tune_dir / "evaluation.json").read_text()) == {"eval_masks": eval_masks}
    assert captured["trial_args"].eval_masks is eval_masks
    captured["restore_enabled"] = True
    tuner_module.Tuner(
        fake_tuning.args(
            calibrate_kf=False,
            eval_masks=eval_masks,
            resume_tune=tune_dir,
            evaluation_config={"layout": "sequence", "annotation_layout": "mots_png"},
        )
    ).fit()

    assert captured["trial_args"].eval_masks is eval_masks
    assert captured["events"].count("restore") == 1
    assert captured["events"].count("fit") == 2


@pytest.mark.parametrize("saved_masks", [False, True])
def test_tuning_rejects_changed_evaluation_mode_before_starting_resume_runtime(
    fake_tuning: SimpleNamespace, saved_masks: bool
) -> None:
    captured = fake_tuning.captured
    args = fake_tuning.args(
        calibrate_kf=False,
        eval_masks=saved_masks,
        evaluation_config={"layout": "sequence", "annotation_layout": "mots_png"},
    )
    _, tune_dir, _, _ = tuner_module.Tuner(args).fit()
    captured["restore_enabled"] = True
    captured["events"].clear()

    with pytest.raises(ValueError, match="same evaluation mode"):
        tuner_module.Tuner(
            fake_tuning.args(
                calibrate_kf=False,
                eval_masks=not saved_masks,
                resume_tune=tune_dir,
                evaluation_config={"layout": "sequence", "annotation_layout": "mots_png"},
            )
        ).fit()

    assert captured["events"] == ["eval_setup"]
    assert json.loads((tune_dir / "evaluation.json").read_text())["eval_masks"] is saved_masks


@pytest.mark.parametrize("contents", [None, "{", "[]", '{"eval_masks": 1}', '{"eval_masks": "false"}'])
def test_tuning_rejects_missing_or_corrupt_evaluation_mode_on_resume(
    fake_tuning: SimpleNamespace, contents: str | None
) -> None:
    captured = fake_tuning.captured
    args = fake_tuning.args(
        calibrate_kf=False, evaluation_config={"layout": "sequence", "annotation_layout": "mots_png"}
    )
    _, tune_dir, _, _ = tuner_module.Tuner(args).fit()
    path = tune_dir / "evaluation.json"
    if contents is None:
        path.unlink()
    else:
        path.write_text(contents)
    captured["restore_enabled"] = True
    captured["events"].clear()

    with pytest.raises(ValueError, match="evaluation mode metadata"):
        tuner_module.Tuner(
            fake_tuning.args(
                calibrate_kf=False,
                resume_tune=tune_dir,
                evaluation_config={"layout": "sequence", "annotation_layout": "mots_png"},
            )
        ).fit()

    assert captured["events"] == ["eval_setup"]


@pytest.mark.parametrize("search_alg", ["optuna", "random"])
def test_direct_calibration_preserves_small_and_large_covariances_as_exact_constants(
    fake_tuning: SimpleNamespace, search_alg: str
) -> None:
    # Statistical calibration has only a numerical positivity floor and does
    # not constrain covariance scales to an arbitrary search interval.
    fitted = dict(zip(KALMAN_NOISE_OPTIONS, (1e-6, 240.0, 0.003, 550.0, 1.3), strict=True))
    fake_tuning.captured["fitted_values"] = fitted

    tuner_module.Tuner(fake_tuning.args(search_alg=search_alg)).fit()

    for key, expected in fitted.items():
        assert fake_tuning.captured["param_space"][key] == expected
        assert all(config[key] == expected for config in fake_tuning.captured["trial_configs"])


def test_resume_rejects_saved_trials_that_disagree_with_the_calibrated_noise(fake_tuning: SimpleNamespace) -> None:
    captured = fake_tuning.captured
    _, tune_dir, _, _ = tuner_module.Tuner(fake_tuning.args()).fit()
    captured["restore_enabled"] = True
    captured["saved_results"][0].config["kalman.noise.measurement_noise_scale"] = 999.0

    with pytest.raises(ValueError, match="fixed KF calibration"):
        tuner_module.Tuner(fake_tuning.args(calibrate_kf=False, resume_tune=tune_dir, variable_dt=None)).fit()

    assert captured["events"].count("calibrate") == 1
    assert captured["events"].count("fit") == 1


@pytest.mark.parametrize("search_alg", ["optuna", "random"])
@pytest.mark.parametrize("calibrate_kf", [False, True])
def test_selected_scale_is_searchable_and_resumes_with_its_original_baseline(
    fake_tuning: SimpleNamespace, search_alg: str, calibrate_kf: bool
) -> None:
    args = fake_tuning.args(search_alg=search_alg, calibrate_kf=calibrate_kf, tune_kf=("measurement_noise_scale",))
    driver = tuner_module.Tuner(args)
    _, directory, *_ = driver.fit()
    captured = fake_tuning.captured
    key = "kalman.noise.measurement_noise_scale"
    baseline = driver._runtime_config[key]
    assert flatten_yaml_config(driver._yaml_cfg)[key] == {
        "type": "loguniform",
        "default": baseline,
        "range": [baseline / 4.0, baseline * 4.0],
    }
    assert all(baseline / 4.0 <= config[key] <= baseline * 4.0 for config in captured["trial_configs"])
    for fixed in set(KALMAN_NOISE_OPTIONS) - {key}:
        assert all(config[fixed] == driver._runtime_config[fixed] for config in captured["trial_configs"])
    metadata = json.loads((directory / "kf-refinement.json").read_text())
    assert metadata["fields"] == [key]
    assert metadata["base_options"][key] == baseline
    captured["restore_enabled"] = True
    resumed = tuner_module.Tuner(
        fake_tuning.args(
            search_alg=search_alg,
            calibrate_kf=False,
            resume_tune=directory,
            variable_dt=None if calibrate_kf else True,
            tune_kf=("measurement_noise_scale",),
        )
    )
    resumed.fit()
    assert captured["events"].count("calibrate") == int(calibrate_kf)
    assert resumed._runtime_config[key] == baseline
    assert key not in resumed._calibrated_fixed_options


def test_class_calibration_refines_selected_child_scale_and_freezes_every_other_prior(
    fake_tuning: SimpleNamespace,
) -> None:
    fitted = {name: float(index + 2) for index, name in enumerate(KALMAN_NOISE_OPTIONS)}
    fitted.update(
        {f"kalman.noise.by_class.1.{key.rsplit('.', 1)[-1]}": value * 3 for key, value in list(fitted.items())}
    )
    fitted.update(
        {
            "kalman.noise.by_class.1.time_unit": "seconds",
            "kalman.noise.by_class.1.reference_dt_s": DEFAULT_REFERENCE_DT_S,
        }
    )
    fake_tuning.captured["fitted_values"] = fitted
    driver = tuner_module.Tuner(fake_tuning.args(per_class=True, tune_kf=("process_velocity_scale",)))
    _, directory, *_ = driver.fit()
    selected = "kalman.noise.by_class.1.process_velocity_scale"
    schema = flatten_yaml_config(driver._yaml_cfg)
    assert schema[selected]["type"] == "loguniform"
    assert schema[selected]["range"] == [fitted[selected] / 4, fitted[selected] * 4]
    assert schema["kalman.noise.process_velocity_scale"] == {"default": fitted["kalman.noise.process_velocity_scale"]}
    assert all(
        config[key] == value
        for config in fake_tuning.captured["trial_configs"]
        for key, value in fitted.items()
        if key != selected
    )
    fake_tuning.captured["restore_enabled"] = True
    tuner_module.Tuner(
        fake_tuning.args(
            per_class=True,
            tune_kf=("process_velocity_scale",),
            calibrate_kf=False,
            resume_tune=directory,
            variable_dt=None,
        )
    ).fit()
    assert fake_tuning.captured["events"].count("calibrate") == 1


def test_refinement_resume_rejects_corrupted_unselected_trial_prior(fake_tuning: SimpleNamespace) -> None:
    _, directory, *_ = tuner_module.Tuner(
        fake_tuning.args(calibrate_kf=False, tune_kf=("process_velocity_scale",))
    ).fit()
    fake_tuning.captured["saved_results"][0].config["kalman.noise.measurement_noise_scale"] = 999.0
    fake_tuning.captured["restore_enabled"] = True
    with pytest.raises(ValueError, match="fixed KF calibration"):
        tuner_module.Tuner(
            fake_tuning.args(
                calibrate_kf=False,
                resume_tune=directory,
                tune_kf=("process_velocity_scale",),
            )
        ).fit()
    assert fake_tuning.captured["events"].count("fit") == 1
