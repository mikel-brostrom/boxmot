import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
import yaml

import boxmot.engine.tuning.tuner as tuner_module
import boxmot.utils.rich.core.ui as ui_module
import boxmot.utils.rich.reporters.tune as tune_reporting
import boxmot.utils.rich.workflow.reporting as rich_reporting
from boxmot.engine.tuning.backends.optuna_backend import yaml_to_optuna_define_space
from boxmot.engine.tuning.postprocessing import write_trial_yaml
from boxmot.engine.tuning.search_space import default_tune_config, flatten_yaml_config


def test_nested_activates_flatten_defaults_and_optuna_children():
    yaml_cfg = {
        "parent": {
            "type": "choice",
            "default": True,
            "options": [False, True],
            "activates": {
                "child": {
                    "type": "choice",
                    "default": True,
                    "options": [False, True],
                    "activates": {
                        "leaf": {
                            "type": "uniform",
                            "default": 0.5,
                            "range": [0.1, 0.9],
                        }
                    },
                }
            },
        }
    }

    class _FakeTrial:
        def __init__(self):
            self.params = {}

        def suggest_categorical(self, param, options):
            value = True if True in options else options[0]
            self.params[param] = value
            return value

        def suggest_float(self, param, low, high, **kwargs):
            self.params[param] = low
            return low

        def suggest_int(self, param, low, high, **kwargs):
            self.params[param] = low
            return low

    flat = flatten_yaml_config(yaml_cfg)
    assert set(flat) == {"parent", "child", "leaf"}
    assert default_tune_config(yaml_cfg) == {
        "parent": True,
        "child": True,
        "leaf": 0.5,
    }

    trial = _FakeTrial()
    yaml_to_optuna_define_space(yaml_cfg)(trial)
    assert trial.params == {"parent": True, "child": True, "leaf": 0.1}


def test_write_trial_yaml_preserves_nested_activates(tmp_path):
    yaml_cfg = {
        "parent": {
            "type": "choice",
            "default": True,
            "options": [False, True],
            "activates": {
                "child": {
                    "type": "choice",
                    "default": True,
                    "options": [False, True],
                    "activates": {
                        "leaf": {
                            "type": "uniform",
                            "default": 0.5,
                            "range": [0.1, 0.9],
                        }
                    },
                }
            },
        },
        "plain": {
            "type": "uniform",
            "default": 1.0,
            "range": [0.0, 2.0],
        },
    }
    output_path = tmp_path / "best_tracker.yaml"

    write_trial_yaml(
        yaml_cfg,
        {"parent": True, "child": False, "leaf": 0.7, "plain": 1.5},
        output_path,
    )

    saved = yaml.safe_load(output_path.read_text())

    assert list(saved) == ["parent", "plain"]
    assert saved["parent"]["default"] is True
    assert saved["parent"]["activates"]["child"]["default"] is False
    assert (
        saved["parent"]["activates"]["child"]["activates"]["leaf"]["default"]
        == 0.7
    )
    assert saved["plain"]["default"] == 1.5


def test_tuner_uses_absolute_ray_paths_after_eval_setup(monkeypatch, tmp_path):
    captured = {}
    workflow_state = {"stopped": False}
    detail_updates: list[tuple[str | None, str | None]] = []
    (tmp_path / "runs" / "ray" / "mot17-mini" / "strongsort_1").mkdir(parents=True)

    class _FakeRequirementsChecker:
        def sync_extra(self, extra, verbose=True):
            captured["extra"] = extra

    monkeypatch.setattr(tuner_module, "load_yaml_config", lambda tracker_name: {})
    monkeypatch.setattr(tuner_module, "_save_all_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuner_module, "run_generate_dets_embs", lambda args: captured.setdefault("generated", True))
    monkeypatch.setattr(
        tuner_module,
        "eval_setup",
        lambda args, pipeline=None: (
            setattr(args, "project", (tmp_path / "runs").resolve()),
            setattr(args, "detector", [tmp_path / "yolox_x_mot17_ablation.pt"]),
            setattr(args, "reid", [tmp_path / "lmbn_n_duke.pt"]),
        ),
    )
    def _fake_tune_intro(args, **kwargs):
        captured["intro_detector"] = args.detector[0]
        captured["intro_reid"] = args.reid[0]
        return SimpleNamespace(
            _started=True,
            start=lambda: None,
            complete=lambda *a, **k: None,
            activate=lambda *a, **k: None,
            set_detail=lambda title, text, **k: detail_updates.append((title, text)),
            clear_detail=lambda *a, **k: None,
            set_detail_renderable=lambda *a, **k: None,
            transition=lambda *a, **k: None,
            stop=lambda: workflow_state.update(stopped=True),
            steps=[],
            fields=[],
        )

    def _fake_create_pipeline(reporter, **kwargs):
        workflow = _fake_tune_intro(reporter.args)

        class _FakePipeline:
            def __init__(self):
                self.workflow = workflow

            def advance(self, *a, **k):
                pass

            def start(self):
                pass

            def stop(self):
                workflow_state.update(stopped=True)

            def finish(self, *a, **k):
                pass

            def callback(self, *a, **k):
                return lambda msg: None

            def complete_step(self, *a, **k):
                pass

            def update(self, *a, **k):
                pass

            def refresh_fields(self, fields):
                for label, value in fields:
                    if label == "Detector":
                        captured["intro_detector"] = value
                    elif label == "ReID":
                        captured["intro_reid"] = value

            def step(self, *a, **k):
                return "fake"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self.stop()

        return _FakePipeline()

    monkeypatch.setattr(tune_reporting.TuneWorkflowReporter, "pipeline", _fake_create_pipeline)

    class _FakeOptunaSearch:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeRunConfig:
        def __init__(self, storage_path, name, callbacks=None, verbose=None, **kwargs):
            self.storage_path = storage_path
            self.name = name
            self.callbacks = callbacks
            self.verbose = verbose

    class _FakeTuneConfig:
        def __init__(self, num_samples, search_alg, trial_dirname_creator, **kwargs):
            self.num_samples = num_samples
            self.search_alg = search_alg
            self.trial_dirname_creator = trial_dirname_creator

    class _FakeTuner:
        @staticmethod
        def can_restore(path):
            captured["restore_path"] = path
            return False

        def __init__(self, trainable, param_space, tune_config, run_config):
            captured["storage_path"] = run_config.storage_path
            captured["run_name"] = run_config.name
            captured["callbacks"] = run_config.callbacks
            captured["verbose"] = run_config.verbose

        def fit(self):
            if captured.get("callbacks"):
                trial = SimpleNamespace(
                    trial_id="trial_001",
                    last_result={
                        "time_total_s": 1.0,
                        "HOTA": 50.0,
                        "MOTA": 45.0,
                        "IDF1": 40.0,
                        "AssA": 35.0,
                    },
                )
                for callback in captured["callbacks"]:
                    callback.on_trial_start(0, [], trial)
                    callback.on_trial_complete(0, [], trial)
            captured["fit"] = True

        def get_results(self):
            return []

    fake_tune = SimpleNamespace(
        Tuner=_FakeTuner,
        TuneConfig=_FakeTuneConfig,
        with_resources=lambda fn, resources: fn,
        Callback=object,
    )

    fake_ray = SimpleNamespace(
        tune=fake_tune,
        is_initialized=lambda: False,
        init=lambda **kwargs: captured.setdefault("ray_init_kwargs", kwargs),
    )

    import sys

    class _FakeConcurrencyLimiter:
        def __init__(self, searcher, max_concurrent):
            self.searcher = searcher
            self.max_concurrent = max_concurrent

    class _FakeFailureConfig:
        def __init__(self, **kwargs):
            pass

    class _FakeCheckpointConfig:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setitem(sys.modules, "boxmot.utils.checks", SimpleNamespace(RequirementsChecker=_FakeRequirementsChecker))
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.tune", SimpleNamespace(
        RunConfig=_FakeRunConfig, FailureConfig=_FakeFailureConfig, CheckpointConfig=_FakeCheckpointConfig))
    monkeypatch.setitem(sys.modules, "ray.tune.search", SimpleNamespace(ConcurrencyLimiter=_FakeConcurrencyLimiter))
    monkeypatch.setitem(sys.modules, "ray.tune.search.optuna", SimpleNamespace(OptunaSearch=_FakeOptunaSearch))

    args = SimpleNamespace(
        detector=[tmp_path / "yolov8n.pt"],
        reid=[tmp_path / "osnet_x0_25_msmt17.pt"],
        tracker="strongsort",
        data="mot17-mini",
        maximize=("HOTA",),
        minimize=(),
        objectives=("HOTA",),
        n_threads=1,
        n_trials=3,
        project=Path("runs"),
        verbose=False,
    )

    tuner_module.main(args)

    assert "restore_path" not in captured
    assert Path(captured["storage_path"]).is_absolute()
    assert Path(captured["storage_path"]) == (tmp_path / "runs" / "ray" / "mot17-mini").resolve()
    assert captured["run_name"] == "strongsort_2"
    assert Path(captured["intro_detector"]).name == "yolox_x_mot17_ablation.pt"
    assert Path(captured["intro_reid"]).name == "lmbn_n_duke.pt"
    assert captured["verbose"] == 0
    assert len(captured["callbacks"]) == 1
    assert captured["ray_init_kwargs"]["include_dashboard"] is False
    assert captured["ray_init_kwargs"]["logging_level"] == logging.ERROR
    assert captured["ray_init_kwargs"]["log_to_driver"] is False
    assert os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] == "0"
    assert any(title == tune_reporting.TUNE_OPTIMIZE_STEP for title, _ in detail_updates)
    assert captured["fit"] is True
    assert workflow_state["stopped"] is True


def test_tuner_keeps_workflow_state_out_of_ray_callback(monkeypatch, tmp_path):
    captured = {}

    class _FakeRequirementsChecker:
        def sync_extra(self, extra, verbose=True):
            captured["extra"] = extra

    monkeypatch.setattr(tuner_module, "load_yaml_config", lambda tracker_name: {})
    monkeypatch.setattr(tuner_module, "_save_all_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuner_module, "run_generate_dets_embs", lambda args: captured.setdefault("generated", True))
    monkeypatch.setattr(
        tuner_module,
        "eval_setup",
        lambda args, pipeline=None: setattr(args, "project", (tmp_path / "runs").resolve()),
    )
    def _fake_create_pipeline(reporter, **kwargs):
        wf = SimpleNamespace(
            _started=True,
            _lock=threading.RLock(),
            start=lambda: None,
            complete=lambda *a, **k: None,
            activate=lambda *a, **k: None,
            set_detail=lambda *a, **k: None,
            clear_detail=lambda *a, **k: None,
            set_detail_renderable=lambda *a, **k: None,
            transition=lambda *a, **k: None,
            stop=lambda: None,
        )
        class _FP:
            workflow = wf
            def advance(self, *a, **k): pass
            def start(self): pass
            def stop(self): pass
            def finish(self, *a, **k): pass
            def callback(self, *a, **k): return lambda msg: None
            def complete_step(self, *a, **k): pass
            def update(self, *a, **k): pass
            def refresh_fields(self, *a, **k): pass
            def step(self, *a, **k): return "fake"
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _FP()
    monkeypatch.setattr(tune_reporting.TuneWorkflowReporter, "pipeline", _fake_create_pipeline)

    class _FakeOptunaSearch:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeRunConfig:
        def __init__(self, storage_path, name, callbacks=None, verbose=None, **kwargs):
            self.storage_path = storage_path
            self.name = name
            self.callbacks = callbacks
            self.verbose = verbose

    class _FakeTuneConfig:
        def __init__(self, num_samples, search_alg, trial_dirname_creator, **kwargs):
            self.num_samples = num_samples
            self.search_alg = search_alg
            self.trial_dirname_creator = trial_dirname_creator

    class _FakeTuner:
        @staticmethod
        def can_restore(path):
            return False

        def __init__(self, trainable, param_space, tune_config, run_config):
            objective = next(
                cell.cell_contents
                for cell in trainable.__closure__ or ()
                if isinstance(cell.cell_contents, tuner_module.TrackerObjective)
            )
            captured["driver_lock_in_trainable_args"] = hasattr(objective.opt, "driver_lock")
            captured["callbacks"] = run_config.callbacks
            captured["callback_has_workflow_lock"] = any(
                hasattr(callback, "_lock")
                for callback in run_config.callbacks or []
            )
            captured["verbose"] = run_config.verbose

        def fit(self):
            captured["fit"] = True
            return []

        def get_results(self):
            return []

    fake_tune = SimpleNamespace(
        Tuner=_FakeTuner,
        TuneConfig=_FakeTuneConfig,
        with_resources=lambda fn, resources: fn,
        Callback=object,
    )

    fake_ray = SimpleNamespace(
        tune=fake_tune,
        is_initialized=lambda: False,
        init=lambda **kwargs: captured.setdefault("ray_init_kwargs", kwargs),
    )

    import sys

    class _FakeConcurrencyLimiter:
        def __init__(self, searcher, max_concurrent):
            self.searcher = searcher
            self.max_concurrent = max_concurrent

    class _FakeFailureConfig:
        def __init__(self, **kwargs):
            pass

    class _FakeCheckpointConfig:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setitem(sys.modules, "boxmot.utils.checks", SimpleNamespace(RequirementsChecker=_FakeRequirementsChecker))
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.tune", SimpleNamespace(
        RunConfig=_FakeRunConfig, FailureConfig=_FakeFailureConfig, CheckpointConfig=_FakeCheckpointConfig))
    monkeypatch.setitem(sys.modules, "ray.tune.search", SimpleNamespace(ConcurrencyLimiter=_FakeConcurrencyLimiter))
    monkeypatch.setitem(sys.modules, "ray.tune.search.optuna", SimpleNamespace(OptunaSearch=_FakeOptunaSearch))

    args = SimpleNamespace(
        detector=[tmp_path / "yolov8n.pt"],
        reid=[tmp_path / "osnet_x0_25_msmt17.pt"],
        tracker="strongsort",
        data="mot17-mini",
        maximize=("HOTA",),
        minimize=(),
        objectives=("HOTA",),
        n_threads=1,
        n_trials=3,
        project=Path("runs"),
        verbose=False,
        driver_lock=threading.RLock(),
    )

    tuner_module.main(args)

    assert captured["extra"] == "evolve"
    assert captured["driver_lock_in_trainable_args"] is False
    assert len(captured["callbacks"]) == 1
    assert captured["callback_has_workflow_lock"] is False
    assert captured["verbose"] == 0
    assert captured["fit"] is True


def test_tune_workflow_callback_is_pickle_safe_with_active_workflow() -> None:
    updates: list[tuple[str, str]] = []
    workflow = SimpleNamespace(
        _lock=threading.RLock(),
        set_detail=lambda title, detail: updates.append((title, detail)),
    )
    callback = tune_reporting.TuneWorkflowCallback(total=1, maximize=["HOTA"], minimize=[])

    assert isinstance(callback, rich_reporting.RichWorkflowCallback)
    assert isinstance(tune_reporting.TuneSilentReporter(), rich_reporting.SilentProgressReporter)

    tune_reporting.set_tune_progress_workflow(workflow)
    try:
        assert tuner_module._is_ray_pickle_safe(callback)
        trial = SimpleNamespace(trial_id="trial_001", last_result={"HOTA": 50.0})
        callback.setup(stop=None, num_samples=1, total_num_samples=1)
        callback.on_step_begin(0, [])
        callback.on_trial_start(0, [], trial)
        callback.on_trial_result(0, [], trial, {"HOTA": 25.0})
        callback.on_trial_save(0, [], trial)
        callback.on_trial_restore(0, [], trial)
        callback.on_trial_complete(0, [], trial)
        callback.on_trial_recover(0, [], trial)
        callback.on_checkpoint(0, [], trial, checkpoint=None)
        callback.on_step_end(1, [])
        callback.on_experiment_end([])
        assert callback.get_state() is None
        callback.set_state({})
    finally:
        tune_reporting.set_tune_progress_workflow(None)

    assert any(title == tune_reporting.TUNE_OPTIMIZE_STEP for title, _ in updates)


def test_tuner_resume_uses_absolute_ray_restore_path(monkeypatch, tmp_path):
    captured = {}

    class _FakeRequirementsChecker:
        def sync_extra(self, extra, verbose=True):
            captured["extra"] = extra

    monkeypatch.setattr(tuner_module, "load_yaml_config", lambda tracker_name: {})
    monkeypatch.setattr(tuner_module, "_save_all_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuner_module, "run_generate_dets_embs", lambda args: None)
    monkeypatch.setattr(
        tuner_module,
        "eval_setup",
        lambda args, pipeline=None: setattr(args, "project", (tmp_path / "runs").resolve()),
    )
    def _fake_create_pipeline(reporter, **kwargs):
        wf = SimpleNamespace(
            _started=True,
            start=lambda: None,
            complete=lambda *a, **k: None,
            activate=lambda *a, **k: None,
            set_detail=lambda *a, **k: None,
            clear_detail=lambda *a, **k: None,
            set_detail_renderable=lambda *a, **k: None,
            transition=lambda *a, **k: None,
            stop=lambda: None,
        )
        class _FP:
            workflow = wf
            def advance(self, *a, **k): pass
            def start(self): pass
            def stop(self): pass
            def finish(self, *a, **k): pass
            def callback(self, *a, **k): return lambda msg: None
            def complete_step(self, *a, **k): pass
            def update(self, *a, **k): pass
            def refresh_fields(self, *a, **k): pass
            def step(self, *a, **k): return "fake"
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _FP()
    monkeypatch.setattr(tune_reporting.TuneWorkflowReporter, "pipeline", _fake_create_pipeline)

    class _FakeOptunaSearch:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeRunConfig:
        def __init__(self, storage_path, name, callbacks=None, verbose=None, **kwargs):
            self.storage_path = storage_path
            self.name = name
            self.callbacks = callbacks
            self.verbose = verbose

    class _FakeTuneConfig:
        def __init__(self, num_samples, search_alg, trial_dirname_creator, **kwargs):
            self.num_samples = num_samples
            self.search_alg = search_alg
            self.trial_dirname_creator = trial_dirname_creator

    class _FakeTuner:
        @staticmethod
        def can_restore(path):
            captured["restore_path"] = path
            return False

        def __init__(self, trainable, param_space, tune_config, run_config):
            captured["storage_path"] = run_config.storage_path
            captured["run_name"] = run_config.name

        def fit(self):
            return []

        def get_results(self):
            return []

    fake_tune = SimpleNamespace(
        Tuner=_FakeTuner,
        TuneConfig=_FakeTuneConfig,
        with_resources=lambda fn, resources: fn,
        Callback=object,
    )

    fake_ray = SimpleNamespace(
        tune=fake_tune,
        is_initialized=lambda: False,
        init=lambda **kwargs: captured.setdefault("ray_init_kwargs", kwargs),
    )

    import sys

    class _FakeConcurrencyLimiter:
        def __init__(self, searcher, max_concurrent):
            self.searcher = searcher
            self.max_concurrent = max_concurrent

    class _FakeFailureConfig:
        def __init__(self, **kwargs):
            pass

    class _FakeCheckpointConfig:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setitem(sys.modules, "boxmot.utils.checks", SimpleNamespace(RequirementsChecker=_FakeRequirementsChecker))
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.tune", SimpleNamespace(
        RunConfig=_FakeRunConfig, FailureConfig=_FakeFailureConfig, CheckpointConfig=_FakeCheckpointConfig))
    monkeypatch.setitem(sys.modules, "ray.tune.search", SimpleNamespace(ConcurrencyLimiter=_FakeConcurrencyLimiter))
    monkeypatch.setitem(sys.modules, "ray.tune.search.optuna", SimpleNamespace(OptunaSearch=_FakeOptunaSearch))

    args = SimpleNamespace(
        detector=[tmp_path / "yolov8n.pt"],
        reid=[tmp_path / "osnet_x0_25_msmt17.pt"],
        tracker="strongsort",
        data="mot17-mini",
        maximize=("HOTA",),
        minimize=(),
        objectives=("HOTA",),
        n_threads=1,
        n_trials=3,
        project=Path("runs"),
        verbose=False,
        resume_tune="strongsort_1",
    )

    tuner_module.main(args)

    assert Path(captured["restore_path"]).is_absolute()
    assert Path(captured["restore_path"]) == (tmp_path / "runs" / "ray" / "mot17-mini" / "strongsort_1").resolve()
    assert captured["run_name"] == "strongsort_1"


def test_tuner_splits_comma_separated_optimization_metrics(monkeypatch, tmp_path):
    captured = {}

    class _FakeRequirementsChecker:
        def sync_extra(self, extra, verbose=True):
            captured["extra"] = extra

    monkeypatch.setattr(tuner_module, "load_yaml_config", lambda tracker_name: {})
    monkeypatch.setattr(tuner_module, "_save_all_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuner_module, "run_generate_dets_embs", lambda args: None)
    monkeypatch.setattr(
        tuner_module,
        "eval_setup",
        lambda args, pipeline=None: setattr(args, "project", (tmp_path / "runs").resolve()),
    )
    def _fake_create_pipeline(reporter, **kwargs):
        wf = SimpleNamespace(
            start=lambda: None,
            complete=lambda *a, **k: None,
            activate=lambda *a, **k: None,
            set_detail=lambda *a, **k: None,
            set_detail_renderable=lambda *a, **k: None,
            transition=lambda *a, **k: None,
            stop=lambda: None,
        )
        class _FP:
            workflow = wf
            def advance(self, *a, **k): pass
            def start(self): pass
            def stop(self): pass
            def finish(self, *a, **k): pass
            def callback(self, *a, **k): return lambda msg: None
            def complete_step(self, *a, **k): pass
            def update(self, *a, **k): pass
            def refresh_fields(self, *a, **k): pass
            def step(self, *a, **k): return "fake"
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _FP()
    monkeypatch.setattr(tune_reporting.TuneWorkflowReporter, "pipeline", _fake_create_pipeline)

    class _FakeOptunaSearch:
        def __init__(self, **kwargs):
            captured["optuna_kwargs"] = kwargs

    class _FakeRunConfig:
        def __init__(self, storage_path, name, callbacks=None, verbose=None, **kwargs):
            self.storage_path = storage_path
            self.name = name
            self.callbacks = callbacks
            self.verbose = verbose

    class _FakeTuneConfig:
        def __init__(self, num_samples, search_alg, trial_dirname_creator, **kwargs):
            self.num_samples = num_samples
            self.search_alg = search_alg
            self.trial_dirname_creator = trial_dirname_creator

    class _FakeTuner:
        @staticmethod
        def can_restore(path):
            return False

        def __init__(self, trainable, param_space, tune_config, run_config):
            captured["callbacks"] = run_config.callbacks

        def fit(self):
            return []

        def get_results(self):
            return []

    fake_tune = SimpleNamespace(
        Tuner=_FakeTuner,
        TuneConfig=_FakeTuneConfig,
        with_resources=lambda fn, resources: fn,
        Callback=object,
    )
    fake_ray = SimpleNamespace(
        tune=fake_tune,
        is_initialized=lambda: False,
        init=lambda **kwargs: captured.setdefault("ray_init_kwargs", kwargs),
    )

    import sys

    class _FakeConcurrencyLimiter:
        def __init__(self, searcher, max_concurrent):
            self.searcher = searcher
            self.max_concurrent = max_concurrent

    class _FakeFailureConfig:
        def __init__(self, **kwargs):
            pass

    class _FakeCheckpointConfig:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setitem(sys.modules, "boxmot.utils.checks", SimpleNamespace(RequirementsChecker=_FakeRequirementsChecker))
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.tune", SimpleNamespace(
        RunConfig=_FakeRunConfig, FailureConfig=_FakeFailureConfig, CheckpointConfig=_FakeCheckpointConfig))
    monkeypatch.setitem(sys.modules, "ray.tune.search", SimpleNamespace(ConcurrencyLimiter=_FakeConcurrencyLimiter))
    monkeypatch.setitem(sys.modules, "ray.tune.search.optuna", SimpleNamespace(OptunaSearch=_FakeOptunaSearch))

    args = SimpleNamespace(
        detector=[tmp_path / "yolox_x_MOT17_ablation.pt"],
        reid=[tmp_path / "lmbn_n_duke.pt"],
        tracker="botsort",
        data="mot17-mini",
        maximize=("HOTA,MOTA,IDF1",),
        minimize=("IDSW_rate",),
        objectives=("HOTA",),
        n_threads=1,
        n_trials=100,
        project=Path("runs"),
        verbose=False,
    )

    tuner_module.main(args)

    assert captured["optuna_kwargs"]["metric"] == ["HOTA", "MOTA", "IDF1", "IDSW_rate"]
    assert captured["optuna_kwargs"]["mode"] == ["max", "max", "max", "min"]
    assert args.maximize == ("HOTA", "MOTA", "IDF1")
    assert args.minimize == ("IDSW_rate",)


def test_tuner_rejects_invalid_metric_names_before_ray_setup() -> None:
    args = SimpleNamespace(
        maximize=("HOTA", "MOTA"),
        minimize=("IDSWs",),
        objectives=(),
    )

    with pytest.raises(click.UsageError) as exc_info:
        tuner_module.main(args)

    message = str(exc_info.value)
    assert "Invalid value for --minimize: IDSWs" in message
    assert "(did you mean IDSW, IDSW_rate?)" in message
    assert "Available maximize metrics: HOTA, MOTA, IDF1, AssA, AssRe" in message
    assert "Available minimize metrics: IDSW, IDs, IDSW_rate" in message


def test_tuner_normalizes_metric_aliases_before_validation() -> None:
    args = SimpleNamespace(
        maximize=("hota",),
        minimize=("id_switches",),
        objectives=("hota", "id_switches"),
    )
    tuner = tuner_module.Tuner(args)

    tuner._resolve_metrics()

    assert args.objectives == ("HOTA", "IDSW")
    assert args.maximize == ("HOTA",)
    assert args.minimize == ("IDSW",)


def test_tuner_renders_sequence_metric_deltas_against_default_config(monkeypatch, tmp_path):
    captured = {}
    tune_dir = tmp_path / "runs" / "ray" / "bytetrack_tune"
    baseline_dir = tune_dir / "trial_baseline"
    best_dir = tune_dir / "trial_best"
    baseline_dir.mkdir(parents=True)
    best_dir.mkdir(parents=True)

    yaml_cfg = {
        "track_thresh": {"type": "uniform", "default": 0.6, "range": [0.3, 0.7]},
        "track_buffer": {"type": "qrandint", "default": 30, "range": [10, 61, 10]},
    }

    class _FakeRequirementsChecker:
        def sync_extra(self, extra, verbose=True):
            captured["extra"] = extra

    def _metrics(trial_id, hota, mota, idsw, config, path):
        row = {
            "HOTA": hota,
            "MOTA": mota,
            "IDF1": 70.0,
            "AssA": 60.0,
            "AssRe": 65.0,
            "IDSW": idsw,
            "IDs": 20,
        }
        raw = {"person": {**row, "per_sequence": {"MOT17-02": row}}}
        return SimpleNamespace(
            error=None,
            path=str(path),
            config=config,
            metrics={
                "trial_id": trial_id,
                **row,
                "IDSW_rate": idsw / 20,
                "_validation": {
                    "benchmark": "mot17-mini",
                    "raw": raw,
                    "summary_label": "single_class",
                    "summary": row,
                    "timings": {},
                },
            },
        )

    monkeypatch.setattr(tuner_module, "load_yaml_config", lambda tracker_name: yaml_cfg)
    monkeypatch.setattr(tuner_module, "_resolve_tune_dir", lambda args, resume=False: tune_dir)
    monkeypatch.setattr(tuner_module, "run_generate_dets_embs", lambda args: None)
    monkeypatch.setattr(
        tuner_module,
        "eval_setup",
        lambda args, pipeline=None: setattr(args, "project", (tmp_path / "runs").resolve()),
    )

    def _set_detail_renderable(title, renderable, **kwargs):
        captured["detail_title"] = title
        captured["detail_rendered"] = ui_module.capture_renderable(renderable, width=150)

    def _fake_create_pipeline_2(reporter, **kwargs):
        fake_workflow = SimpleNamespace(
            start=lambda: None,
            complete=lambda *a, **k: None,
            activate=lambda *a, **k: None,
            set_detail=lambda *a, **k: None,
            set_detail_renderable=_set_detail_renderable,
            transition=lambda *a, **k: None,
            stop=lambda: None,
            steps=[],
            fields=[],
        )

        class _FakePipeline2:
            def __init__(self):
                self.workflow = fake_workflow

            def advance(self, *a, **k):
                pass

            def start(self):
                pass

            def stop(self):
                pass

            def finish(self, renderable=None, **k):
                if renderable is not None:
                    title = k.get("title", "Results")
                    _set_detail_renderable(title, renderable)

            def callback(self, *a, **k):
                return lambda msg: None

            def complete_step(self, *a, **k):
                pass

            def update(self, *a, **k):
                pass

            def refresh_fields(self, *a, **k):
                pass

            def step(self, *a, **k):
                return "fake"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        return _FakePipeline2()

    monkeypatch.setattr(tune_reporting.TuneWorkflowReporter, "pipeline", _fake_create_pipeline_2)

    class _FakeOptunaSearch:
        def __init__(self, **kwargs):
            captured["optuna_kwargs"] = kwargs

    class _FakeRunConfig:
        def __init__(self, storage_path, name, callbacks=None, verbose=None, progress_reporter=None, **kwargs):
            self.storage_path = storage_path
            self.name = name
            self.callbacks = callbacks
            self.verbose = verbose
            self.progress_reporter = progress_reporter

    class _FakeTuneConfig:
        def __init__(self, num_samples, search_alg, trial_dirname_creator, **kwargs):
            self.num_samples = num_samples
            self.search_alg = search_alg
            self.trial_dirname_creator = trial_dirname_creator

    class _FakeTuner:
        @staticmethod
        def can_restore(path):
            return False

        def __init__(self, trainable, param_space, tune_config, run_config):
            captured["param_space"] = param_space

        def fit(self):
            return [
                _metrics("baseline", 50.0, 60.0, 10, {"track_thresh": 0.6, "track_buffer": 30}, baseline_dir),
                _metrics("best", 52.0, 61.0, 7, {"track_thresh": 0.65, "track_buffer": 40}, best_dir),
            ]

    fake_tune = SimpleNamespace(
        Tuner=_FakeTuner,
        TuneConfig=_FakeTuneConfig,
        with_resources=lambda fn, resources: fn,
        uniform=lambda *args: ("uniform", args),
        qrandint=lambda *args: ("qrandint", args),
        Callback=object,
    )
    fake_ray = SimpleNamespace(
        tune=fake_tune,
        is_initialized=lambda: False,
        init=lambda **kwargs: captured.setdefault("ray_init_kwargs", kwargs),
    )

    import sys

    class _FakeConcurrencyLimiter:
        def __init__(self, searcher, max_concurrent):
            self.searcher = searcher
            self.max_concurrent = max_concurrent

    class _FakeFailureConfig:
        def __init__(self, **kwargs):
            pass

    class _FakeCheckpointConfig:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setitem(sys.modules, "boxmot.utils.checks", SimpleNamespace(RequirementsChecker=_FakeRequirementsChecker))
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.tune", SimpleNamespace(
        RunConfig=_FakeRunConfig, FailureConfig=_FakeFailureConfig, CheckpointConfig=_FakeCheckpointConfig))
    monkeypatch.setitem(sys.modules, "ray.tune.search", SimpleNamespace(ConcurrencyLimiter=_FakeConcurrencyLimiter))
    monkeypatch.setitem(sys.modules, "ray.tune.search.optuna", SimpleNamespace(OptunaSearch=_FakeOptunaSearch))

    args = SimpleNamespace(
        detector=[tmp_path / "yolov8n.pt"],
        reid=[tmp_path / "osnet_x0_25_msmt17.pt"],
        tracker="bytetrack",
        data="mot17-mini",
        maximize=("HOTA",),
        minimize=(),
        objectives=("HOTA",),
        n_threads=1,
        n_trials=2,
        project=Path("runs"),
        verbose=False,
    )

    tuner_module.main(args)

    assert captured["optuna_kwargs"]["points_to_evaluate"] == [{"track_thresh": 0.6, "track_buffer": 30}]
    assert captured["detail_title"] == "Results"
    assert "MOT17-02" in captured["detail_rendered"]
    assert "(+2.00)" in captured["detail_rendered"]
    assert "(+1.00)" in captured["detail_rendered"]
    assert "(-3)" in captured["detail_rendered"]


def test_build_tune_workflow_fields_uses_benchmark_data() -> None:
    args = SimpleNamespace(
        tracker="bytetrack",
        detector=[Path("yolov8n.pt")],
        reid=[Path("osnet_x0_25_msmt17.pt")],
        data="mot17-mini",
        benchmark="",
        n_trials=3,
    )

    fields = dict(tune_reporting.build_tune_workflow_fields(args, maximize=["HOTA"], minimize=[]))

    assert fields["Dataset"] == "mot17-mini"


def test_build_tune_workflow_fields_show_pareto_objectives() -> None:
    args = SimpleNamespace(
        tracker="ocsort",
        detector=[Path("yolo11s-obb.pt")],
        reid=[Path("lmbn_n_duke.pt")],
        data="mmot-mini",
        benchmark="mmot-mini",
        n_trials=10,
    )

    fields = dict(
        tune_reporting.build_tune_workflow_fields(
            args,
            maximize=["HOTA", "MOTA"],
            minimize=["IDSW_rate"],
        )
    )

    assert fields["Objective"] == "Pareto: max HOTA, MOTA / min IDSW_rate"


def test_tune_workflow_renderable_is_compact_and_complete() -> None:
    args = SimpleNamespace(
        tracker="ocsort",
        detector=[Path("yolo11s-obb.pt")],
        reid=[Path("lmbn_n_duke.pt")],
        data="mmot-mini",
        benchmark="mmot-mini",
        n_trials=10,
    )
    workflow = tune_reporting.log_tune_pipeline_intro(
        args,
        maximize=["HOTA", "MOTA"],
        minimize=["IDSW_rate"],
    )
    workflow.complete(tune_reporting.TUNE_SETUP_STEP, render=False)
    workflow.complete(tune_reporting.TUNE_GENERATE_STEP, render=False)
    workflow.activate(tune_reporting.TUNE_OPTIMIZE_STEP, render=False)
    workflow.set_detail(
        tune_reporting.TUNE_OPTIMIZE_STEP,
        "Tune     20%  (2/10)  running trial 3/10  remaining 00:34",
        render=False,
    )

    rendered = ui_module.capture_renderable(workflow.renderable(compact=True), width=180)

    assert rendered.count("\n") + 1 <= 12
    assert "Setup" in rendered
    assert "Pipeline" in rendered
    assert "OBJECTIVE" in rendered
    assert "Pareto: max HOTA, MOTA / min IDSW_rate" in rendered
    assert "[✓] Setup / [✓] Generate / [>] Optimize" in rendered
    assert "Tune     20%  (2/10)  running trial 3/10  remaining 00:34" in rendered


def test_build_tune_artifacts_renderable_lists_paths() -> None:
    saved_artifacts = {
        "csv_path": Path("/tmp/results.csv"),
        "best_trial_id": "fe35bbe6",
        "best_yaml_path": Path("/tmp/best_bytetrack.yaml"),
        "summary_path": Path("/tmp/summary.md"),
    }

    rendered = ui_module.capture_renderable(
        tune_reporting.build_tune_artifacts_renderable(saved_artifacts),
        width=140,
    )

    assert "Saved Artifacts" in rendered
    assert "Results CSV" in rendered
    assert str(saved_artifacts["csv_path"]) in rendered
    assert "Best config (fe35bbe6)" in rendered
    assert str(saved_artifacts["best_yaml_path"]) in rendered
    assert "Summary" in rendered
    assert str(saved_artifacts["summary_path"]) in rendered


def test_generate_summary_handles_categorical_choice_params(tmp_path) -> None:
    trial_data = [
        {
            "trial_id": "trial_1",
            "metrics": {"HOTA": 68.0, "MOTA": 77.0, "IDF1": 79.0},
            "config": {"track_high_thresh": 0.5, "cmc_method": "ecc"},
        },
        {
            "trial_id": "trial_2",
            "metrics": {"HOTA": 65.0, "MOTA": 75.0, "IDF1": 76.0},
            "config": {"track_high_thresh": 0.4, "cmc_method": "sof"},
        },
        {
            "trial_id": "trial_3",
            "metrics": {"HOTA": 64.0, "MOTA": 74.0, "IDF1": 75.0},
            "config": {"track_high_thresh": 0.6, "cmc_method": "ecc"},
        },
    ]
    yaml_cfg = {
        "track_high_thresh": {"type": "uniform", "default": 0.45, "range": [0.3, 0.7]},
        "cmc_method": {"type": "choice", "default": "sof", "options": ["sof", "ecc"]},
    }
    args = SimpleNamespace(
        detector=[Path("yolov8n.pt")],
        benchmark="mot17-mini",
        data="mot17-mini",
    )

    summary_path = tuner_module._generate_summary(
        tmp_path,
        trial_data,
        yaml_cfg,
        "botsort",
        ["HOTA"],
        [],
        args,
        emit_logs=False,
    )

    rendered = summary_path.read_text()
    assert "| cmc_method | ['sof', 'ecc'] | ecc: 2, sof: 1 | — | categorical |" in rendered
