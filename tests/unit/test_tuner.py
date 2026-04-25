import logging
from pathlib import Path
import threading
from types import SimpleNamespace

import boxmot.engine.tuner as tuner_module
import boxmot.utils.ui as ui_module


def test_tuner_uses_absolute_ray_paths_after_eval_setup(monkeypatch, tmp_path):
    captured = {}
    workflow_state = {"stopped": False}
    detail_updates: list[tuple[str | None, str | None]] = []
    (tmp_path / "runs" / "ray" / "strongsort_tune").mkdir(parents=True)

    class _FakeRequirementsChecker:
        def sync_extra(self, extra, verbose=True):
            captured["extra"] = extra

    monkeypatch.setattr(tuner_module, "load_yaml_config", lambda tracker_name: {})
    monkeypatch.setattr(tuner_module, "_save_all_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuner_module, "run_generate_dets_embs", lambda args: captured.setdefault("generated", True))
    monkeypatch.setattr(
        tuner_module,
        "eval_setup",
        lambda args, workflow=None: setattr(args, "project", (tmp_path / "runs").resolve()),
    )
    monkeypatch.setattr(
        tuner_module,
        "_log_tune_pipeline_intro",
        lambda *args, **kwargs: SimpleNamespace(
            _started=True,
            start=lambda: None,
            complete=lambda *a, **k: None,
            activate=lambda *a, **k: None,
            set_detail=lambda title, text, **k: detail_updates.append((title, text)),
            clear_detail=lambda *a, **k: None,
            set_detail_renderable=lambda *a, **k: None,
            stop=lambda: workflow_state.update(stopped=True),
        ),
    )

    class _FakeOptunaSearch:
        def __init__(self, metric, mode):
            self.metric = metric
            self.mode = mode

    class _FakeRunConfig:
        def __init__(self, storage_path, name, callbacks=None, verbose=None):
            self.storage_path = storage_path
            self.name = name
            self.callbacks = callbacks
            self.verbose = verbose

    class _FakeTuneConfig:
        def __init__(self, num_samples, search_alg, trial_dirname_creator):
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

    monkeypatch.setitem(sys.modules, "boxmot.utils.checks", SimpleNamespace(RequirementsChecker=_FakeRequirementsChecker))
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.tune", SimpleNamespace(RunConfig=_FakeRunConfig))
    monkeypatch.setitem(sys.modules, "ray.tune.search.optuna", SimpleNamespace(OptunaSearch=_FakeOptunaSearch))

    args = SimpleNamespace(
        detector=[tmp_path / "yolov8n.pt"],
        reid=[tmp_path / "osnet_x0_25_msmt17.pt"],
        tracker="strongsort",
        data="mot17-ablation",
        maximize=("HOTA",),
        minimize=(),
        objectives=("HOTA",),
        n_threads=1,
        n_trials=3,
        project=Path("runs"),
        verbose=False,
    )

    tuner_module.main(args)

    assert captured["extra"] == "evolve"
    assert "restore_path" not in captured
    assert Path(captured["storage_path"]).is_absolute()
    assert Path(captured["storage_path"]) == (tmp_path / "runs" / "ray").resolve()
    assert captured["run_name"] == "strongsort_tune_2"
    assert captured["verbose"] == 0
    assert len(captured["callbacks"]) == 1
    assert captured["ray_init_kwargs"]["include_dashboard"] is False
    assert captured["ray_init_kwargs"]["logging_level"] == logging.ERROR
    assert captured["ray_init_kwargs"]["log_to_driver"] is False
    assert any(title == tuner_module.TUNE_OPTIMIZE_STEP for title, _ in detail_updates)
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
        lambda args, workflow=None: setattr(args, "project", (tmp_path / "runs").resolve()),
    )
    monkeypatch.setattr(
        tuner_module,
        "_log_tune_pipeline_intro",
        lambda *args, **kwargs: SimpleNamespace(
            _started=True,
            _lock=threading.RLock(),
            start=lambda: None,
            complete=lambda *a, **k: None,
            activate=lambda *a, **k: None,
            set_detail=lambda *a, **k: None,
            clear_detail=lambda *a, **k: None,
            set_detail_renderable=lambda *a, **k: None,
            stop=lambda: None,
        ),
    )

    class _FakeOptunaSearch:
        def __init__(self, metric, mode):
            self.metric = metric
            self.mode = mode

    class _FakeRunConfig:
        def __init__(self, storage_path, name, callbacks=None, verbose=None):
            self.storage_path = storage_path
            self.name = name
            self.callbacks = callbacks
            self.verbose = verbose

    class _FakeTuneConfig:
        def __init__(self, num_samples, search_alg, trial_dirname_creator):
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

    monkeypatch.setitem(sys.modules, "boxmot.utils.checks", SimpleNamespace(RequirementsChecker=_FakeRequirementsChecker))
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.tune", SimpleNamespace(RunConfig=_FakeRunConfig))
    monkeypatch.setitem(sys.modules, "ray.tune.search.optuna", SimpleNamespace(OptunaSearch=_FakeOptunaSearch))

    args = SimpleNamespace(
        detector=[tmp_path / "yolov8n.pt"],
        reid=[tmp_path / "osnet_x0_25_msmt17.pt"],
        tracker="strongsort",
        data="mot17-ablation",
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
    callback = tuner_module._TuneWorkflowCallback(total=1, maximize=["HOTA"], minimize=[])

    tuner_module._set_tune_progress_workflow(workflow)
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
        tuner_module._set_tune_progress_workflow(None)

    assert any(title == tuner_module.TUNE_OPTIMIZE_STEP for title, _ in updates)


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
        lambda args, workflow=None: setattr(args, "project", (tmp_path / "runs").resolve()),
    )
    monkeypatch.setattr(
        tuner_module,
        "_log_tune_pipeline_intro",
        lambda *args, **kwargs: SimpleNamespace(
            _started=True,
            start=lambda: None,
            complete=lambda *a, **k: None,
            activate=lambda *a, **k: None,
            set_detail=lambda *a, **k: None,
            clear_detail=lambda *a, **k: None,
            set_detail_renderable=lambda *a, **k: None,
            stop=lambda: None,
        ),
    )

    class _FakeOptunaSearch:
        def __init__(self, metric, mode):
            self.metric = metric
            self.mode = mode

    class _FakeRunConfig:
        def __init__(self, storage_path, name, callbacks=None, verbose=None):
            self.storage_path = storage_path
            self.name = name
            self.callbacks = callbacks
            self.verbose = verbose

    class _FakeTuneConfig:
        def __init__(self, num_samples, search_alg, trial_dirname_creator):
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

    monkeypatch.setitem(sys.modules, "boxmot.utils.checks", SimpleNamespace(RequirementsChecker=_FakeRequirementsChecker))
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.tune", SimpleNamespace(RunConfig=_FakeRunConfig))
    monkeypatch.setitem(sys.modules, "ray.tune.search.optuna", SimpleNamespace(OptunaSearch=_FakeOptunaSearch))

    args = SimpleNamespace(
        detector=[tmp_path / "yolov8n.pt"],
        reid=[tmp_path / "osnet_x0_25_msmt17.pt"],
        tracker="strongsort",
        data="mot17-ablation",
        maximize=("HOTA",),
        minimize=(),
        objectives=("HOTA",),
        n_threads=1,
        n_trials=3,
        project=Path("runs"),
        verbose=False,
        resume_tune=True,
    )

    tuner_module.main(args)

    assert captured["extra"] == "evolve"
    assert Path(captured["restore_path"]).is_absolute()
    assert Path(captured["restore_path"]) == (tmp_path / "runs" / "ray" / "strongsort_tune").resolve()
    assert captured["run_name"] == "strongsort_tune"


def test_build_tune_workflow_fields_uses_benchmark_data() -> None:
    args = SimpleNamespace(
        tracker="bytetrack",
        detector=[Path("yolov8n.pt")],
        reid=[Path("osnet_x0_25_msmt17.pt")],
        data="mot17-ablation",
        benchmark="",
        n_trials=3,
    )

    fields = dict(tuner_module._build_tune_workflow_fields(args, maximize=["HOTA"], minimize=[]))

    assert fields["Dataset"] == "mot17-ablation"


def test_build_tune_artifacts_renderable_lists_paths() -> None:
    saved_artifacts = {
        "csv_path": Path("/tmp/results.csv"),
        "best_trial_id": "fe35bbe6",
        "best_yaml_path": Path("/tmp/best_bytetrack.yaml"),
        "summary_path": Path("/tmp/summary.md"),
    }

    rendered = ui_module.capture_renderable(
        tuner_module._build_tune_artifacts_renderable(saved_artifacts),
        width=140,
    )

    assert "Saved Artifacts" in rendered
    assert "Results CSV" in rendered
    assert str(saved_artifacts["csv_path"]) in rendered
    assert "Best config (fe35bbe6)" in rendered
    assert str(saved_artifacts["best_yaml_path"]) in rendered
    assert "Summary" in rendered
    assert str(saved_artifacts["summary_path"]) in rendered
