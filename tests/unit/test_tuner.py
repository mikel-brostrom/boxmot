from pathlib import Path
from types import SimpleNamespace

import boxmot.engine.tuner as tuner_module


def test_tuner_uses_absolute_ray_paths_after_eval_setup(monkeypatch, tmp_path):
    captured = {}

    class _FakeRequirementsChecker:
        def sync_extra(self, extra):
            captured["extra"] = extra

    monkeypatch.setattr(tuner_module, "TrialSaveCallback", lambda yaml_cfg, tracking_method: object())
    monkeypatch.setattr(tuner_module, "load_yaml_config", lambda tracking_method: {})
    monkeypatch.setattr(tuner_module, "_save_all_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuner_module, "run_generate_dets_embs", lambda args: captured.setdefault("generated", True))
    monkeypatch.setattr(
        tuner_module,
        "eval_setup",
        lambda args: setattr(args, "project", (tmp_path / "runs").resolve()),
    )

    class _FakeOptunaSearch:
        def __init__(self, metric, mode):
            self.metric = metric
            self.mode = mode

    class _FakeRunConfig:
        def __init__(self, storage_path, name, callbacks):
            self.storage_path = storage_path
            self.name = name
            self.callbacks = callbacks

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

        def fit(self):
            captured["fit"] = True

        def get_results(self):
            return []

    fake_tune = SimpleNamespace(
        Tuner=_FakeTuner,
        TuneConfig=_FakeTuneConfig,
        with_resources=lambda fn, resources: fn,
    )

    import sys

    monkeypatch.setitem(sys.modules, "boxmot.utils.checks", SimpleNamespace(RequirementsChecker=_FakeRequirementsChecker))
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(tune=fake_tune))
    monkeypatch.setitem(sys.modules, "ray.tune", SimpleNamespace(RunConfig=_FakeRunConfig))
    monkeypatch.setitem(sys.modules, "ray.tune.search.optuna", SimpleNamespace(OptunaSearch=_FakeOptunaSearch))

    args = SimpleNamespace(
        detector=[tmp_path / "yolov8n.pt"],
        reid=[tmp_path / "osnet_x0_25_msmt17.pt"],
        tracker="strongsort",
        maximize=("HOTA",),
        minimize=(),
        objectives=("HOTA",),
        n_threads=1,
        n_trials=3,
        project=Path("runs"),
    )

    tuner_module.main(args)

    assert captured["extra"] == "evolve"
    assert Path(captured["restore_path"]).is_absolute()
    assert Path(captured["storage_path"]).is_absolute()
    assert Path(captured["storage_path"]) == (tmp_path / "runs" / "ray").resolve()
    assert captured["fit"] is True
