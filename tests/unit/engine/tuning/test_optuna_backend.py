"""Check actual Ray/Optuna dimensions separately from returned runtime constants."""

from __future__ import annotations

import pickle

import pytest

from boxmot.engine.tuning.backends.optuna_backend import OptunaBackend, yaml_to_optuna_define_space
from boxmot.engine.tuning.conditions import SCOPED_PARAMETER_PREFIX
from boxmot.engine.tuning.search_space import default_tune_config, normalize_trial_config

tune = pytest.importorskip("ray.tune")
optuna = pytest.importorskip("optuna")


@pytest.mark.parametrize("mode", ["off", "gate", "blend"])
@pytest.mark.parametrize("enabled", [False, True])
def test_optimizer_samples_only_active_dimensions(mode: str, enabled: bool) -> None:
    """Inactive defaults have no distribution; active defaults remain real samples."""
    schema = {
        "mode": {"type": "choice", "options": ["off", "gate", "blend"], "default": "blend"},
        "weight": {
            "type": "uniform",
            "range": [0.1, 1.0],
            "default": 0.98,
            "depends_on": "mode",
            "cases": {"off": {"default": 0.98}, "gate": {"default": 0.98}},
        },
        "enabled": {
            "type": "choice",
            "options": [False, True],
            "default": False,
            "depends_on": "mode",
            "cases": {"off": {"default": False}},
            "activates": {"child": {"type": "uniform", "range": [0.1, 0.9], "default": 0.5}},
        },
    }
    backend = OptunaBackend(
        yaml_cfg=schema,
        opt_metrics=["HOTA"],
        opt_modes=["max"],
        seed=17,
        baseline_config={"mode": mode, "enabled": enabled, "weight": 0.98, "child": 0.5},
    )
    search, param_space = backend.build(tune)
    assert param_space == {}
    config = normalize_trial_config(search.suggest("baseline"))
    assert config["mode"] == mode
    assert config["weight"] == 0.98
    assert config["enabled"] == (enabled if mode != "off" else False)
    search.on_trial_complete("baseline", result={"HOTA": 70.0})

    # Inspect the optimizer history, not the Ray config containing returned constants.
    (trial,) = search.searcher._ot_study.get_trials()
    assert trial.state == optuna.trial.TrialState.COMPLETE
    assert set(trial.params) == set(trial.distributions)
    sampled = {
        label.removeprefix(SCOPED_PARAMETER_PREFIX).rsplit("__", 1)[0]
        if label.startswith(SCOPED_PARAMETER_PREFIX)
        else label: value
        for label, value in trial.params.items()
    }
    expected = {"mode"}
    if mode == "blend":
        expected.add("weight")
    if mode != "off":
        expected.add("enabled")
        if enabled:
            expected.add("child")
    assert set(sampled) == expected
    assert len(sampled) == len(trial.params)
    assert sampled == {name: config[name] for name in expected}


def test_visited_cases_are_reused_and_checkpointed_without_derived_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suggestions reuse prepared paths; restored checkpoints lazily rebuild them."""
    import boxmot.engine.tuning.backends.optuna_backend as backend

    schema = {
        "mode": {"type": "choice", "options": ["off", "blend"], "default": "off"},
        "weight": {
            "type": "uniform",
            "range": [0.1, 1.0],
            "default": 0.7,
            "depends_on": "mode",
            "cases": {"off": {"default": 0.7}},
        },
        "enabled": {
            "type": "choice",
            "options": [False, True],
            "default": True,
            "depends_on": "mode",
            "cases": {"off": {"default": False}},
            "activates": {"child": {"type": "uniform", "range": [0.1, 0.9], "default": 0.5}},
        },
    }
    space = yaml_to_optuna_define_space(schema)
    assert not space.root.children  # Unvisited combinations remain uncompiled.
    examples = []
    for mode in ("off", "blend"):
        defaults = default_tune_config(schema, defaults={"mode": mode})
        baseline = space.baseline(defaults)
        trial = optuna.trial.FixedTrial(baseline)
        expected = space(trial)
        examples.append((defaults, baseline, expected, trial.params))

    restored = pickle.loads(pickle.dumps(space))
    assert not restored.root.children
    for defaults, baseline, expected, labels in examples:
        assert restored.baseline(defaults) == baseline
        trial = optuna.trial.FixedTrial(baseline)
        assert restored(trial) == expected
        assert trial.params == labels

    def rebuild(*args, **kwargs):
        raise AssertionError("A visited conditional path was recompiled.")

    for name in ("categorical_branch", "flatten_yaml_config", "conditional_yaml_tree", "scoped_parameter_name"):
        monkeypatch.setattr(backend, name, rebuild)

    for sampler in (space, restored):
        for defaults, baseline, expected, labels in examples:
            assert sampler.baseline(defaults) == baseline
            trial = optuna.trial.FixedTrial(baseline)
            assert sampler(trial) == expected
            assert trial.params == labels
