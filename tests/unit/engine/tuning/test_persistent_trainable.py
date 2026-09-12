"""Ray actor reuse keeps inputs alive while trial namespaces remain isolated."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from boxmot.engine.eval.session import ReplaySession
from boxmot.engine.tuning import tuner
from boxmot.engine.tuning.trainable import build_tracker_trainable


def _result(score: float) -> SimpleNamespace:
    return SimpleNamespace(
        raw={"HOTA": score}, benchmark="fixture", summary_label="all", summary={"HOTA": score}, timings={}, exp_dir=None
    )


@pytest.mark.parametrize("cache_inputs", [False, True])
def test_actor_reset_reuses_session_and_starts_from_clean_options(monkeypatch, cache_inputs) -> None:
    options = SimpleNamespace(sequence_workers=2, cache_inputs=cache_inputs, nested={"value": 1})
    received = []

    def evaluate(args, *, replay_session, evolve_config, **kwargs):
        assert args.nested == {"value": 1}
        args.nested["value"] = 99
        received.append((replay_session, dict(evolve_config)))
        return _result(evolve_config["threshold"])

    monkeypatch.setattr(tuner, "run_eval", evaluate)
    monkeypatch.setattr(tuner, "aggregate_results", dict)
    trainable = build_tracker_trainable(SimpleNamespace(Trainable=object), options)
    actor = trainable()
    actor.setup({"threshold": 0.3})
    actor.reset_config({"threshold": 0.3})
    first = actor.step()
    assert first["HOTA"] == 0.3 and first["done"] is True
    with pytest.raises(RuntimeError, match="already completed"):
        actor.step()
    actor.reset_config({"threshold": 0.7})
    second = actor.step()
    assert second["HOTA"] == 0.7
    assert received[0][0] is received[1][0]
    assert isinstance(received[0][0], ReplaySession)
    assert received[0][0].cache_inputs is cache_inputs
    assert received[0][0].workers == 2
    assert options.nested == {"value": 1}
    actor.cleanup()
    actor.cleanup()
    assert received[0][0]._closed


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_objective_discards_session_on_failed_trial(monkeypatch, error) -> None:
    sessions = []

    def evaluate(_args, *, replay_session, **kwargs):
        sessions.append(replay_session)
        if len(sessions) == 1:
            raise error("evaluation failed")
        return _result(50.0)

    monkeypatch.setattr(tuner, "run_eval", evaluate)
    monkeypatch.setattr(tuner, "aggregate_results", dict)
    objective = tuner.TrackerObjective(SimpleNamespace(sequence_workers=1))
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            objective({})
    else:
        assert objective({})["HOTA"] == 0.0
    assert sessions[0]._closed
    assert objective({})["HOTA"] == 50.0
    assert sessions[0] is not sessions[1]
    objective.close()


def test_actor_lifecycle_is_serializable_before_resources_start() -> None:
    from ray import cloudpickle, tune

    trainable = build_tracker_trainable(tune, SimpleNamespace(sequence_workers=1, cache_inputs=False))
    restored = cloudpickle.loads(cloudpickle.dumps(trainable))
    assert issubclass(restored, tune.Trainable)
    assert restored.workflow_options.sequence_workers == 1
    assert not hasattr(restored, "_objective")
