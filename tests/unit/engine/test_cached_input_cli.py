"""The optional replay cache belongs to eval/tune and preserves build selection."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from boxmot.engine.cli import boxmot


@pytest.mark.parametrize("command", ("eval", "tune"))
@pytest.mark.parametrize("flags,enabled", (((), False), (("--cache-inputs",), True), (("--no-cache-inputs",), False)))
def test_cached_input_flag_preserves_explicit_build(monkeypatch, command, flags, enabled):
    """The accelerator never requests perception or changes the selected build."""
    captured = {}
    module = "boxmot.engine.eval.evaluator" if command == "eval" else "boxmot.engine.tuning.tuner"
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(main=lambda args: captured.setdefault("args", args)))
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=lambda _: pytest.fail("Explicit builds must bypass perception")),
    )
    result = CliRunner().invoke(
        boxmot,
        [command, "--dataset", "mot17", "--build", "selected-build", *flags],
    )
    assert result.exit_code == 0, result.output
    assert captured["args"].cache_inputs is enabled
    assert captured["args"].build == "selected-build"


@pytest.mark.parametrize("command", ("track", "materialize", "research"))
def test_cached_input_flag_is_not_exposed_on_unrelated_commands(command):
    """Only workflows that consume the derived replay cache advertise it."""
    result = CliRunner().invoke(boxmot, [command, "--help"])
    assert result.exit_code == 0, result.output
    assert "--cache-inputs" not in result.output


@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("persistent", (False, True))
def test_evaluator_routes_cache_and_session_to_replay(monkeypatch, tmp_path, enabled, persistent):
    """Cache selection reaches replay, and metrics share the session's safe executor."""
    from boxmot.engine.eval import evaluator
    from boxmot.trackers import TrackerSpec

    captured = {}
    metric_scope = []

    class Session:
        @contextmanager
        def metric_execution(self):
            metric_scope.append(True)
            try:
                yield
            finally:
                metric_scope.pop()

    session = Session() if persistent else None

    def replay(build, spec, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_dir=tmp_path / "output", frames=2)

    def metrics(args, *, verbose):
        assert bool(metric_scope) is persistent
        return {"pedestrian": {"HOTA": 70.0}}

    monkeypatch.setattr(evaluator, "_ensure_setup", lambda _: None)
    monkeypatch.setattr(evaluator, "_tracker_spec", lambda *args: TrackerSpec(name="bytetrack"))
    monkeypatch.setattr(evaluator, "replay_build", replay)
    monkeypatch.setattr(evaluator, "run_motmetrics", metrics)
    args = SimpleNamespace(
        build_path=tmp_path / "build",
        split="validation",
        sequence_names=("sequence",),
        seq_info={"sequence": 2},
        sequence_workers=1,
        experiment_id="fixture",
        cache_inputs=enabled,
    )
    result = evaluator.run_eval(args, setup=False, output_dir=tmp_path / "output", replay_session=session)
    assert captured.get("cache_inputs", False) is enabled
    assert captured.get("session") is session
    assert not metric_scope
    assert result.timings["frames"] == 2
