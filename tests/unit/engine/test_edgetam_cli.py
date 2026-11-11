"""The launcher and CLI route manual prompts without constructing perception models."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.config.trackers import validate_image_tracker
from boxmot.engine.ui.launcher.model import LauncherModel, field_choices


def test_launcher_selecting_edgetam_replaces_detection_controls() -> None:
    model = LauncherModel(boxmot)
    for name, value in {"detector": "yolo26n", "geometry": "obb", "tracker_backend": "cpp", "half": True}.items():
        model.set_value(name, value)
    model.set_value("tracker", "edgetam")
    model.set_value("prompt_weights", "models/my edgetam.pt")
    names = {field.name for field in model.fields}
    assert {"source", "tracker", "prompt_weights", "device", "save"} <= names
    assert not {"detector", "reid", "segmentor", "geometry", "tracker_backend", "half", "show"} & names
    assert model.validate() == ["track", "--tracker", "edgetam", "--prompt-weights", "models/my edgetam.pt"]
    model.set_value("tracker", "bytetrack")
    assert "prompt_weights" not in model.values
    assert "detector" in {field.name for field in model.fields}


@pytest.mark.parametrize("mode", ["eval", "tune"])
def test_launcher_keeps_prompted_tracker_out_of_detector_benchmarks(mode: str) -> None:
    model = LauncherModel(boxmot, initial_command=mode)
    assert "edgetam" not in {choice.value for choice in field_choices(mode, model.options["tracker"])}
    with pytest.raises(ValueError, match="prompt"):
        validate_image_tracker("edgetam")
    validate_image_tracker("edgetam", allow_prompted=True)


def test_edgetam_cli_forces_interactive_window_and_omits_perception(monkeypatch) -> None:
    module = importlib.import_module("boxmot.engine.commands.track")
    captured = {}

    def dispatch(ctx, mode, target, payload):
        captured.update(payload)

    monkeypatch.setattr(module, "_dispatch_cli_workflow", dispatch)
    result = CliRunner().invoke(
        boxmot, ["track", "--tracker", "edgetam", "--source", "clip.mp4", "--prompt-weights", "weights.pt", "--save"]
    )
    assert result.exit_code == 0, result.output
    assert captured["prompt_weights"] == Path("weights.pt")
    assert captured["show"] is True and captured["save"] is True
    assert captured["detector"] is captured["reid"] is captured["segmentor"] is None


@pytest.mark.parametrize(
    "options",
    [
        ["--detector", "yolo26n"],
        ["--segmentor", "sam"],
        ["--reid", "osnet"],
        ["--geometry", "obb"],
        ["--tracker-backend", "cpp"],
        ["--per-class"],
        ["--conf", "0.4"],
        ["--half"],
        ["--variable-dt"],
        ["--edgetam"],
        ["--no-edgetam"],
        ["--mask-guidance-weights", "edgetam.pt"],
        ["--mask-guidance-max-objects", "96"],
    ],
)
def test_edgetam_cli_rejects_incompatible_options_before_workflow(options: list[str]) -> None:
    result = CliRunner().invoke(boxmot, ["track", "--tracker", "edgetam", *options])
    assert result.exit_code == 2, result.output
    assert "EdgeTAM" in result.output


def test_prompt_weights_requires_prompted_tracker() -> None:
    result = CliRunner().invoke(boxmot, ["track", "--prompt-weights", "edgetam.pt"])
    assert result.exit_code == 2
    assert "requires --tracker edgetam" in result.output


def test_run_track_routes_prompted_mode_before_perception(monkeypatch) -> None:
    from boxmot.engine.tracking import prompted, workflow

    expected = object()
    captured = {}

    def run(args, **kwargs):
        captured["args"] = args
        return expected

    def unexpected(*args, **kwargs):
        pytest.fail("Prompted workflow must not build a detector")

    monkeypatch.setattr(prompted, "run_prompted_track", run)
    monkeypatch.setattr(workflow, "create_detector", unexpected)
    assert workflow.run_track(tracker="edgetam", source="clip.mp4") is expected
    assert captured["args"].tracker == "edgetam"


@pytest.mark.parametrize("component", ["detector", "segmentor", "reid"])
def test_run_track_rejects_explicit_perception_selectors_with_prompts(component: str) -> None:
    from boxmot.engine.tracking.workflow import run_track

    with pytest.raises(ValueError, match="does not use detector"):
        run_track(tracker="edgetam", **{component: "some-model"})
