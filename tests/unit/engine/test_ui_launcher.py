"""Launcher interactions must produce valid argv without executing workflows."""

from __future__ import annotations

import io
import json
import shlex
import subprocess
import sys

import click
import pytest
from rich.console import Console

from boxmot.engine.cli import boxmot
from boxmot.engine.ui.launcher.app import Editor, LauncherApp
from boxmot.engine.ui.launcher.model import Choice, LauncherModel, field_choices


def test_defaults_are_displayed_without_becoming_explicit_arguments() -> None:
    """Preserve CLI parameter-source logic, including webcam display defaults."""
    model = LauncherModel(boxmot)
    assert model.value("source") == "0"
    assert model.value("detector")
    assert model.argv() == ["track"]
    model.set_value("source", "videos/a clip [blue].mp4")
    model.set_value("reid", "lmbn_n_duke")
    assert model.argv() == ["track", "--source", "videos/a clip [blue].mp4", "--reid", "lmbn_n_duke"]
    assert shlex.split(model.preview()) == ["boxmot", "--dashboard", *model.argv()]
    model.set_value("reid", None)
    assert "--reid" not in model.argv()


@pytest.mark.parametrize("detector", ["yolov8n", "yolov8n-obb", "models/custom.pt"])
def test_tracking_geometry_follows_detector_without_an_editable_field(detector: str) -> None:
    """Model selection and image resolution cannot add a conflicting geometry override."""
    model = LauncherModel(boxmot)
    model.set_value("detector", detector)
    model.set_value("imgsz", "1024")
    assert model.default("geometry") == "auto"
    assert "geometry" not in {field.name for field in model.fields}
    assert "--geometry" not in model.validate()
    app = LauncherApp(model)
    app.handle_key("3")
    console = Console(file=io.StringIO(), width=144, height=24, color_system=None)
    console.print(app.render(console))
    assert "Automatic from detector" in console.file.getvalue()
    assert "Geometry  --geometry" not in console.file.getvalue()


def test_experiment_selection_clears_conflicting_component_arguments() -> None:
    model = LauncherModel(boxmot, initial_command="eval")
    model.set_value("dataset", "mot17")
    model.set_value("detector", "yolox-x-mot17/ablation")
    model.set_value("reid", "lmbn-n-duke")
    model.set_value("experiment", "mot17/ablation-yolox-lmbn.yaml")
    assert model.argv() == ["eval", "--experiment", "mot17/ablation-yolox-lmbn.yaml"]
    assert not {"dataset", "detector", "reid"}.intersection(field.name for field in model.fields)
    resolved = model.resolved_experiment()
    assert resolved is not None
    assert resolved["dataset"]["split"] == "ablation"
    assert resolved["detector"]["checkpoint"] == "ablation"
    assert resolved["reid"]["id"] == "lmbn-n-duke"
    assert model.validate() == model.argv()
    model.set_value("experiment", None)
    assert {"dataset", "detector", "reid"}.issubset(field.name for field in model.fields)


def test_explicit_build_clears_perception_device() -> None:
    model = LauncherModel(boxmot, initial_command="eval")
    model.set_value("device", "0")
    model.set_value("build_ref", "my-build")
    assert "--device" not in model.argv()
    assert "device" not in {field.name for field in model.fields}
    model.set_value("build_ref", None)
    assert "device" in {field.name for field in model.fields}


def test_unset_values_are_absence_not_truthy_sentinel() -> None:
    model = LauncherModel(boxmot, initial_command="eval")
    assert model.value("build_ref") is None
    assert not model.value("sequence_names")
    console = Console(file=io.StringIO(), width=80, height=24)
    console.print(LauncherApp(model).render(console))
    assert "UNSET" not in console.file.getvalue()


def test_boolean_toggle_and_reset_preserve_flag_semantics() -> None:
    model = LauncherModel(boxmot)
    app = LauncherApp(model)
    app.cursor = next(index + 1 for index, field in enumerate(model.fields) if field.name == "show")
    assert model.field_display("show") == "auto (live input)"
    app.handle_key("enter")
    assert model.argv() == ["track", "--show"]
    assert model.field_display("show") == "on"
    app.handle_key("enter")
    assert model.argv() == ["track"]
    model.select_command("materialize")
    app.cursor = next(index + 1 for index, field in enumerate(model.fields) if field.name == "resume")
    app.handle_key("enter")
    assert "--no-resume" in model.argv()
    app.handle_key("d")
    assert "--no-resume" not in model.argv()
    assert model.value("resume") is True


def test_required_inputs_and_real_option_callbacks_validate_before_review() -> None:
    model = LauncherModel(boxmot, initial_command="tune")
    app = LauncherApp(model)
    app.handle_key("r")
    assert "Choose an experiment" in app.error
    assert not app.review
    model.set_value("experiment", "mot17/ablation-yolox-lmbn.yaml")
    model.set_value("n_trials", "0")
    app.handle_key("r")
    assert "n-trials" in app.error
    assert not app.review
    assert app.stage == "Search"
    assert model.fields[app.cursor - 1].name == "n_trials"
    model.set_value("n_trials", "2")
    app.handle_key("r")
    assert app.review


def test_parse_and_review_do_not_invoke_command_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    model = LauncherModel(boxmot)

    def forbidden_callback(*args, **kwargs) -> None:
        pytest.fail("Launcher invoked a command before returning reviewed argv")

    monkeypatch.setattr(model.command, "callback", forbidden_callback)
    app = LauncherApp(model)
    app.handle_key("r")
    assert app.review and not app.finished and app.result is None
    app.handle_key("escape")
    assert not app.review and not app.finished
    app.handle_key("r")
    app.handle_key("enter")
    assert app.finished
    assert app.result == ["track"]


def test_cancel_and_closed_input_never_return_run_arguments() -> None:
    for key in ("q", "escape", "eof"):
        app = LauncherApp(LauncherModel(boxmot))
        app.handle_key(key)
        assert app.finished and app.result is None


def test_multiple_options_and_negative_flags_preserve_argv_boundaries() -> None:
    model = LauncherModel(boxmot, initial_command="materialize")
    model.set_value("experiment", "mot17/ablation-yolox-lmbn.yaml")
    model.set_value("resume", False)
    model.set_value("plan_overrides", 'detect.batch_size=8 "embed.some_path=path with spaces"')
    argv = model.argv()
    assert "--no-resume" in argv
    assert argv[-4:] == ["--set", "detect.batch_size=8", "--set", "embed.some_path=path with spaces"]


def test_picker_search_custom_paths_and_multiple_selection() -> None:
    editor = Editor("detector", "Detector", "", [Choice("yolov8n"), Choice("yolo11n")])
    for character in "yolo11":
        editor.handle_key(character)
    assert editor.handle_key("enter") == (True, "yolo11n")
    editor.handle_key("ctrl+u")
    for character in "models/my detector.pt":
        editor.handle_key(character)
    assert editor.handle_key("enter") == (True, "models/my detector.pt")
    editor = Editor("include", "Formats", "", [Choice("onnx"), Choice("engine")], multiple=True)
    editor.handle_key("down")
    editor.handle_key(" ")
    editor.handle_key("down")
    editor.handle_key(" ")
    assert editor.handle_key("enter") == (True, "onnx engine")


def test_picker_enter_replaces_default_multiple_selection() -> None:
    editor = Editor("include", "Formats", "", [Choice("onnx"), Choice("engine")], multiple=True)
    editor.selected = ["onnx"]
    editor.handle_key("end")
    assert editor.handle_key("enter") == (True, "engine")


def test_malformed_multiple_value_stays_in_editor_until_corrected() -> None:
    model = LauncherModel(boxmot, initial_command="materialize")
    app = LauncherApp(model)
    app.cursor = next(index + 1 for index, field in enumerate(model.fields) if field.name == "plan_overrides")
    app.handle_key("enter")
    for character in '"embed.path=directory with spaces':
        app.handle_key(character)
    app.handle_key("enter")
    assert "No closing quotation" in app.error
    assert app.editor is not None
    assert "plan_overrides" not in model.values
    app.handle_key('"')
    app.handle_key("enter")
    assert app.editor is None
    assert model.argv()[-2:] == ["--set", "embed.path=directory with spaces"]


@pytest.mark.parametrize("size", [(60, 20), (80, 24), (120, 40)])
def test_form_and_review_fit_terminal_with_literal_user_text(size: tuple[int, int]) -> None:
    width, height = size
    model = LauncherModel(boxmot)
    model.set_value("source", "clips/[red]literal[/red] " + "a-long-path/" * 250 + "final.mp4")
    app = LauncherApp(model)
    console = Console(file=io.StringIO(), width=width, height=height, color_system=None)
    console.print(app.render(console), end="")
    assert len(console.file.getvalue().splitlines()) <= height
    app.handle_key("r")
    app.handle_key("end")
    console.file.seek(0)
    console.file.truncate()
    console.print(app.render(console), end="")
    assert len(console.file.getvalue().splitlines()) <= height
    assert app.review_offset > 0
    app.handle_key("home")
    console.file.seek(0)
    console.file.truncate()
    console.print(app.render(console), end="")
    assert "[red]literal[/red]" in console.file.getvalue()


def test_catalogs_use_profiles_and_canonical_export_names() -> None:
    model = LauncherModel(boxmot)
    assert "yolov8n" in {choice.value for choice in field_choices("track", model.options["detector"])}
    assert "lmbn_n_duke" in {choice.value for choice in field_choices("track", model.options["reid"])}
    model.select_command("eval")
    assert "yolox-x-mot17/ablation" in {choice.value for choice in field_choices("eval", model.options["detector"])}
    model.select_command("export")
    formats = {choice.value for choice in field_choices("export", model.options["include"])}
    assert "engine" in formats and "tensorrt" not in formats and "-" not in formats


def test_command_drafts_preserve_edits_and_experiment_cache_checks_mode() -> None:
    model = LauncherModel(boxmot, initial_command="eval", experiment="kitti-mots/2d.yaml")
    assert model.resolved_experiment() is not None
    model.select_command("materialize")
    model.set_value("experiment", "kitti-mots/2d.yaml")
    with pytest.raises(ValueError, match="supports only eval"):
        model.resolved_experiment()
    model.select_command("eval")
    assert model.values["experiment"] == "kitti-mots/2d.yaml"


def test_catalog_and_command_selection_do_not_import_ml_runtimes() -> None:
    script = """
import json, sys
from boxmot.engine.cli import boxmot
from boxmot.engine.ui.launcher.model import COMMANDS, LauncherModel, field_choices
model = LauncherModel(boxmot)
for name in COMMANDS:
    model.select_command(name)
    for field in model.fields:
        field_choices(name, field.option)
print(json.dumps(sorted(set(sys.modules) & {'torch', 'cv2', 'numpy', 'ultralytics', 'pyarrow'})))
"""
    completed = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == []


def test_reid_evaluation_checks_required_checkpoint_paths(tmp_path) -> None:
    model = LauncherModel(boxmot, initial_command="eval-reid")
    model.set_value("weights", str(tmp_path / "missing.pt"))
    model.set_value("dataset", "market1501")
    model.set_value("data_dir", str(tmp_path))
    with pytest.raises(click.BadParameter, match="does not exist"):
        model.validate()
    app = LauncherApp(model)
    app.handle_key("r")
    assert not app.review
    assert "does not exist" in app.error
    assert model.fields[app.cursor - 1].name == "weights"
    assert app.stage == "Model"
    app.handle_key("enter")
    assert app.editor is not None
    checkpoint = tmp_path / "valid weights.pt"
    checkpoint.touch()
    app.handle_key("ctrl+u")
    for character in str(checkpoint):
        app.handle_key(character)
    app.handle_key("enter")
    assert model.values["weights"] == str(checkpoint)
    app.handle_key("r")
    assert app.review and not app.error


def test_stage_navigation_keeps_the_workflow_and_review_reachable() -> None:
    """Stage shortcuts and field navigation share one continuous form."""
    model = LauncherModel(boxmot)
    app = LauncherApp(model)
    assert model.stages == ["Setup", "Perception", "Tracking", "Output"]
    assert app.stage == "Setup"
    app.handle_key("shift+tab")
    assert app.cursor == 0
    app.handle_key("tab")
    assert app.stage == "Perception"
    assert model.fields[app.cursor - 1].name == "detector"
    app.cursor = next(index + 1 for index, field in enumerate(model.fields) if field.name == "classes")
    app.handle_key("down")
    assert app.stage == "Tracking"
    assert model.fields[app.cursor - 1].name == "tracker"
    app.handle_key("up")
    assert app.stage == "Perception"
    assert model.fields[app.cursor - 1].name == "classes"
    app.handle_key("4")
    assert app.stage == "Output"
    app.handle_key("tab")
    assert app.stage == "Output"
    app.handle_key("shift+tab")
    assert app.stage == "Tracking"
    app.handle_key("home")
    assert app.cursor == 0 and app.stage == "Setup"
    app.handle_key("end")
    assert app.cursor == len(model.fields) + 1 and app.stage == "Output"
    app.handle_key("enter")
    assert app.review and not app.finished


def test_picker_input_does_not_trigger_stage_shortcuts() -> None:
    """Numbers remain searchable model text while a picker owns input."""
    model = LauncherModel(boxmot)
    app = LauncherApp(model)
    app.handle_key("tab")
    app.handle_key("right")
    assert app.editor is not None and app.editor.name == "detector"
    cursor = app.cursor
    for key in ("1", "1", "tab"):
        app.handle_key(key)
    assert app.editor is not None and app.editor.query == "11"
    assert app.cursor == cursor and app.stage == "Perception"
    app.handle_key("left")
    assert app.editor is None
    assert "detector" not in model.values
    assert app.stage == "Perception"


def test_tracker_change_and_reset_preserve_focus_when_stages_disappear() -> None:
    """Changing tracking families must not leave focus on an unrelated row."""
    model = LauncherModel(boxmot)
    app = LauncherApp(model)
    app.handle_key("3")
    app.handle_key("right")
    for character in "edgetam":
        app.handle_key(character)
    app.handle_key("enter")
    assert model.value("tracker") == "edgetam"
    assert "Perception" not in model.stages
    assert app.stage == "Tracking"
    assert model.fields[app.cursor - 1].name == "tracker"
    assert "prompt_weights" in {field.name for field in model.fields}
    app.handle_key("d")
    assert "Perception" in model.stages
    assert app.stage == "Tracking"
    assert model.fields[app.cursor - 1].name == "tracker"
    assert "prompt_weights" not in {field.name for field in model.fields}


def test_experiment_selection_skips_hidden_perception_stage_until_reset() -> None:
    """Authored experiments supply perception without leaving an empty step."""
    model = LauncherModel(boxmot, initial_command="eval")
    app = LauncherApp(model)
    app.handle_key("down")
    app.handle_key("right")
    for character in "mot17/ablation-yolox-lmbn.yaml":
        app.handle_key(character)
    app.handle_key("enter")
    assert model.values["experiment"] == "mot17/ablation-yolox-lmbn.yaml"
    assert app.stage == "Setup"
    assert model.fields[app.cursor - 1].name == "experiment"
    assert "Perception" not in model.stages
    app.handle_key("tab")
    assert app.stage == "Tracking"
    app.handle_key("home")
    app.handle_key("down")
    app.handle_key("d")
    assert model.fields[app.cursor - 1].name == "experiment"
    app.handle_key("tab")
    assert app.stage == "Perception"


@pytest.mark.parametrize("size", [(40, 20), (60, 20), (80, 24), (120, 40)])
def test_stage_form_shows_only_its_fields_and_keeps_picker_visible(size: tuple[int, int]) -> None:
    """Resizing preserves the active stage and the searchable picker state."""
    model = LauncherModel(boxmot)
    app = LauncherApp(model)
    width, height = size
    console = Console(file=io.StringIO(), width=width, height=height, color_system=None)
    console.print(app.render(console), end="")
    setup = console.file.getvalue()
    assert "Configure" in setup and "Source" in setup
    assert "r review" in setup and "q quit" in setup
    assert "Detector" not in setup and "ReID" not in setup
    app.handle_key("tab")
    console.file.seek(0)
    console.file.truncate()
    console.print(app.render(console), end="")
    perception = console.file.getvalue()
    assert "Detector" in perception and "ReID" in perception
    assert "Imgsz" not in perception and "Source" not in perception
    app.handle_key("right")
    for character in "yolo11":
        app.handle_key(character)
    cursor = app.cursor
    for resized_width, resized_height in (size, (60, 20), (120, 40)):
        resized = Console(file=io.StringIO(), width=resized_width, height=resized_height, color_system=None)
        resized.print(app.render(resized), end="")
        rendered = resized.file.getvalue()
        assert len(rendered.splitlines()) <= resized_height
        assert "Perception" in rendered
        assert "Search / value" in rendered and "yolo11" in rendered
        assert app.cursor == cursor and app.stage == "Perception"
        assert app.editor is not None and app.editor.query == "yolo11"
