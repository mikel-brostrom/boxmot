"""Launcher form state backed by canonical Click options and config catalogs."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import click

from boxmot.configs import CONFIG_ROOT

COMMANDS = ("track", "materialize", "eval", "tune", "export", "train-reid", "eval-reid")
COMMAND_LABELS = {
    "track": "Track a video or camera",
    "materialize": "Build a perception dataset",
    "eval": "Evaluate a tracker",
    "tune": "Tune tracker parameters",
    "export": "Export a ReID model",
    "train-reid": "Train a ReID model",
    "eval-reid": "Evaluate a ReID model",
}

# This is presentation order only: types, defaults, help, and validation belong
# to the selected command. Unlisted advanced options remain available in its CLI.
_FORM_GROUPS = {
    "track": (
        ("Input", ("source",)),
        ("Models", ("detector", "reid", "tracker", "tracker_backend", "geometry", "segmentor")),
        ("Inference", ("device", "imgsz", "conf", "classes", "half")),
        ("Output", ("show", "save", "save_txt", "project", "name")),
    ),
    "materialize": (
        ("Input", ("experiment", "data_root", "build_root")),
        ("Inference", ("device", "fps", "plan_path", "plan_overrides")),
        ("Output", ("publish_image_refs", "publish_embeddings", "publish_masks", "resume")),
    ),
    "eval": (
        ("Input", ("experiment", "dataset", "detector", "reid", "split", "build_ref", "sequence_names")),
        ("Tracker", ("tracker", "tracker_backend", "tracker_config", "postprocessing")),
        ("Execution", ("device", "sequence_workers", "fps", "data_root", "build_root")),
        ("Output", ("show", "save", "show_timing", "project", "name")),
    ),
    "tune": (
        ("Input", ("experiment", "dataset", "detector", "reid", "split", "build_ref", "sequence_names")),
        ("Tracker", ("tracker", "tracker_backend", "tracker_config")),
        ("Search", ("n_trials", "search_alg", "objectives", "maximize", "minimize", "seed")),
        ("Execution", ("device", "sequence_workers", "max_concurrent_trials", "resume_tune", "project")),
    ),
    "export": (
        ("Model", ("weights", "include")),
        ("Export", ("device", "imgsz", "batch_size", "half", "dynamic", "simplify", "opset")),
    ),
    "train-reid": (
        ("Input", ("model", "recipe", "dataset", "data_dir", "cfg", "resume")),
        ("Training", ("device", "epochs", "batch_size", "imgsz", "lr")),
        ("Output", ("project", "name")),
    ),
    "eval-reid": (
        ("Input", ("weights", "model", "dataset", "data_dir")),
        ("Evaluation", ("device", "batch_size", "imgsz", "flip_tta")),
        ("Output", ("output",)),
    ),
}


@dataclass(frozen=True)
class Choice:
    """One catalog entry with an argv value and optional explanation."""

    value: str
    detail: str = ""


@dataclass(frozen=True)
class FormField:
    """A Click option positioned in a launcher section."""

    group: str
    option: click.Option

    @property
    def name(self) -> str:
        """Return the canonical parameter name."""
        assert self.option.name is not None
        return self.option.name

    @property
    def label(self) -> str:
        """Use the public option name as an unambiguous field label."""
        return self.option.opts[0].removeprefix("--").replace("-", " ").capitalize()


def _profile_choices(directory: str) -> list[Choice]:
    """List installed YAML profile references without loading model code."""
    root = CONFIG_ROOT / directory
    return [
        Choice(path.relative_to(root).with_suffix("").as_posix(), "Profile")
        for path in sorted(root.rglob("*.yaml"))
    ]


def field_choices(command: str, option: click.Option) -> list[Choice]:
    """Return local, dependency-light suggestions for a public option."""
    if isinstance(option.type, click.Choice):
        return [Choice(str(value)) for value in option.type.choices]
    name = option.name
    if name == "experiment":
        root = CONFIG_ROOT / "experiments"
        return [Choice(path.relative_to(root).as_posix()) for path in sorted(root.rglob("*.yaml"))]
    if name == "detector":
        if command == "track":
            from boxmot.detectors._model_names import DetectorName

            return [Choice(value) for value in get_args(DetectorName)]
        from boxmot.utils.config import load_yaml_mapping

        choices = []
        for path in sorted((CONFIG_ROOT / "detectors").glob("*.yaml")):
            profile = load_yaml_mapping(path)
            checkpoints = profile.get("checkpoints", {})
            if len(checkpoints) > 1:
                choices.extend(
                    Choice(f"{profile['id']}/{checkpoint}", "Profile / checkpoint") for checkpoint in checkpoints
                )
            else:
                choices.append(Choice(str(profile["id"]), "Profile"))
        return choices
    if name == "reid":
        profiles = _profile_choices("reid")
        if command == "track":
            from boxmot.reid.core.catalog import TRAINED_URLS

            profiles.extend(Choice(Path(name).stem, "Pretrained weights") for name in sorted(TRAINED_URLS))
        return profiles
    if name == "segmentor":
        return _profile_choices("segmentors")
    if name == "dataset":
        if command in {"train-reid", "eval-reid"}:
            from boxmot.reid.datasets.registry import registered_dataset_names

            return [Choice(value) for value in registered_dataset_names()]
        return _profile_choices("datasets")
    if name == "model":
        from boxmot.reid.backbones import registered_backbone_names

        return [Choice(value) for value in registered_backbone_names()]
    if name == "device":
        return [Choice("cpu", "CPU"), Choice("0", "First CUDA GPU"), Choice("mps", "Apple GPU")]
    if name == "include":
        from boxmot.reid.core.formats import REID_FORMATS

        return [Choice(value.argument, value.name) for value in REID_FORMATS if value.argument != "-"]
    if name == "tracker_config":
        return [Choice(str(path)) for path in sorted((CONFIG_ROOT / "trackers" / "presets").glob("*.yaml"))]
    return []


def display_value(value: Any) -> str:
    """Format a Click default or form value for the editor and preview."""
    if value is None or value == ():
        return ""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (tuple, list)):
        return shlex.join(str(item) for item in value)
    return str(value)


class LauncherModel:
    """Track explicit edits independently from defaults and generated argv."""

    def __init__(
        self,
        root_command: click.Group,
        *,
        initial_command: str | None = None,
        experiment: str | None = None,
    ) -> None:
        self.root_command = root_command
        self._drafts: dict[str, dict[str, str | bool]] = {}
        self._experiment_cache: tuple[str, str, str, dict[str, Any]] | None = None
        self.select_command(initial_command or ("eval" if experiment else "track"))
        if experiment:
            if "experiment" not in self.options:
                raise click.UsageError("Preselect an experiment with materialize, eval, or tune.")
            self.set_value("experiment", experiment)

    def select_command(self, name: str) -> None:
        """Load only the selected command and retain each command's edits."""
        if name not in COMMANDS:
            raise click.BadParameter(f"Choose one of: {', '.join(COMMANDS)}", param_hint="command")
        root_context = click.Context(self.root_command)
        command = self.root_command.get_command(root_context, name)
        if command is None:
            raise click.UsageError(f"Command {name!r} is not available.")
        self.name = name
        self.command = command
        self.context = click.Context(command, info_name=name, parent=root_context)
        self.options = {option.name: option for option in command.params if isinstance(option, click.Option)}
        self.values = self._drafts.setdefault(name, {})

    @property
    def fields(self) -> list[FormField]:
        """Return fields appropriate to the selected command and input mode."""
        hidden = {"dataset", "detector", "reid"} if self.values.get("experiment") else set()
        if self.name in {"eval", "tune"} and self.values.get("build_ref"):
            hidden.add("device")
        return [
            FormField(group, self.options[name])
            for group, names in _FORM_GROUPS[self.name]
            for name in names
            if name in self.options and name not in hidden
        ]

    def default(self, name: str) -> Any:
        """Read the selected command's default, preserving its type."""
        value = self.options[name].get_default(self.context)
        # Click represents options without a declared default with UNSET in
        # newer releases. It is absence, not a value to display or toggle.
        return None if value is getattr(click.core, "UNSET", None) else value

    def value(self, name: str) -> Any:
        """Return the edited value or canonical CLI default."""
        return self.values.get(name, self.default(name))

    def field_display(self, name: str) -> str:
        """Explain implicit live-preview behavior without making it explicit."""
        if self.name == "track" and name == "show" and not {"show", "save", "save_txt"}.intersection(self.values):
            from boxmot.engine.commands.track import _is_live_source_value

            if _is_live_source_value(str(self.value("source"))):
                return "auto (live input)"
        return display_value(self.value(name))

    def set_value(self, name: str, value: str | bool | None) -> None:
        """Apply an edit; empty text restores the CLI default."""
        if value is None or value == "":
            self.values.pop(name, None)
        else:
            self.values[name] = value
        if name == "experiment" and value:
            for selector in ("dataset", "detector", "reid"):
                self.values.pop(selector, None)
        if name == "build_ref" and value and self.name in {"eval", "tune"}:
            self.values.pop("device", None)

    def argv(self) -> list[str]:
        """Serialize explicit values as arguments, never as a shell command."""
        arguments = [self.name]
        for name, value in self.values.items():
            option = self.options[name]
            flag = option.opts[0]
            if option.is_flag:
                if value:
                    arguments.append(flag)
                elif option.secondary_opts:
                    arguments.append(option.secondary_opts[0])
                continue
            values = shlex.split(str(value)) if option.multiple else [str(value)]
            for part in values:
                arguments.extend((flag, part))
        return arguments

    def preview(self) -> str:
        """Render a copyable invocation with shell-safe argument quoting."""
        return shlex.join(("boxmot", "--dashboard", *self.argv()))

    def resolved_experiment(self) -> dict[str, Any] | None:
        """Resolve configuration locally, without downloads or model loading."""
        reference = str(self.values.get("experiment") or "")
        if not reference:
            return None
        split = str(self.values.get("split") or "")
        key = (self.name, reference, split)
        if self._experiment_cache is not None and self._experiment_cache[:3] == key:
            return self._experiment_cache[3]
        from boxmot.engine.config.experiments import resolve_experiment_config

        result = resolve_experiment_config(reference, split=split or None, mode=self.name)
        self._experiment_cache = (*key, result)
        return result

    def validate(self) -> list[str]:
        """Parse actual Click options without invoking the command callback."""
        arguments = self.argv()
        with self.command.make_context(self.name, arguments[1:]) as context:
            # make_context runs the CLI's pure option parsing callbacks only;
            # workflow execution stays in the outer ui command after Run.
            values = context.params
            if self.name == "materialize" and not values.get("experiment"):
                raise click.UsageError("Choose an experiment to build a perception dataset.")
            if self.name in {"eval", "tune"} and not values.get("experiment") and not values.get("dataset"):
                raise click.UsageError("Choose an experiment, or a dataset and its perception inputs.")
            if self.name in {"eval", "tune"} and values.get("dataset"):
                self._validate_dataset_selection(values)
            if self.name == "export":
                from boxmot.reid.core.formats import resolve_export_formats

                resolve_export_formats(values["include"])
            if self.name == "train-reid" and not any(values.get(key) for key in ("data_dir", "data", "resume", "cfg")):
                raise click.UsageError("Set the dataset directory before starting training.")
            self._experiment_cache = None
            self.resolved_experiment()
        return arguments

    def _validate_dataset_selection(self, values: dict[str, Any]) -> None:
        """Validate authored component combinations before any build is started."""
        from boxmot.engine.config.datasets import is_saved_2d_dataset, is_sensor_dataset
        from boxmot.engine.config.experiments import load_workflow_dataset, resolve_matching_experiment_path

        dataset = load_workflow_dataset(values["dataset"])
        split = values.get("split") or dataset["default_split"]
        if is_saved_2d_dataset(dataset, split) or is_sensor_dataset(dataset, split):
            return
        if values.get("reid") and not values.get("detector"):
            raise click.UsageError("Select a detector when selecting a ReID profile directly.")
        if not values.get("build_ref") and not values.get("detector"):
            raise click.UsageError("Select a detector for automatic materialization, or provide an existing build.")
        if values.get("detector"):
            resolve_matching_experiment_path(
                dataset=values["dataset"],
                detector=values["detector"],
                reid=values.get("reid"),
                split=split,
                mode=self.name,
            )
