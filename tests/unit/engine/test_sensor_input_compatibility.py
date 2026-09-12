"""Explain saved-sensor workflow limits using declared data and tracker inputs."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.config.datasets import validate_sensor_workflow_inputs
from boxmot.engine.config.experiments import resolve_experiment_config
from boxmot.trackers.common.registry import TRACKER_DEFINITIONS
from boxmot.trackers.common.specs import TrackerSpec
from tests._paths import REPO_ROOT


@pytest.fixture
def declared_dataset(tmp_path: Path) -> Path:
    """Provide only a portable manifest, with none of its declared payloads."""
    source = REPO_ROOT / "boxmot/configs/datasets/sensor-fusion.yaml"
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return dataset


def _replace_manifest(dataset: Path, payload: dict[str, Any]) -> None:
    """Persist a test's modality selection without manufacturing sensor files."""
    dataset.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _concise_error(message: str) -> list[str]:
    """Require one reason and at most one next step, without a capability table."""
    lines = message.strip().splitlines()
    assert 1 <= len(lines) <= 2, message
    assert all(lines), message
    assert "Dataset declares" not in message
    assert "Traceback" not in message
    return lines


@pytest.mark.parametrize("mode", ("eval", "tune"))
def test_cli_names_unused_botsort_inputs_and_one_next_step(
    declared_dataset: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = importlib.import_module(f"boxmot.engine.commands.{mode}")

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Incompatible declarations must be explained before preparing or dispatching a workflow.")

    monkeypatch.setattr(command, "_prepare_replay_build", unexpected)
    monkeypatch.setattr(command, "_dispatch_cli_workflow", unexpected)
    result = CliRunner().invoke(
        boxmot,
        [mode, "--dataset", str(declared_dataset.parent), "--tracker", "botsort", "--split", "val"],
    )

    assert result.exit_code == 2, (result.output, result.exception)
    reason, action = _concise_error(result.output.split("Error: ", 1)[1])
    assert "'botsort' does not use inputs required by dataset 'sensor-fusion' (split 'val'):" in reason
    assert "instance masks, 3D boxes, calibration, ego motion." in reason
    assert "ground_truth" not in reason
    assert action == "Use --tracker eagermot --tracker-backend python."


@pytest.mark.parametrize(
    ("name", "unused_masks"),
    (
        ("botsort", True),
        ("bytetrack", True),
        ("strongsort", True),
        ("maf_hda", False),
    ),
)
def test_unused_inputs_reflect_box_appearance_and_mask_tracker_contracts(
    declared_dataset: Path, name: str, unused_masks: bool
) -> None:
    """Accepted inputs stay out of the rejection for each algorithm family."""
    with pytest.raises(ValueError) as raised:
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec(name), mode="eval")

    reason = _concise_error(str(raised.value))[0]
    assert "does not use inputs required" in reason
    assert ("instance masks" in reason) is unused_masks
    assert "3D boxes, calibration, ego motion" in reason
    assert "images" not in reason
    assert "embeddings" not in reason


def test_mismatch_follows_changed_registry_metadata(declared_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """New accepted inputs must stop being rejected without a tracker-specific list."""
    definition = TRACKER_DEFINITIONS["botsort"]
    capabilities = replace(definition.capabilities, requires_masks=True, accepts_masks=True)
    monkeypatch.setitem(TRACKER_DEFINITIONS, "botsort", replace(definition, capabilities=capabilities))

    with pytest.raises(ValueError) as raised:
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec("botsort"), mode="eval")

    reason = _concise_error(str(raised.value))[0]
    assert "instance masks" not in reason
    assert "3D boxes, calibration, ego motion" in reason


def test_supported_workflow_rejects_declared_inputs_the_tracker_cannot_consume(
    declared_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workflow availability cannot override the registered algorithm input contract."""
    definition = TRACKER_DEFINITIONS["eagermot"]
    capabilities = replace(definition.capabilities, accepts_masks=False)
    monkeypatch.setitem(TRACKER_DEFINITIONS, "eagermot", replace(definition, capabilities=capabilities))

    with pytest.raises(ValueError, match="does not use inputs required") as raised:
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec("eagermot"), mode="eval")

    assert "instance masks" in _concise_error(str(raised.value))[0]
    assert "supports only" not in str(raised.value)
    assert "Use --tracker eagermot" not in str(raised.value)


def test_unused_input_rejection_respects_split_selection_and_excludes_scoring_labels(declared_dataset: Path) -> None:
    """Only tracking inputs retained by the selected split constrain the tracker."""
    payload = yaml.safe_load(declared_dataset.read_text(encoding="utf-8"))
    payload["splits"]["val"]["modalities"] = {"detections_2d": None, "poses": None}
    _replace_manifest(declared_dataset, payload)

    with pytest.raises(ValueError) as raised:
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec("botsort"), mode="eval")

    mismatch = _concise_error(str(raised.value))[0]
    assert "3D boxes" in mismatch
    assert "calibration" in mismatch
    assert "poses" not in mismatch
    assert "masks" not in mismatch
    assert "ground_truth" not in mismatch
    assert "ego motion" not in mismatch


@pytest.mark.parametrize("mode", ("eval", "tune"))
@pytest.mark.parametrize("name", ("botsort", "eagermot"))
def test_backend_unavailability_is_distinct_from_workflow_support(declared_dataset: Path, mode: str, name: str) -> None:
    with pytest.raises(ValueError) as raised:
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec(name, backend="cpp"), mode=mode)

    message = str(raised.value)
    _concise_error(message)
    assert ("has no C++ backend" in message) is (name == "eagermot")
    if name == "eagermot":
        assert message.endswith("Use --tracker-backend python.")
        assert "does not use inputs required" not in message
    else:
        assert "does not use inputs required" in message


@pytest.mark.parametrize("mode", ("eval", "tune"))
def test_compatible_inputs_still_require_a_supported_workflow(
    declared_dataset: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Algorithm input support does not imply that its replay workflow exists."""
    definition = TRACKER_DEFINITIONS["botsort"]
    capabilities = replace(
        definition.capabilities,
        accepts_masks=True,
        accepts_detections_3d=True,
        accepts_camera=True,
        accepts_ego_motion=True,
    )
    monkeypatch.setitem(TRACKER_DEFINITIONS, "botsort", replace(definition, capabilities=capabilities))

    with pytest.raises(ValueError) as raised:
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec("botsort", backend="cpp"), mode=mode)

    assert _concise_error(str(raised.value)) == [
        f"Saved-sensor {mode} supports only --tracker eagermot --tracker-backend python."
    ]


def test_calibration_acceptance_does_not_imply_ego_motion_support(
    declared_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = TRACKER_DEFINITIONS["eagermot"]
    capabilities = replace(definition.capabilities, accepts_ego_motion=False)
    monkeypatch.setitem(TRACKER_DEFINITIONS, "eagermot", replace(definition, capabilities=capabilities))

    with pytest.raises(ValueError) as raised:
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec("eagermot"), mode="eval")

    reason = _concise_error(str(raised.value))[0]
    assert reason.endswith("ego motion.")
    assert "calibration" not in reason

    payload = yaml.safe_load(declared_dataset.read_text(encoding="utf-8"))
    payload["modalities"].pop("poses")
    _replace_manifest(declared_dataset, payload)
    validate_sensor_workflow_inputs(declared_dataset, TrackerSpec("eagermot"), mode="eval")


@pytest.mark.parametrize("mode", ("eval", "tune"))
@pytest.mark.parametrize(
    ("missing", "required"),
    (
        (("calibration", "poses"), ("calibration",)),
        (("detections_2d", "ground_truth"), ("detections_2d", "ground_truth")),
        (("images", "detections_3d"), ("images", "detections_3d")),
    ),
)
def test_missing_modalities_are_reported_together_before_payload_loading(
    declared_dataset: Path, mode: str, missing: tuple[str, ...], required: tuple[str, ...]
) -> None:
    payload = yaml.safe_load(declared_dataset.read_text(encoding="utf-8"))
    payload["splits"]["val"]["modalities"] = dict.fromkeys(missing)
    _replace_manifest(declared_dataset, payload)

    result = CliRunner().invoke(boxmot, [mode, "--dataset", str(declared_dataset), "--tracker", "eagermot"])

    assert result.exit_code == 2, (result.output, result.exception)
    reason, action = _concise_error(result.output.split("Error: ", 1)[1])
    assert f"Dataset 'sensor-fusion' (split 'val') is missing inputs for EagerMOT {mode}:" in reason
    assert all(role in reason for role in required), result.output
    assert "poses" not in reason
    assert action == "Add them to dataset.yaml."


def test_optional_poses_do_not_prevent_recommending_a_compatible_tracker(declared_dataset: Path) -> None:
    payload = yaml.safe_load(declared_dataset.read_text(encoding="utf-8"))
    payload["modalities"].pop("poses")
    _replace_manifest(declared_dataset, payload)

    with pytest.raises(ValueError) as raised:
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec("botsort"), mode="eval")

    message = str(raised.value)
    reason = _concise_error(message)[0]
    assert "does not use inputs required" in reason
    assert "ego motion" not in reason
    assert "poses" not in reason
    assert "missing inputs" not in message
    assert message.endswith("Use --tracker eagermot --tracker-backend python.")


def test_compatibility_uses_selected_split_overrides_and_default(declared_dataset: Path) -> None:
    payload = yaml.safe_load(declared_dataset.read_text(encoding="utf-8"))
    poses = payload["modalities"].pop("poses")
    payload["splits"]["train"]["modalities"] = {"poses": poses}
    payload["splits"]["val"]["modalities"] = {"calibration": None}
    _replace_manifest(declared_dataset, payload)

    assert (
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec("eagermot"), mode="eval", split="train") is None
    )
    with pytest.raises(ValueError) as raised:
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec("eagermot"), mode="eval")

    reason = _concise_error(str(raised.value))[0]
    assert "split 'val'" in reason
    assert "calibration." in reason
    assert "poses" not in reason


@pytest.mark.parametrize("mode", ("eval", "tune"))
@pytest.mark.parametrize("calibrate_kf", (False, True))
def test_sensor_workflows_accept_no_declared_ego_poses(
    declared_dataset: Path, mode: str, calibrate_kf: bool
) -> None:
    payload = yaml.safe_load(declared_dataset.read_text(encoding="utf-8"))
    payload["modalities"].pop("poses")
    _replace_manifest(declared_dataset, payload)

    validate_sensor_workflow_inputs(
        declared_dataset, TrackerSpec("eagermot"), mode=mode, calibrate_kf=calibrate_kf
    )


@pytest.mark.parametrize("mode", ("eval", "tune"))
def test_spatial_scoring_accepts_no_declared_image_detections(declared_dataset: Path, mode: str) -> None:
    payload = yaml.safe_load(declared_dataset.read_text(encoding="utf-8"))
    payload["splits"]["val"]["modalities"] = {"detections_2d": None, "poses": None}
    _replace_manifest(declared_dataset, payload)

    validate_sensor_workflow_inputs(declared_dataset, TrackerSpec("eagermot"), mode=mode, eval_3d=True)
    with pytest.raises(ValueError, match="missing inputs.*detections_2d"):
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec("eagermot"), mode=mode)


@pytest.mark.parametrize("mode", ("eval", "tune"))
@pytest.mark.parametrize("build", (False, True))
def test_saved_mask_only_dataset_is_not_silently_routed_to_perception(
    declared_dataset: Path, mode: str, build: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from boxmot.datasets.inputs import resolve_sensor_dataset_config_path

    payload = yaml.safe_load(declared_dataset.read_text())
    for role in ("detections_3d", "calibration", "poses"):
        payload["modalities"].pop(role)
    _replace_manifest(declared_dataset, payload)
    assert resolve_sensor_dataset_config_path(declared_dataset) == declared_dataset

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Declared saved detections must be validated before resolving a perception build.")

    command = importlib.import_module(f"boxmot.engine.commands.{mode}")
    monkeypatch.setattr(command, "_prepare_replay_build", unexpected)
    monkeypatch.setattr(command, "_dispatch_cli_workflow", unexpected)
    options = ["--build", "unused-build"] if build else []
    result = CliRunner().invoke(
        boxmot, [mode, "--dataset", str(declared_dataset), "--tracker", "maf_hda", *options]
    )

    assert result.exit_code == 2, (result.output, result.exception)
    reason, action = _concise_error(result.output.split("Error: ", 1)[1])
    assert f"saved TrackR-CNN inputs that {mode} cannot replay" in reason
    assert "boxmot track --detections ... --images ... --instances ..." in action
    assert "eagermot" not in result.output


def test_eagermot_with_only_image_detections_reports_its_missing_sensor_inputs(declared_dataset: Path) -> None:
    payload = yaml.safe_load(declared_dataset.read_text())
    for role in ("detections_3d", "calibration", "poses"):
        payload["modalities"].pop(role)
    _replace_manifest(declared_dataset, payload)

    with pytest.raises(ValueError) as raised:
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec("eagermot"), mode="eval")

    reason, action = _concise_error(str(raised.value))
    assert "missing inputs for EagerMOT eval: detections_3d, calibration." in reason
    assert action == "Add them to dataset.yaml."
    assert "boxmot track" not in str(raised.value)


@pytest.mark.parametrize("mode", ("eval", "tune"))
@pytest.mark.parametrize("spatial", (False, True))
def test_perception_experiments_cannot_discard_declared_saved_tracking_inputs(
    declared_dataset: Path, mode: str, spatial: bool
) -> None:
    payload = yaml.safe_load(declared_dataset.read_text())
    if not spatial:
        for role in ("detections_3d", "calibration", "poses"):
            payload["modalities"].pop(role)
        _replace_manifest(declared_dataset, payload)
    experiment = declared_dataset.parent / "saved-inputs.yaml"
    experiment.write_text(
        yaml.safe_dump({"dataset": {"ref": "dataset.yaml", "split": "val"}, "detector": {"ref": "unloaded"}})
    )

    with pytest.raises(ValueError) as raised:
        resolve_experiment_config(experiment, mode=mode)
    reason, action = _concise_error(str(raised.value))
    if spatial:
        assert "Perception experiments cannot consume" in reason
        assert "detections_3d" in reason
        assert "--dataset" in action
    else:
        assert "saved TrackR-CNN inputs" in reason
        assert "boxmot track --detections" in action


def test_saved_input_routing_and_experiments_respect_image_only_split_overrides(declared_dataset: Path) -> None:
    from boxmot.datasets.inputs import resolve_sensor_dataset_config_path

    payload = yaml.safe_load(declared_dataset.read_text())
    payload["splits"]["val"]["modalities"] = dict.fromkeys(("detections_2d", "detections_3d", "calibration", "poses"))
    _replace_manifest(declared_dataset, payload)
    assert resolve_sensor_dataset_config_path(declared_dataset, split="train") == declared_dataset
    assert resolve_sensor_dataset_config_path(declared_dataset, split="val") is None
    experiment = declared_dataset.parent / "image-only.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "dataset": {"ref": "dataset.yaml", "split": "val"},
                "detector": {"ref": "yolo26n", "checkpoint": "default"},
                "evaluation": {"class_map": {"car": "car", "pedestrian": "person"}},
            }
        )
    )
    assert resolve_experiment_config(experiment, mode="eval")["dataset"]["split"] == "val"


@pytest.mark.parametrize("mode", ("eval", "tune"))
def test_diagnostics_do_not_import_readers_trackers_or_execution_runtimes(declared_dataset: Path, mode: str) -> None:
    """Checking declarations must work without any tracking or decoding runtime."""
    probe = """
import json
import sys
from boxmot.engine.config.datasets import validate_sensor_workflow_inputs
from boxmot.trackers.common.specs import TrackerSpec
validate_sensor_workflow_inputs(sys.argv[2], TrackerSpec('eagermot'), mode=sys.argv[1])
try:
    validate_sensor_workflow_inputs(sys.argv[2], TrackerSpec('botsort'), mode=sys.argv[1])
except ValueError as error:
    assert 'does not use inputs required' in str(error), error
else:
    raise AssertionError('BoT-SORT should receive a saved-sensor diagnostic')
blocked = ('torch', 'numpy', 'cv2', 'PIL', 'optuna', 'ray', 'pyarrow',
           'boxmot.datasets.readers', 'boxmot.datasets.sequence',
           'boxmot.engine.eval.evaluator', 'boxmot.engine.eval.eagermot_kitti',
           'boxmot.engine.tuning.tuner')
loaded = [name for name in sys.modules if
          any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)
          or (name.startswith('boxmot.trackers.') and name.endswith(('.tracker', '.native')))]
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, mode, str(declared_dataset)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
