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


def _row(message: str, label: str) -> str:
    """Find a semantic matrix row independently of column widths or borders."""
    matches = [line for line in message.splitlines() if line.strip(" |\t").startswith(label)]
    assert len(matches) == 1, message
    return matches[0]


@pytest.mark.parametrize("mode", ("eval", "tune"))
def test_cli_explains_botsort_inputs_and_saved_sensor_workflow_limit(
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
    assert f"Saved-sensor {mode} currently supports only --tracker eagermot --tracker-backend python." in result.output
    assert "botsort" in result.output
    assert "sensor-fusion" in result.output
    assert "val" in result.output
    assert "AABB" in _row(result.output, "2D boxes")
    assert "Configurable" in _row(result.output, "Images")
    assert "image-directory" in _row(result.output, "Images")
    assert "Configurable" in _row(result.output, "ReID embeddings")
    assert "not exposed" in _row(result.output, "ReID embeddings").lower()
    assert "trackrcnn" in _row(result.output, "Instance masks")
    assert "Unused" in _row(result.output, "3D boxes")
    assert "kitti-detections" in _row(result.output, "3D boxes")
    assert "Unused" in _row(result.output, "Ego motion (poses)")
    assert "camera-to-world-npy" in _row(result.output, "Ego motion (poses)")
    assert "instance-png" in _row(result.output, "Ground-truth masks")
    mismatch = next(line for line in result.output.splitlines() if line.startswith("Dataset/model input mismatch:"))
    assert "instance masks (detections_2d)" in mismatch
    assert "3D boxes (detections_3d)" in mismatch
    assert "calibration" in mismatch
    assert "ego motion (poses)" in mismatch
    assert "ground_truth" not in mismatch
    assert "Declared tracking inputs must be consumed" in result.output
    assert result.output.index(mismatch) < result.output.index("Saved-sensor")
    assert "Extra sensor modalities do not prevent" not in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("botsort", {"ReID embeddings": "Configurable", "Images": "Configurable", "3D boxes": "Unused"}),
        ("bytetrack", {"ReID embeddings": "Unused", "Instance masks": "Unused", "Calibration": "Unused"}),
        ("strongsort", {"ReID embeddings": "Required", "Images": "Required"}),
        ("maf_hda", {"Instance masks": "Required", "Images": "Required", "3D boxes": "Unused"}),
        (
            "eagermot",
            {
                "3D boxes": "Required",
                "Calibration": "Required",
                "Ego motion (poses)": "Optional",
                "Instance masks": "Optional",
            },
        ),
    ),
)
def test_matrix_distinguishes_box_appearance_mask_and_sensor_trackers(
    declared_dataset: Path, name: str, expected: dict[str, str]
) -> None:
    """Required inputs remain distinct from configurable features and unused data."""
    spec = TrackerSpec(name, backend="cpp" if name == "eagermot" else "python")
    with pytest.raises(ValueError) as raised:
        validate_sensor_workflow_inputs(declared_dataset, spec, mode="eval")

    message = str(raised.value)
    for label, requirement in expected.items():
        assert requirement in _row(message, label), message
    assert "instance-png" in _row(message, "Ground-truth masks")


def test_matrix_follows_changed_registry_metadata(declared_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """New input requirements must appear without a second tracker-specific list."""
    definition = TRACKER_DEFINITIONS["botsort"]
    capabilities = replace(definition.capabilities, requires_masks=True, accepts_masks=True)
    monkeypatch.setitem(TRACKER_DEFINITIONS, "botsort", replace(definition, capabilities=capabilities))

    with pytest.raises(ValueError) as raised:
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec("botsort"), mode="eval")

    assert "Required" in _row(str(raised.value), "Instance masks")


def test_supported_workflow_rejects_declared_inputs_the_tracker_cannot_consume(
    declared_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workflow availability cannot override the registered algorithm input contract."""
    definition = TRACKER_DEFINITIONS["eagermot"]
    capabilities = replace(definition.capabilities, accepts_masks=False)
    monkeypatch.setitem(TRACKER_DEFINITIONS, "eagermot", replace(definition, capabilities=capabilities))

    with pytest.raises(ValueError, match="Dataset/model input mismatch:") as raised:
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec("eagermot"), mode="eval")

    assert "instance masks (detections_2d)" in str(raised.value)
    assert "supports only" not in str(raised.value)


def test_unused_input_rejection_respects_split_selection_and_excludes_scoring_labels(declared_dataset: Path) -> None:
    """Only tracking inputs retained by the selected split constrain the tracker."""
    payload = yaml.safe_load(declared_dataset.read_text(encoding="utf-8"))
    payload["splits"]["val"]["modalities"] = {"detections_2d": None, "poses": None}
    _replace_manifest(declared_dataset, payload)

    with pytest.raises(ValueError) as raised:
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec("botsort"), mode="eval")

    mismatch = next(line for line in str(raised.value).splitlines() if line.startswith("Dataset/model input mismatch:"))
    assert "3D boxes (detections_3d)" in mismatch
    assert "calibration" in mismatch
    assert "poses" not in mismatch
    assert "masks" not in mismatch
    assert "ground_truth" not in mismatch
    assert "instance-png" in _row(str(raised.value), "Ground-truth masks")


@pytest.mark.parametrize("mode", ("eval", "tune"))
@pytest.mark.parametrize("name", ("botsort", "eagermot"))
def test_backend_unavailability_is_distinct_from_workflow_support(declared_dataset: Path, mode: str, name: str) -> None:
    with pytest.raises(ValueError) as raised:
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec(name, backend="cpp"), mode=mode)

    message = str(raised.value)
    assert ("has no C++ backend" in message) is (name == "eagermot")
    assert "Python" in message
    assert f"Saved-sensor {mode} currently supports only" in message


@pytest.mark.parametrize("mode", ("eval", "tune"))
@pytest.mark.parametrize(
    "missing",
    (("calibration", "poses"), ("detections_2d", "ground_truth"), ("images", "detections_3d")),
)
def test_missing_modalities_are_reported_together_before_payload_loading(
    declared_dataset: Path, mode: str, missing: tuple[str, ...]
) -> None:
    payload = yaml.safe_load(declared_dataset.read_text(encoding="utf-8"))
    payload["splits"]["val"]["modalities"] = dict.fromkeys(missing)
    if "ground_truth" in missing:
        payload["splits"]["val"]["has_ground_truth"] = False
    _replace_manifest(declared_dataset, payload)

    result = CliRunner().invoke(boxmot, [mode, "--dataset", str(declared_dataset), "--tracker", "eagermot"])

    assert result.exit_code == 2, (result.output, result.exception)
    reason = next(
        line for line in result.output.splitlines() if f"Missing modalities for EagerMOT saved-sensor {mode}:" in line
    )
    assert all(role in reason for role in missing), result.output
    assert result.output.lower().count("not declared") >= len(missing)
    assert "Traceback" not in result.output


def test_missing_workflow_modality_is_not_presented_as_a_botsort_requirement(declared_dataset: Path) -> None:
    payload = yaml.safe_load(declared_dataset.read_text(encoding="utf-8"))
    payload["modalities"].pop("poses")
    _replace_manifest(declared_dataset, payload)

    with pytest.raises(ValueError) as raised:
        validate_sensor_workflow_inputs(declared_dataset, TrackerSpec("botsort"), mode="eval")

    message = str(raised.value)
    assert "Missing modalities for EagerMOT saved-sensor eval: poses." in message
    assert "Unused" in _row(message, "Ego motion (poses)")
    assert "not declared" in _row(message, "Ego motion (poses)")


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

    message = str(raised.value)
    assert "val" in message
    assert "not declared" in _row(message, "Calibration").lower()
    assert "not declared" in _row(message, "Ego motion (poses)").lower()
    assert "kitti-p2" not in _row(message, "Calibration")


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
    assert 'Saved-sensor' in str(error), error
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
