"""Dataset capability requirements for tracking workflows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from boxmot.datasets.config import dataset_modalities, load_dataset_config
from boxmot.datasets.inputs import DatasetInputs, load_dataset_inputs

if TYPE_CHECKING:
    from boxmot.trackers.common.specs import TrackerCapabilities, TrackerSpec

_SENSOR_EVALUATION_FORMATS = {
    "images": "image-directory",
    "ground_truth": "instance-png",
    "detections_2d": "trackrcnn",
    "detections_3d": "kitti-detections",
    "calibration": "kitti-p2",
    "poses": "camera-to-world-npy",
}


def _unused_sensor_inputs(modalities: Mapping[str, Mapping[str, Any]], capabilities: TrackerCapabilities) -> list[str]:
    """Find declared tracking inputs the algorithm cannot consume; exclude scoring labels."""

    inputs = [
        ("images", "frame", "images"),
        ("detections_3d", "detections_3d", "3D boxes"),
        ("calibration", "camera", "calibration"),
        ("poses", "camera", "ego motion"),
    ]
    if modalities.get("detections_2d", {}).get("format") == "trackrcnn":
        inputs.insert(1, ("detections_2d", "masks", "instance masks"))
    return [
        label
        for role, capability, label in inputs
        if role in modalities and not getattr(capabilities, f"accepts_{capability}")
    ]


def validate_sensor_workflow_inputs(
    reference: str | Path,
    spec: TrackerSpec,
    *,
    mode: str,
    split: str | None = None,
) -> None:
    """Explain unsupported sensor selections before reading payloads or loading models."""

    from boxmot.trackers.common.registry import get_tracker_definition

    if mode not in {"eval", "tune"}:
        raise ValueError(f"Unknown sensor workflow: {mode!r}.")
    config = load_dataset_config(reference)
    split_name = config["default_split"] if split is None else split
    modalities = dataset_modalities(config, split_name)
    definition = get_tracker_definition(spec.name)
    if spec.backend == "cpp" and definition.native_class_path is None:
        raise ValueError(f"Tracker '{spec.name}' has no C++ backend.\nUse --tracker-backend python.")

    missing = [role for role in _SENSOR_EVALUATION_FORMATS if role not in modalities]
    unused = _unused_sensor_inputs(modalities, definition.capabilities)
    context = f"dataset '{config['id']}' (split '{split_name}')"
    if unused:
        eager_capabilities = get_tracker_definition("eagermot").capabilities
        advice = (
            "Use --tracker eagermot --tracker-backend python."
            if not missing and not _unused_sensor_inputs(modalities, eager_capabilities)
            else "Select a compatible tracker or explicitly change the dataset modalities."
        )
        raise ValueError(f"'{spec.name}' does not use inputs required by {context}: {', '.join(unused)}.\n{advice}")
    if spec.name != "eagermot" or spec.backend != "python":
        raise ValueError(f"Saved-sensor {mode} supports only --tracker eagermot --tracker-backend python.")
    if missing:
        raise ValueError(
            f"Dataset '{config['id']}' (split '{split_name}') is missing inputs for EagerMOT {mode}: "
            f"{', '.join(missing)}.\nAdd them to dataset.yaml."
        )


def validate_mots_evaluation_inputs(
    classes: Mapping[str, Mapping[str, Any]], ground_truth_options: Mapping[str, Any]
) -> None:
    """Reject annotation conventions the current MOTS metrics cannot interpret."""

    targets = {name: metadata["id"] for name, metadata in classes.items() if metadata["evaluation"] == "target"}
    if targets != {"car": 1, "pedestrian": 2}:
        raise ValueError("MOTS evaluation requires classes.target car: 1 and pedestrian: 2.")
    ignored = {metadata["id"] for metadata in classes.values() if metadata["evaluation"] == "ignore"}
    if ignored.difference({10}):
        raise ValueError("MOTS evaluation supports only ignored class ID 10.")
    if ground_truth_options.get("class_divisor") != 1000 or ground_truth_options.get("background_id") != 0:
        raise ValueError("MOTS evaluation requires ground_truth options class_divisor: 1000 and background_id: 0.")
    if set(ground_truth_options.get("ignore_ids", ())).difference({10000}):
        raise ValueError("MOTS evaluation supports only ground_truth ignore_ids: [10000].")


def load_sensor_evaluation_inputs(
    reference: str | Path,
    *,
    split: str | None = None,
    sequence_names: tuple[str, ...] = (),
) -> DatasetInputs:
    """Resolve generic inputs and enforce the current EagerMOT mask evaluator contract."""

    dataset = load_dataset_inputs(reference, split=split, sequence_names=sequence_names)
    for sequence in dataset.sequences:
        for role, encoding in _SENSOR_EVALUATION_FORMATS.items():
            source = sequence.modalities.get(role)
            if source is None:
                raise ValueError(f"EagerMOT mask evaluation requires {role} for sequence {sequence.sequence_id}.")
            if source.format != encoding:
                raise ValueError(f"EagerMOT mask evaluation requires {role} format {encoding}.")
        validate_mots_evaluation_inputs(dataset.classes, sequence.modalities["ground_truth"].options)
    return dataset
