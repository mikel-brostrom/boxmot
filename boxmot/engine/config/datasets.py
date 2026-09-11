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


def _sensor_input_matrix(
    modalities: Mapping[str, Mapping[str, Any]],
    box_type: str,
    tracker: str,
    capabilities: TrackerCapabilities,
) -> str:
    """Compare declared encodings with static Python algorithm capabilities."""

    def declared(role: str) -> str:
        return str(modalities[role]["format"]) if role in modalities else "not declared"

    def requirement(name: str) -> str:
        if getattr(capabilities, f"requires_{name}"):
            return "Required"
        if not getattr(capabilities, f"accepts_{name}"):
            return "Unused"
        return "Configurable" if name in {"frame", "embeddings"} else "Optional"

    image_detections = modalities.get("detections_2d", {}).get("format") == "trackrcnn"
    rows = [
        ("Input", "Dataset declares", f"{tracker} (Python)"),
        (
            "2D boxes",
            box_type.upper() if "detections_2d" in modalities else "not declared",
            "/".join(sorted(kind.value.upper() for kind in capabilities.geometry_kinds)),
        ),
        ("Images", declared("images"), requirement("frame")),
        ("ReID embeddings", "not exposed" if image_detections else "not declared", requirement("embeddings")),
        ("Instance masks", "trackrcnn" if image_detections else "not declared", requirement("masks")),
        ("3D boxes", declared("detections_3d"), requirement("detections_3d")),
        ("Calibration", declared("calibration"), requirement("camera")),
        ("Ego motion (poses)", declared("poses"), "Optional" if capabilities.accepts_camera else "Unused"),
        ("Ground-truth masks", declared("ground_truth"), "Scoring only"),
    ]
    widths = [max(len(row[column]) for row in rows) for column in range(3)]
    return "\n".join("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip() for row in rows)


def _unused_sensor_inputs(modalities: Mapping[str, Mapping[str, Any]], capabilities: TrackerCapabilities) -> list[str]:
    """Find declared tracking inputs the algorithm cannot consume; exclude scoring labels."""

    inputs = [
        ("images", "frame", "images"),
        ("detections_3d", "detections_3d", "3D boxes (detections_3d)"),
        ("calibration", "camera", "calibration"),
        ("poses", "camera", "ego motion (poses)"),
    ]
    if modalities.get("detections_2d", {}).get("format") == "trackrcnn":
        inputs.insert(1, ("detections_2d", "masks", "instance masks (detections_2d)"))
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
    supported = spec.name == "eagermot" and spec.backend == "python"
    missing = [role for role in _SENSOR_EVALUATION_FORMATS if role not in modalities]
    unused = _unused_sensor_inputs(modalities, definition.capabilities)
    if supported and not missing and not unused:
        return

    mismatch = ""
    if unused:
        mismatch = (
            f"Dataset/model input mismatch: '{spec.name}' declares these dataset inputs unused: "
            f"{', '.join(unused)}.\n"
            "Declared tracking inputs must be consumed; they will not be silently dropped.\n\n"
        )
    reasons = []
    if spec.backend == "cpp" and definition.native_class_path is None:
        reasons.append(f"Tracker '{spec.name}' has no C++ backend.")
    if not supported:
        reasons.append(f"Saved-sensor {mode} currently supports only --tracker eagermot --tracker-backend python.")
    if missing:
        reasons.append(f"Missing modalities for EagerMOT saved-sensor {mode}: {', '.join(missing)}.")
        reasons.append(
            f"To use that MOTS workflow, add these replay and scoring inputs to dataset.yaml for split '{split_name}'."
        )

    notes = [
        "The matrix describes Python tracker inputs; Configurable inputs may be required by enabled features. "
        "Dataset declarations are shown before checking payload files."
    ]
    if definition.capabilities.accepts_embeddings and "detections_2d" in modalities:
        notes.append(
            "The trackrcnn reader does not expose its stored embeddings. "
            "ReID-enabled trackers can generate embeddings from images in a supported image workflow."
        )
    if spec.name != "eagermot":
        notes.append(
            "Choose a tracker that consumes the declared inputs. "
            f"For an intentional image-only experiment with {spec.name}, explicitly select images and ground truth "
            "in a separate dataset config and use a compatible perception build (--build) or --detector."
        )

    matrix = _sensor_input_matrix(modalities, config["box_type"], spec.name, definition.capabilities)
    raise ValueError(
        f"Cannot run saved-sensor {mode} for dataset '{config['id']}' (split '{split_name}') "
        f"with --tracker {spec.name} --tracker-backend {spec.backend}.\n\n"
        + mismatch
        + matrix
        + "\n\n"
        + "\n".join(reasons)
        + "\n\n"
        + "\n".join(notes)
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
