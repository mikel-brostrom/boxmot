"""Dataset capability requirements for tracking workflows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from boxmot.datasets.inputs import DatasetInputs, load_dataset_inputs


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
    required = {
        "images": "image-directory",
        "ground_truth": "instance-png",
        "detections_2d": "trackrcnn",
        "detections_3d": "kitti-detections",
        "calibration": "kitti-p2",
        "poses": "camera-to-world-npy",
    }
    for sequence in dataset.sequences:
        for role, encoding in required.items():
            source = sequence.modalities.get(role)
            if source is None:
                raise ValueError(f"EagerMOT mask evaluation requires {role} for sequence {sequence.sequence_id}.")
            if source.format != encoding:
                raise ValueError(f"EagerMOT mask evaluation requires {role} format {encoding}.")
        validate_mots_evaluation_inputs(dataset.classes, sequence.modalities["ground_truth"].options)
    return dataset
