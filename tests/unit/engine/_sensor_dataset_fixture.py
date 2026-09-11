"""Shared dataset schema for CLI and real multimodal sequence fixtures."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from boxmot.datasets.inputs import SequenceInputs


def sensor_dataset_fixture(root: Path) -> SimpleNamespace:
    """Create one dataset YAML and sequence paths without loading sensor readers."""
    sequence = root / "sequences/training/0002"
    reader_paths = {
        "images": sequence / "images",
        "calibration": sequence / "calibration.txt",
        "poses": sequence / "poses.npy",
        "detections_2d": root / "predictions/trackrcnn/training/0002.txt",
        "car_detections_3d": root / "predictions/pointgnn-car/training/0002",
        "pedestrian_detections_3d": root / "predictions/pointgnn-pedestrian/training/0002",
    }
    ground_truth = sequence / "ground_truth"
    for directory in (
        reader_paths["images"],
        ground_truth,
        reader_paths["car_detections_3d"],
        reader_paths["pedestrian_detections_3d"],
        reader_paths["detections_2d"].parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for name in ("calibration", "poses", "detections_2d"):
        reader_paths[name].touch()
    dataset = root / "dataset.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "id": "kitti-mots-fusion",
                "format": {"layout": "sequence", "box_type": "aabb"},
                "storage": {"root": "."},
                "classes": {"target": {"car": 1, "pedestrian": 2}, "ignore": {"ignore": 10}},
                "fps": 10,
                "default_split": "val",
                "modalities": {
                    "images": {"format": "image-directory", "path": "sequences/{partition}/{sequence}/images"},
                    "ground_truth": {
                        "format": "instance-png",
                        "path": "sequences/{partition}/{sequence}/ground_truth",
                        "options": {"class_divisor": 1000, "background_id": 0, "ignore_ids": [10000]},
                    },
                    "calibration": {
                        "format": "kitti-p2",
                        "path": "sequences/{partition}/{sequence}/calibration.txt",
                    },
                    "poses": {"format": "camera-to-world-npy", "path": "sequences/{partition}/{sequence}/poses.npy"},
                    "detections_2d": {
                        "format": "trackrcnn",
                        "path": "predictions/trackrcnn/{partition}/{sequence}.txt",
                    },
                    "detections_3d": {
                        "format": "kitti-detections",
                        "paths": [
                            "predictions/pointgnn-car/{partition}/{sequence}",
                            "predictions/pointgnn-pedestrian/{partition}/{sequence}",
                        ],
                        "options": {"score_transform": "odds", "ignore_classes": ["Cyclist"]},
                    },
                },
                "splits": {"val": {"partition": "training", "sequences": ["0002"], "has_ground_truth": True}},
            }
        ),
        encoding="utf-8",
    )

    def sequence_inputs() -> SequenceInputs:
        """Resolve current fixture paths through the shared dataset input loader."""
        from boxmot.datasets.inputs import load_dataset_inputs

        return load_dataset_inputs(dataset).sequences[0]

    return SimpleNamespace(
        root=root,
        dataset=dataset,
        reader_paths=reader_paths,
        ground_truth=ground_truth,
        sequence_inputs=sequence_inputs,
        project=root / "results",
    )
