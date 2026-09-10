"""Shared canonical KITTI manifests for CLI and real sensor replay fixtures."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml


def sensor_dataset_fixture(root: Path) -> SimpleNamespace:
    """Create manifest-resolved sequence paths without loading any sensor runtime."""
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
                "format": "kitti-fusion",
                "version": 1,
                "id": "kitti-mots-fusion",
                "classes": {1: "car", 2: "pedestrian"},
                "default_split": "val",
                "replay": "replay.yaml",
                "sequence_layout": {
                    "images": "sequences/{partition}/{sequence}/images",
                    "ground_truth": "sequences/{partition}/{sequence}/ground_truth",
                    "calibration": "sequences/{partition}/{sequence}/calibration.txt",
                    "poses": "sequences/{partition}/{sequence}/poses.npy",
                },
                "splits": {"val": {"partition": "training", "sequences": ["0002"]}},
            }
        ),
        encoding="utf-8",
    )
    predictions = {}
    for role, name, classes in (
        ("image", "trackrcnn", {1: "car", 2: "pedestrian"}),
        ("car", "pointgnn-car", {1: "car"}),
        ("pedestrian", "pointgnn-pedestrian", {2: "pedestrian"}),
    ):
        manifest = root / "predictions" / name / "manifest.yaml"
        manifest.write_text(
            yaml.safe_dump(
                {
                    "format": "trackrcnn" if role == "image" else "pointgnn",
                    "version": 1,
                    "id": name,
                    "classes": classes,
                    "path": "{partition}/{sequence}.txt" if role == "image" else "{partition}/{sequence}",
                    "sequences": {"training": ["0002"]},
                    "provenance": {"source_directory": "test-fixture", "training": "synthetic"},
                }
            ),
            encoding="utf-8",
        )
        predictions[role] = manifest
    (root / "replay.yaml").write_text(
        yaml.safe_dump(
            {"version": 1, "splits": {"val": {role: str(path.relative_to(root)) for role, path in predictions.items()}}}
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        root=root,
        dataset=dataset,
        reader_paths=reader_paths,
        ground_truth=ground_truth,
        prediction_manifests=predictions,
        project=root / "results",
    )
