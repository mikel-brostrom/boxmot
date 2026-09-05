from __future__ import annotations

import pytest

from boxmot.engine.experiment_config import (
    EXPERIMENT_CONFIGS_DIR,
    ConfigurationError,
    resolve_experiment_config,
)


def test_every_built_in_experiment_has_a_materializable_model_source() -> None:
    paths = sorted(EXPERIMENT_CONFIGS_DIR.rglob("*.yaml"))

    assert paths
    for path in paths:
        resolved = resolve_experiment_config(path, mode="materialize")
        assert resolved["detections"]["source"] == "model", path
        assert resolved["detector"]["model"], path


@pytest.mark.parametrize(
    "experiment_id",
    (
        "mmot-obb-test-yolo11l-lmbn",
        "mmot-obb-mini-train-yolo11l-lmbn",
    ),
)
def test_mmot_benchmark_experiments_use_perspective_reid_crops(experiment_id) -> None:
    resolved = resolve_experiment_config(experiment_id, mode="materialize")

    assert resolved["reid"]["crop_strategy"] == "perspective"


def test_mmot_experiment_uses_one_native_identity_class_domain() -> None:
    resolved = resolve_experiment_config("mmot-obb-test-yolo11l-lmbn", mode="materialize")

    assert resolved["reid"]["crop_strategy"] == "perspective"
    assert [
        (entry["name"], entry["dataset_id"], entry["detector_id"])
        for entry in resolved["evaluation"]["classes"]
    ] == [
        ("car", 0, 0),
        ("bike", 1, 1),
        ("pedestrian", 2, 2),
        ("van", 3, 3),
        ("truck", 4, 4),
        ("bus", 5, 5),
        ("tricycle", 6, 6),
        ("awning-bike", 7, 7),
    ]


def test_experiment_rejects_unknown_reid_crop_strategy(tmp_path) -> None:
    experiment = tmp_path / "invalid-crop.yaml"
    experiment.write_text(
        """
id: invalid-crop
dataset:
  ref: mmot
  split: test
detections:
  source: model
  model:
    ref: yolo11l-mmot-obb
    checkpoint: default
reid:
  ref: lmbn-n-duke
  crop_strategy: diagonal
evaluation:
  class_map: auto
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="reid.crop_strategy"):
        resolve_experiment_config(experiment, mode="materialize")


@pytest.mark.parametrize("crop_strategy", (False, 0, ""))
def test_experiment_rejects_non_text_reid_crop_strategy(tmp_path, crop_strategy) -> None:
    experiment = tmp_path / "invalid-crop.yaml"
    experiment.write_text(
        f"""
id: invalid-crop
dataset:
  ref: mmot
  split: test
detections:
  source: model
  model:
    ref: yolo11l-mmot-obb
    checkpoint: default
reid:
  ref: lmbn-n-duke
  crop_strategy: {crop_strategy!r}
evaluation:
  class_map: auto
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="reid.crop_strategy"):
        resolve_experiment_config(experiment, mode="materialize")


@pytest.mark.parametrize("crop_strategy", ("perspective", "rotated"))
def test_experiment_rejects_oriented_reid_crop_for_aabb_dataset(tmp_path, crop_strategy) -> None:
    experiment = tmp_path / "invalid-aabb-crop.yaml"
    experiment.write_text(
        f"""
id: invalid-aabb-crop
dataset:
  ref: mot17
  split: val
detections:
  source: model
  model:
    ref: yolox
    checkpoint: x
reid:
  ref: lmbn-n-duke
  crop_strategy: {crop_strategy}
evaluation:
  class_map: auto
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="requires an OBB dataset"):
        resolve_experiment_config(experiment, mode="materialize")
