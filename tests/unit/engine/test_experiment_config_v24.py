from __future__ import annotations

import pytest

from boxmot.engine.experiment_config import (
    EXPERIMENT_CONFIGS_DIR,
    ConfigurationError,
    resolve_experiment_config,
    resolve_experiment_path,
)
from boxmot.utils.config import load_yaml_mapping


def test_every_built_in_experiment_has_a_materializable_detector() -> None:
    paths = sorted(EXPERIMENT_CONFIGS_DIR.rglob("*.yaml"))

    assert paths
    for path in paths:
        resolved = resolve_experiment_config(path, mode="materialize")
        relative = path.relative_to(EXPERIMENT_CONFIGS_DIR).with_suffix("")
        assert "id" not in load_yaml_mapping(path), path
        assert resolved["id"] == "-".join(relative.parts), path
        assert "detections" not in resolved, path
        assert resolved["detector"]["ref"], path
        assert resolved["detector"]["checkpoint"], path
        assert resolved["detector"]["model"], path
        if resolved["reid"] is not None:
            assert "crop_strategy" not in resolved["reid"], path


@pytest.mark.parametrize("reference", ("test-yolo11l-lmbn", "test-yolo11l-lmbn.yaml"))
def test_experiment_resolves_unique_bare_filename_or_stem(reference: str) -> None:
    expected = EXPERIMENT_CONFIGS_DIR / "mmot-obb" / "test-yolo11l-lmbn.yaml"

    assert resolve_experiment_path(reference) == expected.resolve()


@pytest.mark.parametrize(
    "reference",
    ("mot17/ablation-yolox-lmbn", "mot17/ablation-yolox-lmbn.yaml"),
)
def test_experiment_resolves_catalog_relative_filename_or_stem(reference: str) -> None:
    expected = EXPERIMENT_CONFIGS_DIR / "mot17" / "ablation-yolox-lmbn.yaml"

    assert resolve_experiment_path(reference) == expected.resolve()


def test_mot17_osnet_experiment_references_component_filenames() -> None:
    path = EXPERIMENT_CONFIGS_DIR / "mot17" / "ablation-yolox-osnet.yaml"
    authored = load_yaml_mapping(path)

    assert authored["dataset"]["ref"] == "mot17.yaml"
    assert authored["detector"]["ref"] == "yolox-x-mot17.yaml"
    assert authored["reid"]["ref"] == "osnet-x0-25-msmt17.yaml"

    resolved = resolve_experiment_config(path, mode="materialize")
    assert resolved["reid"]["id"] == "osnet-x0-25-msmt17"


def test_experiment_rejects_ambiguous_bare_filename() -> None:
    with pytest.raises(ConfigurationError, match='Ambiguous experiment filename "val-yolox-lmbn"'):
        resolve_experiment_path("val-yolox-lmbn")


def test_experiment_does_not_resolve_former_declared_id() -> None:
    with pytest.raises(FileNotFoundError, match="filename"):
        resolve_experiment_path("mmot-obb-test-yolo11l-lmbn")


def test_experiment_does_not_replace_an_unrelated_filename_suffix() -> None:
    with pytest.raises(FileNotFoundError, match="filename"):
        resolve_experiment_path("test-yolo11l-lmbn.json")


def test_external_experiment_identity_comes_from_filename(tmp_path) -> None:
    source = EXPERIMENT_CONFIGS_DIR / "mmot-obb" / "test-yolo11l-lmbn.yaml"
    experiment = tmp_path / "custom-mmot.yaml"
    experiment.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    resolved = resolve_experiment_config(experiment, mode="materialize")

    assert resolved["id"] == "custom-mmot"


def test_experiment_rejects_authored_id(tmp_path) -> None:
    experiment = tmp_path / "custom-mmot.yaml"
    experiment.write_text("id: unrelated-name\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match='must not define "id"'):
        resolve_experiment_config(experiment, mode="materialize")


def test_experiment_rejects_non_kebab_case_filename(tmp_path) -> None:
    experiment = tmp_path / "invalid_name.yaml"
    experiment.write_text("dataset: {}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="filename stem"):
        resolve_experiment_config(experiment, mode="materialize")


def test_experiment_rejects_legacy_detections_key(tmp_path) -> None:
    experiment = tmp_path / "legacy-detections.yaml"
    experiment.write_text("detections: {}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="unknown keys: detections"):
        resolve_experiment_config(experiment, mode="materialize")


def test_experiment_rejects_unknown_detector_key(tmp_path) -> None:
    experiment = tmp_path / "invalid-detector.yaml"
    experiment.write_text(
        """
dataset:
  ref: mmot
  split: test
detector:
  ref: yolo11l-mmot-obb
  checkpoint: default
  source: model
evaluation:
  class_map: auto
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="detector has unknown keys: source"):
        resolve_experiment_config(experiment, mode="materialize")


def test_experiment_preserves_external_detector_reference(tmp_path) -> None:
    detector = tmp_path / "custom-detector.yaml"
    detector.write_text(
        """
id: custom-detector
box_type: aabb
classes:
  0: person
inference:
  image_size: [640, 640]
  confidence_threshold: 0.25
checkpoints:
  default:
    path: custom-detector.pt
""".strip(),
        encoding="utf-8",
    )
    experiment = tmp_path / "external-detector.yaml"
    experiment.write_text(
        f"""
dataset:
  ref: mot17
  split: val
detector:
  ref: {detector}
  checkpoint: default
evaluation:
  class_map:
    pedestrian: person
""".strip(),
        encoding="utf-8",
    )

    resolved = resolve_experiment_config(experiment, mode="materialize")

    assert resolved["detector"]["ref"] == str(detector)
    assert resolved["detector"]["id"] == "custom-detector"
    assert resolved["detector"]["checkpoint"] == "default"


def test_mot17_mini_ci_experiment_uses_the_prepared_yolo26n_profile() -> None:
    resolved = resolve_experiment_config("mot17-mini/train-yolo26n-lmbn", mode="materialize")

    assert "detections" not in resolved
    assert resolved["detector"]["ref"] == "yolo26n"
    assert resolved["detector"]["checkpoint"] == "default"
    assert resolved["detector"]["id"] == "yolo26n"
    assert resolved["detector"]["model"] == "models/yolo26n.pt"
    assert resolved["detector"]["uri"] == (
        "https://github.com/mikel-brostrom/boxmot/releases/download/v22.0.0/yolo26n.pt"
    )
    assert resolved["detector"]["sha256"] == "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
    assert resolved["detector"]["classes"][0] == "person"
    assert resolved["evaluation"]["classes"] == [
        {
            "name": "pedestrian",
            "dataset_id": 1,
            "detector_name": "person",
            "detector_id": 0,
        }
    ]


def test_mmot_experiment_uses_one_native_identity_class_domain() -> None:
    resolved = resolve_experiment_config("test-yolo11l-lmbn", mode="materialize")

    assert [
        (entry["name"], entry["dataset_id"], entry["detector_id"]) for entry in resolved["evaluation"]["classes"]
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


def test_mmot_osnet_experiment_resolves_osnet_profile() -> None:
    resolved = resolve_experiment_config("mmot-obb/test-yolo11l-osnet.yaml", mode="materialize")

    assert resolved["id"] == "mmot-obb-test-yolo11l-osnet"
    assert resolved["reid"]["id"] == "osnet-x0-25-msmt17"
    assert resolved["reid"]["model"] == "models/osnet_x0_25_msmt17.pt"
    assert resolved["reid"]["precision"] == "fp32"
    assert resolved["reid"]["image_size"] == [256, 128]
    assert "crop_strategy" not in resolved["reid"]


def test_experiment_rejects_reid_crop_strategy(tmp_path) -> None:
    experiment = tmp_path / "invalid-crop.yaml"
    experiment.write_text(
        """
dataset:
  ref: mmot
  split: test
detector:
  ref: yolo11l-mmot-obb
  checkpoint: default
reid:
  ref: lmbn-n-duke
  crop_strategy: perspective
evaluation:
  class_map: auto
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="reid has unknown keys: crop_strategy"):
        resolve_experiment_config(experiment, mode="materialize")
