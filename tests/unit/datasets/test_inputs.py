"""Resolve portable, independently encoded dataset modalities through one schema."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml

from boxmot.datasets.config import (
    ConfigurationError,
    dataset_modalities,
    load_dataset_config,
    resolve_dataset_config_path,
    resolve_dataset_storage_root,
)
from boxmot.datasets.inputs import load_dataset_inputs, resolve_dataset_inputs, resolve_sensor_dataset_config_path


def _write(path: Path, payload: dict[str, Any]) -> Path:
    """Write an authored configuration with stable mapping order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _fixture(root: Path) -> Path:
    """Create generic classes, custom split membership, and distinct prediction sets."""

    payload = {
        "id": "my-sensor-dataset",
        "format": {"layout": "sequence", "box_type": "aabb"},
        "storage": {"root": "."},
        "classes": {"target": {"vehicle": 7, "walker": 12}, "ignore": {"untracked": 99}},
        "fps": 12.5,
        "default_split": "val",
        "modalities": {
            "images": {"format": "image-directory", "path": "sequences/{partition}/{sequence}/images"},
            "ground_truth": {
                "format": "instance-png",
                "path": "sequences/{partition}/{sequence}/ground_truth",
                "options": {"class_divisor": 1000, "background_id": 0},
            },
            "calibration": {"format": "kitti-p2", "path": "calibration/{partition}.txt"},
            "poses": {"format": "camera-to-world-npy", "path": "sequences/{partition}/{sequence}/poses.npy"},
            "detections_2d": {"format": "trackrcnn", "path": "predictions/image/{partition}/{sequence}.txt"},
            "detections_3d": {
                "format": "kitti-detections",
                "paths": [
                    "predictions/3d-first/{partition}/{sequence}",
                    "predictions/3d-second/{partition}/{sequence}",
                ],
                "options": {"score_transform": "identity"},
            },
        },
        "splits": {
            "val": {
                "partition": "recordings",
                "sequences": ["drive-002", "drive-003"],
                "modalities": {
                    "detections_3d": {
                        "format": "kitti-detections",
                        "path": "predictions/3d-{split}/{partition}/{sequence}",
                        "options": {"score_transform": "odds"},
                    }
                },
            },
            "train": {"partition": "recordings", "sequences": ["drive-001"]},
            "all": {"partition": "recordings", "sequences": ["drive-001", "drive-002", "drive-003"]},
            "test": {
                "partition": "new-recordings",
                "sequences": ["drive-004"],
                "modalities": {"ground_truth": None},
            },
        },
    }
    path = _write(root / "dataset.yaml", payload)
    for split_name, split in payload["splits"].items():
        for sequence in split["sequences"]:
            for specification in {**payload["modalities"], **split.get("modalities", {})}.values():
                if specification is None:
                    continue
                for template in specification.get("paths", [specification.get("path")]):
                    target = root / template.format(partition=split["partition"], split=split_name, sequence=sequence)
                    if specification["format"] in {"image-directory", "instance-png", "kitti-detections"}:
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.touch()
    return path


def _change(path: Path, keys: tuple[str, ...], value: Any) -> None:
    """Change one authored value without depending on normalization internals."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    target = payload
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    _write(path, payload)


def test_folder_and_yaml_resolve_native_classes_and_paths_independently_of_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _fixture(tmp_path / "dataset with spaces")
    monkeypatch.chdir(tmp_path)

    dataset = load_dataset_inputs(path.parent)

    assert dataset == load_dataset_inputs(path)
    assert dataset.config_path == resolve_sensor_dataset_config_path(path.parent) == path.resolve()
    assert dataset.root == path.parent
    assert dataset.id == "my-sensor-dataset"
    assert dataset.fps == 12.5
    assert dataset.classes == {
        "vehicle": {"id": 7, "evaluation": "target"},
        "walker": {"id": 12, "evaluation": "target"},
        "untracked": {"id": 99, "evaluation": "ignore"},
    }
    assert dataset.split == "val"
    assert dataset.sequence_names == ("drive-002", "drive-003")
    sequence = dataset.sequences[0]
    assert sequence.sequence_id == "drive-002"
    assert sequence.modalities["images"].paths == (path.parent / "sequences/recordings/drive-002/images",)
    assert sequence.modalities["calibration"].paths == (path.parent / "calibration/recordings.txt",)
    assert sequence.modalities["poses"].format == "camera-to-world-npy"
    assert sequence.modalities["detections_3d"].paths == (path.parent / "predictions/3d-val/recordings/drive-002",)
    assert sequence.modalities["detections_3d"].options == {"score_transform": "odds"}
    with pytest.raises(FrozenInstanceError):
        sequence.sequence_id = "changed"


def test_split_overrides_replace_whole_inputs_and_can_remove_ground_truth(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    training = load_dataset_inputs(path, split="train")
    all_sequences = load_dataset_inputs(path, split="all")
    validation = load_dataset_inputs(path, sequence_names=("drive-003", "drive-002"))
    test = load_dataset_inputs(path, split="test")

    assert training.sequence_names == ("drive-001",)
    assert training.sequences[0] == all_sequences.sequences[0]
    assert len(training.sequences[0].modalities["detections_3d"].paths) == 2
    assert validation.sequence_names == ("drive-002", "drive-003")
    assert "ground_truth" not in test.sequences[0].modalities
    assert load_dataset_config(path)["splits"]["test"]["has_ground_truth"] is False


def test_3d_annotations_are_optional_files_and_mark_the_split_as_annotated(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    specification = {
        "format": "kitti-tracking-labels",
        "path": "labels/{sequence}.txt",
        "options": {"class_map": {"Car": "vehicle"}, "ignore_classes": ["DontCare"]},
    }
    _change(path, ("modalities", "ground_truth_3d"), specification)
    annotations = tmp_path / "labels/drive-004.txt"
    annotations.parent.mkdir()
    annotations.touch()

    config = load_dataset_config(path)
    dataset = load_dataset_inputs(path, split="test")

    assert config["splits"]["test"]["has_ground_truth"] is True
    assert "ground_truth" not in dataset.sequences[0].modalities
    assert dataset.sequences[0].modalities["ground_truth_3d"].paths == (annotations,)
    assert dataset.sequences[0].modalities["ground_truth_3d"].options == specification["options"]
    annotations.unlink()
    annotations.mkdir()
    with pytest.raises(ConfigurationError, match="ground_truth_3d requires a file"):
        load_dataset_inputs(path, split="test")


@pytest.mark.parametrize(
    "specification,message",
    [
        ({"format": "kitti-detections", "path": "labels.txt"}, "format must be kitti-tracking-labels"),
        ({"format": "kitti-tracking-labels", "paths": ["first.txt", "second.txt"]}, "exactly one input path"),
        (
            {"format": "kitti-tracking-labels", "path": "labels.txt", "options": {"score_transform": "odds"}},
            "unsupported options: score_transform",
        ),
    ],
)
def test_3d_annotation_schema_requires_one_identity_bearing_source(
    tmp_path: Path, specification: dict[str, Any], message: str
) -> None:
    path = _fixture(tmp_path)
    _change(path, ("modalities", "ground_truth_3d"), specification)

    with pytest.raises(ConfigurationError, match=message):
        load_dataset_config(path)


def test_missing_unselected_inputs_do_not_block_a_selected_sequence_or_role(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    (tmp_path / "sequences/recordings/drive-003/poses.npy").unlink()

    assert len(load_dataset_inputs(path, sequence_names=("drive-002",)).sequences) == 1
    image_inputs = load_dataset_inputs(path, roles=("images", "ground_truth"))
    assert len(image_inputs.sequences) == 2
    assert set(image_inputs.sequences[0].modalities) == {"images", "ground_truth"}
    assert resolve_sensor_dataset_config_path(path) == path.resolve()
    with pytest.raises(ConfigurationError, match="Sequence drive-003 poses requires a file"):
        load_dataset_inputs(path)


def test_in_memory_normalized_config_respects_changed_split_selection(tmp_path: Path) -> None:
    config = load_dataset_config(_fixture(tmp_path))
    config["splits"]["val"]["sequences"] = ["drive-003"]

    dataset = resolve_dataset_inputs(config)

    assert dataset.sequence_names == ("drive-003",)
    assert dataset_modalities(config, "val")["detections_3d"]["options"] == {"score_transform": "odds"}


def test_automatic_discovery_extracts_sequence_names_from_a_directory_template(tmp_path: Path) -> None:
    config = load_dataset_config(_fixture(tmp_path))
    del config["splits"]["all"]["sequences"]
    (tmp_path / "sequences/recordings/.hidden/images").mkdir(parents=True)
    (tmp_path / "sequences/recordings/._drive-001/images").mkdir(parents=True)

    dataset = resolve_dataset_inputs(config, split="all")

    assert dataset.sequence_names == ("drive-001", "drive-002", "drive-003")
    assert len(resolve_dataset_inputs(config, split="all", sequence_names=("drive-003",)).sequences) == 1


def test_automatic_discovery_requires_a_sequence_placeholder(tmp_path: Path) -> None:
    config = load_dataset_config(_fixture(tmp_path))
    del config["splits"]["all"]["sequences"]
    config["modalities"]["images"]["paths"] = ["images"]

    with pytest.raises(ConfigurationError, match="requires.*sequence"):
        resolve_dataset_inputs(config, split="all")


def test_local_storage_root_and_explicit_data_root_use_the_same_relative_storage_path(tmp_path: Path) -> None:
    path = _fixture(tmp_path / "data" / "payload")
    config_path = tmp_path / "data/dataset.yaml"
    path.rename(config_path)
    _change(config_path, ("storage", "root"), "payload")
    config = load_dataset_config(config_path)

    assert load_dataset_inputs(config_path).root == tmp_path / "data/payload"
    assert resolve_dataset_storage_root(config, tmp_path / "override") == tmp_path / "override/payload"


def test_storage_root_symlink_cannot_escape_its_selected_base(tmp_path: Path) -> None:
    path = _fixture(tmp_path / "dataset")
    external = tmp_path / "external"
    external.mkdir()
    (path.parent / "linked-storage").symlink_to(external, target_is_directory=True)
    _change(path, ("storage", "root"), "linked-storage")

    with pytest.raises(ConfigurationError, match="must remain beneath the configured data root"):
        load_dataset_inputs(path)


def test_sensor_routing_uses_only_the_selected_split_modalities(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    _change(
        path,
        ("splits", "test", "modalities"),
        {"ground_truth": None, "detections_3d": None, "calibration": None, "poses": None},
    )

    assert resolve_sensor_dataset_config_path(path) == path.resolve()
    assert resolve_sensor_dataset_config_path(path, split="test") is None


def test_external_input_symlinks_resolve_and_broken_links_report_their_location(tmp_path: Path) -> None:
    path = _fixture(tmp_path / "dataset")
    external = tmp_path / "downloaded-images"
    external.mkdir()
    link = path.parent / "sequences/recordings/drive-002/images"
    link.rmdir()
    link.symlink_to(external, target_is_directory=True)

    assert load_dataset_inputs(path).sequences[0].modalities["images"].paths == (external,)
    external.rmdir()
    with pytest.raises(ConfigurationError, match="Sequence drive-002 images contains a broken link.*Restore"):
        load_dataset_inputs(path)


@pytest.mark.parametrize(
    "relative,description",
    [
        ("sequences/recordings/drive-002/images", "images requires a directory"),
        ("sequences/recordings/drive-002/ground_truth", "ground_truth requires a directory"),
        ("calibration/recordings.txt", "calibration requires a file"),
        ("sequences/recordings/drive-002/poses.npy", "poses requires a file"),
        ("predictions/image/recordings/drive-002.txt", "detections_2d requires a file"),
        ("predictions/3d-val/recordings/drive-002", "detections_3d requires a directory"),
    ],
)
def test_selected_inputs_report_missing_files_and_directories(tmp_path: Path, relative: str, description: str) -> None:
    path = _fixture(tmp_path)
    target = tmp_path / relative
    target.rmdir() if target.is_dir() else target.unlink()

    with pytest.raises(ConfigurationError, match=description):
        load_dataset_inputs(path)


@pytest.mark.parametrize(
    "template",
    [
        "../{sequence}",
        "/{sequence}",
        "C:/{sequence}",
        "a\\{sequence}",
        "{unknown}",
        "{sequence.__class__}",
        "{sequence[0]}",
        "{sequence!s}",
        "{sequence:>4}",
        "{sequence",
        None,
    ],
)
def test_templates_reject_escaping_paths_and_unsupported_formatting(tmp_path: Path, template: Any) -> None:
    path = _fixture(tmp_path)
    _change(path, ("modalities", "images", "path"), template)

    with pytest.raises(ConfigurationError):
        load_dataset_config(path)


@pytest.mark.parametrize(
    "keys,value,message",
    [
        (("format", "layout"), "kitti-fusion", "layout must be"),
        (("format", "box_type"), "obb", 'inputs require box_type "aabb"'),
        (("format", "layuot"), "sequence", "format has unknown keys"),
        (("storage", "rot"), ".", "storage has unknown keys"),
        (("default_split",), "missing", "default_split"),
        (("splits", "val", "sequences"), [2], "canonical directory names"),
        (("splits", "val", "sequences"), ["drive-002", "drive-002"], "duplicate"),
        (("splits", "val", "partition"), "../recordings", "canonical directory names"),
        (("splits", "val", "sequnces"), ["drive-002"], "has unknown keys: sequnces"),
        (("splits", "val", "has_ground_truth"), False, "must agree"),
        (("modalities", "poses", "paths"), ["poses.npy"], "exactly one of path or paths"),
        (("modalities", "detections_3d", "paths"), [], "non-empty list"),
        (
            ("modalities", "images"),
            {"format": "image-directory", "paths": ["first/{sequence}", "second/{sequence}"]},
            "requires exactly one input path",
        ),
        (("modalities", "poses", "format"), 1, "non-empty format"),
        (("modalities", "poses", "options"), "invalid", "options must be a mapping"),
        (("modalities", "poses", "format"), "kitti-p2", "format must be camera-to-world-npy"),
        (("modalities", "images", "options"), {"fps": 10}, "unsupported options"),
        (("modalities", "ground_truth", "options"), {}, "class_divisor must be an integer"),
        (("modalities", "detections_3d", "options"), {"score_transform": "log"}, "score_transform must be"),
        (("modalities", "unknown"), {}, "unknown modalities"),
        (("replay",), "replay.yaml", "unknown keys"),
    ],
)
def test_invalid_configuration_fails_before_sensor_routing(
    tmp_path: Path, keys: tuple[str, ...], value: Any, message: str
) -> None:
    path = _fixture(tmp_path)
    _change(path, keys, value)

    with pytest.raises(ConfigurationError, match=message):
        resolve_sensor_dataset_config_path(path)


@pytest.mark.parametrize("fps", [True, "10", 0, -1, None, float("nan"), float("inf")])
def test_frame_rate_requires_a_finite_positive_number(tmp_path: Path, fps: Any) -> None:
    path = _fixture(tmp_path)
    _change(path, ("fps",), fps)

    with pytest.raises(ConfigurationError, match="fps must be a finite positive number"):
        load_dataset_config(path)


def test_image_only_sequence_config_can_select_oriented_detector_geometry(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["format"]["box_type"] = "obb"
    payload["modalities"] = {"images": payload["modalities"]["images"]}
    for split in payload["splits"].values():
        split.pop("modalities", None)
    _write(path, payload)

    assert load_dataset_config(path)["box_type"] == "obb"


@pytest.mark.parametrize("name", ["../validation", "/validation", "a:b", 3])
def test_split_names_are_safe_output_directory_components(tmp_path: Path, name: Any) -> None:
    path = _fixture(tmp_path)
    _change(path, ("splits",), {name: {"partition": "recordings", "sequences": ["drive-002"]}})
    _change(path, ("default_split",), name)

    with pytest.raises(ConfigurationError, match="split names.*canonical directory names"):
        load_dataset_config(path)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"split": "missing"}, "has no split 'missing'"),
        ({"sequence_names": ("drive-001",)}, "absent from dataset split 'val'"),
        ({"sequence_names": ("drive-002", "drive-002")}, "duplicate"),
        ({"sequence_names": "drive-002"}, "list or tuple"),
    ],
)
def test_explicit_selection_cannot_expand_the_dataset(tmp_path: Path, kwargs: dict[str, Any], message: str) -> None:
    path = _fixture(tmp_path)

    with pytest.raises(ConfigurationError, match=message):
        load_dataset_inputs(path, **kwargs)


def test_catalog_id_precedes_a_same_named_folder_and_explicit_paths_select_local_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _fixture(tmp_path / "kitti-mots")
    monkeypatch.chdir(tmp_path)

    assert resolve_sensor_dataset_config_path("kitti-mots") is None
    assert resolve_sensor_dataset_config_path("mot17") is None
    for reference in ("./kitti-mots", Path("kitti-mots"), "kitti-mots/dataset.yaml"):
        assert resolve_dataset_config_path(reference) == path.resolve()
        assert resolve_sensor_dataset_config_path(reference) == path.resolve()


def test_missing_folder_manifest_and_invalid_yaml_report_the_configuration_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match=r"Dataset folder requires .*dataset.yaml"):
        resolve_sensor_dataset_config_path(tmp_path)
    path = tmp_path / "dataset.yaml"
    path.write_text("format: [", encoding="utf-8")

    with pytest.raises(ConfigurationError, match=r"Failed to parse configuration .*dataset.yaml"):
        resolve_sensor_dataset_config_path(path)


def test_input_resolution_imports_no_tensor_readers_or_engine_modules() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import boxmot.datasets.inputs; "
            "assert 'torch' not in sys.modules; "
            "assert not any(name.startswith('boxmot.engine') for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_tracked_kitti_bundle_profile_is_self_contained_and_preserves_prediction_selections(tmp_path: Path) -> None:
    """The root dataset example must load without downloaded payloads or replay manifests."""
    repository = Path(__file__).resolve().parents[3]
    authored = (repository / "kitti-mots/dataset.yaml").read_text(encoding="utf-8")
    relocated = tmp_path / "dataset.yaml"
    relocated.write_text(authored, encoding="utf-8")

    config = load_dataset_config(tmp_path)
    builtin = load_dataset_config(repository / "boxmot/configs/datasets/kitti-mots.yaml")

    assert sorted(path.name for path in tmp_path.iterdir()) == ["dataset.yaml"]
    assert config["layout"] == "sequence"
    assert config["box_type"] == "aabb"
    assert config["default_split"] == "val"
    assert config["fps"] == builtin["fps"] == 10.0
    assert config["classes"] == builtin["classes"]
    assert resolve_dataset_storage_root(config) == tmp_path
    for split in ("train", "val"):
        assert config["splits"][split]["sequences"] == builtin["splits"][split]["sequences"]
    training = set(builtin["splits"]["train"]["sequences"])
    validation = set(builtin["splits"]["val"]["sequences"])
    assert training.isdisjoint(validation)
    assert config["splits"]["fulltrain"]["sequences"] == sorted(training | validation)
    for split, car_source in (
        ("val", "pointgnn-car-t2"),
        ("train", "pointgnn-car-t3"),
        ("fulltrain", "pointgnn-car-t3"),
    ):
        assert config["splits"][split]["partition"] == "training"
        sources = dataset_modalities(config, split)
        assert sources["detections_3d"]["format"] == "kitti-detections"
        assert sources["detections_3d"]["paths"] == [
            f"predictions/{car_source}/{{partition}}/{{sequence}}",
            "predictions/pointgnn-pedestrian/{partition}/{sequence}",
        ]
        assert sources["detections_3d"]["options"] == {"score_transform": "odds", "ignore_classes": ["Cyclist"]}
        assert sources["detections_2d"]["paths"] == ["predictions/trackrcnn/{partition}/{sequence}.txt"]
        assert sources["ground_truth"]["options"] == dataset_modalities(builtin, split)["ground_truth"]["options"]
        assert all("manifest.yaml" not in path for source in sources.values() for path in source["paths"])
    assert "replay" not in yaml.safe_load(authored)
    assert "replay.yaml" not in authored
