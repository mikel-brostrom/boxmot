"""Dataset facts, prediction selection, and portable KITTI input resolution."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml

from boxmot.datasets.kitti_fusion_config import load_kitti_fusion_dataset, resolve_kitti_fusion_config_path
from boxmot.utils.config import ConfigurationError


def _write(path: Path, payload: dict[str, Any]) -> Path:
    """Write a readable manifest under an existing or new fixture directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _fixture(root: Path) -> Path:
    """Create independent data and prediction manifests without decoding inputs."""

    names = ["0000", "0002", "0006"]
    for name in names:
        sequence = root / "sequences/training" / name
        for directory in ("images", "ground_truth"):
            (sequence / directory).mkdir(parents=True)
        for filename in ("calibration.txt", "poses.npy"):
            (sequence / filename).touch()
    variants = {
        "camera-results": ("trackrcnn", {1: "car", 2: "pedestrian"}, names),
        "car-validation": ("pointgnn", {1: "car"}, ["0002", "0006"]),
        "car-all": ("pointgnn", {1: "car"}, names),
        "pedestrian-results": ("pointgnn", {2: "pedestrian", 3: "cyclist"}, names),
    }
    for variant, (format_name, classes, sequences) in variants.items():
        relative = "payload/{partition}/{sequence}" + (".txt" if format_name == "trackrcnn" else "")
        directory = root / "predictions" / variant
        _write(
            directory / "manifest.yaml",
            {
                "format": format_name,
                "version": 1,
                "id": variant,
                "classes": classes,
                "path": relative,
                "sequences": {"training": sequences, "testing": ["0028"]},
                "provenance": {"source_directory": "download", "training": "not independently verified"},
            },
        )
        for name in sequences:
            target = directory / relative.format(partition="training", sequence=name)
            if format_name == "trackrcnn":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            else:
                target.mkdir(parents=True)
    _write(
        root / "replay.yaml",
        {
            "version": 1,
            "splits": {
                split: {
                    "image": "predictions/camera-results/manifest.yaml",
                    "car": f"predictions/{car}/manifest.yaml",
                    "pedestrian": "predictions/pedestrian-results/manifest.yaml",
                }
                for split, car in (("val", "car-validation"), ("train", "car-all"), ("fulltrain", "car-all"))
            },
        },
    )
    return _write(
        root / "dataset.yaml",
        {
            "format": "kitti-fusion",
            "version": 1,
            "id": "kitti-mots-fusion",
            "classes": {1: "car", 2: "pedestrian"},
            "default_split": "val",
            "replay": "replay.yaml",
            "sequence_layout": {
                name: f"sequences/{{partition}}/{{sequence}}/{filename}"
                for name, filename in (
                    ("images", "images"),
                    ("ground_truth", "ground_truth"),
                    ("calibration", "calibration.txt"),
                    ("poses", "poses.npy"),
                )
            },
            "splits": {
                "val": {"partition": "training", "sequences": ["0002", "0006"]},
                "train": {"partition": "training", "sequences": ["0000"]},
                "fulltrain": {"partition": "training", "sequences": names},
            },
        },
    )


def _change(path: Path, keys: tuple[str, ...], value: Any) -> None:
    """Change one authored value without coupling tests to loader internals."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    target = payload
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    _write(path, payload)


def test_folder_and_yaml_resolve_explicit_paths_independently_of_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _fixture(tmp_path / "dataset with spaces")
    monkeypatch.chdir(tmp_path)

    dataset = load_kitti_fusion_dataset(path.parent)

    assert dataset == load_kitti_fusion_dataset(path)
    assert dataset.config_path == resolve_kitti_fusion_config_path(path.parent) == path.resolve()
    assert dataset.id == "kitti-mots-fusion"
    assert dataset.split == "val"
    assert dataset.sequence_names == ("0002", "0006")
    assert dataset.replay_path == path.parent / "replay.yaml"
    assert dataset.predictions["car"] == path.parent / "predictions/car-validation/manifest.yaml"
    sequence = dataset.sequences[0]
    assert sequence.sequence_id == "0002"
    assert sequence.images == path.parent / "sequences/training/0002/images"
    assert sequence.ground_truth == path.parent / "sequences/training/0002/ground_truth"
    assert sequence.calibration == path.parent / "sequences/training/0002/calibration.txt"
    assert sequence.poses == path.parent / "sequences/training/0002/poses.npy"
    assert sequence.detections_2d == path.parent / "predictions/camera-results/payload/training/0002.txt"
    assert sequence.car_detections_3d == path.parent / "predictions/car-validation/payload/training/0002"
    assert sequence.pedestrian_detections_3d == path.parent / "predictions/pedestrian-results/payload/training/0002"
    with pytest.raises(FrozenInstanceError):
        sequence.images = tmp_path


def test_splits_select_prediction_variants_without_duplicating_dataset_sequences(tmp_path: Path) -> None:
    path = _fixture(tmp_path)

    training = load_kitti_fusion_dataset(path, split="train")
    fulltrain = load_kitti_fusion_dataset(path, split="fulltrain")
    validation = load_kitti_fusion_dataset(path, sequence_names=("0006", "0002"))

    assert training.sequence_names == ("0000",)
    assert training.predictions["car"] == tmp_path / "predictions/car-all/manifest.yaml"
    assert training.sequences[0] == fulltrain.sequences[0]
    assert validation.sequence_names == ("0002", "0006")
    assert validation.sequences[0].images == fulltrain.sequences[1].images


def test_unselected_sequence_payloads_are_not_required(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    (tmp_path / "sequences/training/0006/poses.npy").unlink()

    selected = load_kitti_fusion_dataset(path, sequence_names=("0002",))

    assert len(selected.sequences) == 1
    with pytest.raises(ConfigurationError, match="Sequence 0006 poses requires a file"):
        load_kitti_fusion_dataset(path)


def test_routing_validates_dataset_facts_without_requiring_predictions(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    (tmp_path / "replay.yaml").unlink()

    assert resolve_kitti_fusion_config_path(path) == path.resolve()
    with pytest.raises(ConfigurationError, match="replay manifest requires a file"):
        load_kitti_fusion_dataset(path)


def test_external_image_links_are_supported_and_broken_links_are_actionable(tmp_path: Path) -> None:
    path = _fixture(tmp_path / "dataset")
    external = tmp_path / "downloaded-images"
    external.mkdir()
    link = path.parent / "sequences/training/0002/images"
    link.rmdir()
    link.symlink_to(external, target_is_directory=True)

    assert load_kitti_fusion_dataset(path).sequences[0].images == external
    external.rmdir()
    with pytest.raises(ConfigurationError, match="Sequence 0002 images contains a broken link.*Restore"):
        load_kitti_fusion_dataset(path)


@pytest.mark.parametrize(
    "relative,description",
    [
        ("sequences/training/0002/images", "images requires a directory"),
        ("sequences/training/0002/ground_truth", "ground_truth requires a directory"),
        ("sequences/training/0002/calibration.txt", "calibration requires a file"),
        ("sequences/training/0002/poses.npy", "poses requires a file"),
        ("predictions/camera-results/payload/training/0002.txt", "image predictions requires a file"),
        ("predictions/car-validation/payload/training/0002", "car predictions requires a directory"),
        ("predictions/pedestrian-results/payload/training/0002", "pedestrian predictions requires a directory"),
    ],
)
def test_selected_inputs_report_the_exact_missing_file_or_directory(
    tmp_path: Path, relative: str, description: str
) -> None:
    path = _fixture(tmp_path)
    target = tmp_path / relative
    target.rmdir() if target.is_dir() else target.unlink()

    with pytest.raises(ConfigurationError, match=description):
        load_kitti_fusion_dataset(path)


@pytest.mark.parametrize(
    "template",
    ["../{sequence}", "/{sequence}", "C:/{sequence}", "a\\{sequence}", "{unknown}/{sequence}",
     "{sequence.__class__}", "{sequence[0]}", "{sequence!s}", "{sequence:>4}", "{sequence", "fixed", None],
)
@pytest.mark.parametrize("prediction", [False, True])
def test_sequence_templates_reject_escaping_paths_and_unsupported_formatting(
    tmp_path: Path, template: Any, prediction: bool
) -> None:
    path = _fixture(tmp_path)
    target = tmp_path / "predictions/car-validation/manifest.yaml" if prediction else path
    keys = ("path",) if prediction else ("sequence_layout", "images")
    _change(target, keys, template)

    with pytest.raises(ConfigurationError):
        load_kitti_fusion_dataset(path)


@pytest.mark.parametrize("location", ["dataset", "replay"])
def test_manifest_references_cannot_escape_their_parent(tmp_path: Path, location: str) -> None:
    path = _fixture(tmp_path)
    if location == "dataset":
        _change(path, ("replay",), "../replay.yaml")
    else:
        _change(tmp_path / "replay.yaml", ("splits", "val", "car"), "../manifest.yaml")

    with pytest.raises(ConfigurationError, match="relative path without"):
        load_kitti_fusion_dataset(path)


@pytest.mark.parametrize(
    "keys,value,message",
    [
        (("version",), True, "version must be integer 1"),
        (("version",), 2, "version must be integer 1"),
        (("id",), 42, "id must be"),
        (("classes",), {0: "car", 1: "pedestrian"}, "classes must map native IDs"),
        (("classes",), {True: "car", 2: "pedestrian"}, "classes must map native IDs"),
        (("classes",), {1: "car", 2: "pedestrian", 4: None}, "classes must map native IDs"),
        (("default_split",), "test", "default_split must name"),
        (("splits", "val", "sequences"), [2], "canonical directory names"),
        (("splits", "val", "sequences"), ["0002", "0002"], "duplicate"),
        (("splits", "val", "sequences"), ["002"], "four-digit"),
        (("splits", "val", "sequences"), ["０００２"], "four-digit"),
        (("splits", "val", "sequences"), ["0000"], "outside the official KITTI MOTS val split"),
        (("splits", "val", "partition"), "testing", "partition must be training"),
        (("predictions",), {}, "unknown keys"),
    ],
)
def test_invalid_dataset_facts_fail_before_routing(
    tmp_path: Path, keys: tuple[str, ...], value: Any, message: str
) -> None:
    path = _fixture(tmp_path)
    _change(path, keys, value)

    with pytest.raises(ConfigurationError, match=message):
        resolve_kitti_fusion_config_path(path)


@pytest.mark.parametrize(
    "variant,keys,value,message",
    [
        ("car-validation", ("format",), "trackrcnn", "format must be pointgnn"),
        ("camera-results", ("format",), "pointgnn", "format must be trackrcnn"),
        ("car-validation", ("classes",), {2: "pedestrian"}, "classes must map native IDs"),
        ("car-validation", ("classes",), {1: "car", 2: "pedestrian"}, "classes must map native IDs"),
        ("camera-results", ("classes",), {1: "car", 2: "pedestrian", 3: "cyclist"}, "classes must map native IDs"),
        ("car-validation", ("sequences", "training"), ["0006"], "does not cover training sequence.*0002"),
        ("car-validation", ("sequences", "testing"), [28], "canonical directory names"),
        ("car-validation", ("version",), "1", "version must be integer 1"),
        ("car-validation", ("provenance",), "unknown", "provenance must be a mapping"),
    ],
)
def test_prediction_formats_native_classes_and_coverage_are_validated(
    tmp_path: Path, variant: str, keys: tuple[str, ...], value: Any, message: str
) -> None:
    path = _fixture(tmp_path)
    _change(tmp_path / "predictions" / variant / "manifest.yaml", keys, value)

    with pytest.raises(ConfigurationError, match=message):
        load_kitti_fusion_dataset(path)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"split": "test"}, "has no split 'test'"),
        ({"sequence_names": ("0000",)}, "absent from dataset split 'val'"),
        ({"sequence_names": ("0008",)}, "absent from dataset split 'val'"),
        ({"sequence_names": ("0002", "0002")}, "duplicate"),
        ({"sequence_names": "0002"}, "list or tuple"),
    ],
)
def test_explicit_selection_cannot_silently_expand_the_dataset(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    path = _fixture(tmp_path)

    with pytest.raises(ConfigurationError, match=message):
        load_kitti_fusion_dataset(path, **kwargs)


def test_builtin_id_keeps_catalog_precedence_over_a_same_named_local_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _fixture(tmp_path / "kitti-mots")
    monkeypatch.chdir(tmp_path)

    assert resolve_kitti_fusion_config_path("kitti-mots") is None
    assert resolve_kitti_fusion_config_path("./kitti-mots") == path.resolve()
    assert resolve_kitti_fusion_config_path(Path("kitti-mots")) == path.resolve()
    assert resolve_kitti_fusion_config_path("kitti-mots/dataset.yaml") == path.resolve()


def test_ordinary_yaml_stays_with_image_workflows_and_incorrect_folder_format_is_actionable(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    _change(path, ("format",), {"layout": "kitti-mots"})

    assert resolve_kitti_fusion_config_path(path) is None
    assert resolve_kitti_fusion_config_path("mot17") is None
    with pytest.raises(ConfigurationError, match="must declare.*format: kitti-fusion"):
        resolve_kitti_fusion_config_path(tmp_path)


def test_missing_manifest_and_invalid_yaml_report_the_configuration_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match=r"Dataset folder requires .*dataset.yaml"):
        resolve_kitti_fusion_config_path(tmp_path)
    path = tmp_path / "dataset.yaml"
    path.write_text("format: kitti-fusion\nclasses: [", encoding="utf-8")

    with pytest.raises(ConfigurationError, match=r"Failed to parse configuration .*dataset.yaml"):
        resolve_kitti_fusion_config_path(path)


def test_config_import_does_not_load_tensor_readers_or_engine_modules() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import boxmot.datasets.kitti_fusion_config; "
            "assert 'torch' not in sys.modules; "
            "assert not any(name.startswith('boxmot.engine') for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
