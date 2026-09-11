from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from boxmot.datasets import ImageDataset, ImageSample
from boxmot.datasets.readers.frames import numeric_frame_paths

_CLASSES = {"car": {"id": 1, "evaluation": "target"}, "pedestrian": {"id": 2, "evaluation": "target"}}
_INSTANCE_OPTIONS = {"class_divisor": 1000, "background_id": 0, "ignore_ids": [10000]}


def _dataset(image_root: Path, instances_root: Path | None = None, **options: object) -> ImageDataset:
    """Select the fixture's explicit classes, timing and annotation encoding."""
    return ImageDataset(
        image_root, instances_root, classes=_CLASSES, fps=10.0, instance_options=_INSTANCE_OPTIONS, **options
    )


def _write_frame(
    root: Path,
    *,
    sequence: str = "0000",
    name: str = "000000.png",
    labels: np.ndarray | None = None,
    image_size: tuple[int, int] = (4, 6),
) -> tuple[Path, Path]:
    """Create a tiny RGB source and optional native instance PNG."""

    image_root = root / "training" / "image_02"
    instances_root = root / "instances"
    image_path = image_root / sequence / name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((*image_size, 3), (10, 20, 30), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    if labels is not None:
        annotation_path = instances_root / sequence / name
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(annotation_path), labels)
    return image_root, instances_root


def test_image_dataset_decodes_native_tracks_masks_ignore_and_rgb(tmp_path: Path) -> None:
    labels = np.zeros((4, 6), dtype=np.uint16)
    labels[0, 0] = 10000
    labels[1:3, 2:4] = 1000
    labels[3, 5] = 2000
    images, instances = _write_frame(tmp_path, labels=labels)

    sample = _dataset(images, instances)[0]

    assert isinstance(sample, ImageSample)
    assert sample.frame.sample_id == "train:0000:0"
    assert sample.frame.sequence_id == "0000"
    assert sample.frame.frame_index == 0
    assert sample.frame.timestamp_s == 0.0
    assert sample.frame.image_size == (4, 6)
    assert sample.frame.image.dtype == torch.uint8
    assert sample.frame.image.is_contiguous()
    assert sample.frame.image[:, 0, 0].tolist() == [30, 20, 10]
    assert sample.frame.source_uri == (images / "0000" / "000000.png").as_uri()
    tracks = sample.ground_truth
    assert tracks is not None
    assert tracks.sample_id == sample.frame.sample_id
    assert tracks.track_ids.tolist() == [1000, 2000]
    assert tracks.class_ids.tolist() == [1, 2]
    assert tracks.scores.tolist() == [1.0, 1.0]
    assert tracks.detection_indices.tolist() == [-1, -1]
    assert tracks.geometry.values.tolist() == [[2.0, 1.0, 4.0, 3.0], [5.0, 3.0, 6.0, 4.0]]
    assert tracks.masks is not None
    assert tracks.masks.values.dtype == torch.bool
    assert tracks.masks.values.shape == (2, 4, 6)
    assert tracks.masks.values.is_contiguous()
    for mask, track_id in zip(tracks.masks.values, tracks.track_ids.tolist(), strict=True):
        assert torch.equal(mask, torch.from_numpy(labels == track_id))
    assert sample.ignore_mask is not None
    assert sample.ignore_mask.dtype == torch.bool
    assert torch.equal(sample.ignore_mask, torch.from_numpy(labels == 10000))


@pytest.mark.parametrize("label", [0, 10000])
def test_image_dataset_empty_annotated_frames_preserve_mask_shape(tmp_path: Path, label: int) -> None:
    labels = np.full((4, 6), label, dtype=np.uint16)
    images, instances = _write_frame(tmp_path, labels=labels)

    sample = _dataset(images, instances)[0]

    assert sample.ground_truth is not None
    assert len(sample.ground_truth) == 0
    assert sample.ground_truth.geometry.values.shape == (0, 4)
    assert sample.ground_truth.masks is not None
    assert sample.ground_truth.masks.values.shape == (0, 4, 6)
    assert sample.ignore_mask is not None
    assert bool(sample.ignore_mask.all()) == (label == 10000)


@pytest.mark.parametrize("directory", ["frames#archive", "frames?archive", "frames%20archive"])
def test_image_dataset_preserves_special_characters_in_local_paths(tmp_path: Path, directory: str) -> None:
    images, instances = _write_frame(tmp_path / directory, labels=np.zeros((4, 6), dtype=np.uint16))

    sample = _dataset(images, instances)[0]

    assert sample.frame.image_size == (4, 6)
    assert sample.frame.source_uri == (images / "0000" / "000000.png").as_uri()
    assert sample.ground_truth is not None


def test_image_dataset_preserves_numeric_frame_order_and_sparse_native_indices(tmp_path: Path) -> None:
    for sequence, name in [("0001", "1.png"), ("0000", "10.png"), ("0000", "2.png"), ("0000", "000000.png")]:
        images, _ = _write_frame(tmp_path, sequence=sequence, name=name)
    (images / "0000" / "._000000.png").write_bytes(b"AppleDouble metadata")
    (images / "._0000").mkdir()

    dataset = _dataset(images, split="test")

    assert dataset.sequence_ids == ("0000", "0001")
    assert len(dataset) == 4
    assert [sample.frame.sample_id for sample in dataset] == [
        "test:0000:0",
        "test:0000:2",
        "test:0000:10",
        "test:0001:1",
    ]
    assert [sample.frame.timestamp_s for sample in dataset] == [0.0, 0.2, 1.0, 0.1]
    assert dataset[-1].frame.sequence_id == "0001"
    assert [sample.frame.frame_index for sample in dataset[1:3]] == [2, 10]
    assert all(sample.ground_truth is None and sample.ignore_mask is None for sample in dataset)
    with pytest.raises(IndexError):
        dataset[4]


def test_image_dataset_decodes_only_requested_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images, instances = _write_frame(tmp_path, labels=np.zeros((4, 6), dtype=np.uint16))
    _write_frame(tmp_path, name="000001.png", labels=np.zeros((4, 6), dtype=np.uint16))
    calls: list[str] = []
    original = cv2.imread

    def record_read(path: str, flags: int) -> np.ndarray:
        calls.append(path)
        return original(path, flags)

    monkeypatch.setattr(cv2, "imread", record_read)
    dataset = _dataset(images, instances)

    assert len(dataset) == 2
    assert calls == []
    assert dataset[1].frame.frame_index == 1
    assert calls == [str(images / "0000" / "000001.png"), str(instances / "0000" / "000001.png")]


def test_image_dataset_sequence_filter_is_exact_and_deterministic(tmp_path: Path) -> None:
    for sequence in ("0002", "0000", "0001"):
        images, _ = _write_frame(tmp_path, sequence=sequence)

    dataset = _dataset(images, sequence_ids=["0002", "0000"])

    assert dataset.sequence_ids == ("0000", "0002")
    assert [sample.frame.sequence_id for sample in dataset] == ["0000", "0002"]
    with pytest.raises(ValueError, match="requested sequences"):
        _dataset(images, sequence_ids=["0"])


@pytest.mark.parametrize("sequence_ids", [[], ["../0000"], ["/0000"], ["0000/../0000"], [".."], [""], ["0000", "0000"]])
def test_image_dataset_rejects_unsafe_or_duplicate_sequence_filters(tmp_path: Path, sequence_ids: list[str]) -> None:
    images, _ = _write_frame(tmp_path)
    with pytest.raises(ValueError, match="sequence_ids"):
        _dataset(images, sequence_ids=sequence_ids)


def test_image_dataset_rejects_string_sequence_filter(tmp_path: Path) -> None:
    images, _ = _write_frame(tmp_path)
    with pytest.raises(TypeError, match="sequence_ids"):
        _dataset(images, sequence_ids="0000")


def test_image_dataset_rejects_missing_paired_annotation(tmp_path: Path) -> None:
    images, instances = _write_frame(tmp_path, labels=np.zeros((4, 6), dtype=np.uint16))
    _write_frame(tmp_path, name="000001.png")
    with pytest.raises(FileNotFoundError, match="Missing paired.*000001.png"):
        _dataset(images, instances)


def test_image_dataset_requires_matching_annotation_filename(tmp_path: Path) -> None:
    images, instances = _write_frame(tmp_path, labels=np.zeros((4, 6), dtype=np.uint16))
    (instances / "0000" / "000000.png").rename(instances / "0000" / "0.png")
    with pytest.raises(FileNotFoundError, match="Missing paired"):
        _dataset(images, instances)


@pytest.mark.parametrize("label", [1, 999, 3000, 10001, 65535])
def test_image_dataset_rejects_unknown_instance_labels(tmp_path: Path, label: int) -> None:
    images, instances = _write_frame(tmp_path, labels=np.full((4, 6), label, dtype=np.uint16))
    dataset = _dataset(images, instances)
    with pytest.raises(ValueError, match="unsupported labels"):
        dataset[0]


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (np.zeros((4, 6), dtype=np.uint8), "single-channel uint16"),
        (np.zeros((4, 6, 3), dtype=np.uint16), "single-channel uint16"),
        (np.zeros((3, 6), dtype=np.uint16), "dimensions"),
    ],
)
def test_image_dataset_rejects_invalid_annotation_images(tmp_path: Path, labels: np.ndarray, message: str) -> None:
    images, instances = _write_frame(tmp_path, labels=labels)
    dataset = _dataset(images, instances)
    with pytest.raises(ValueError, match=message):
        dataset[0]


def test_image_dataset_rejects_undecodable_annotation(tmp_path: Path) -> None:
    images, instances = _write_frame(tmp_path, labels=np.zeros((4, 6), dtype=np.uint16))
    (instances / "0000" / "000000.png").write_bytes(b"not a PNG")
    with pytest.raises(ValueError, match="Unable to decode"):
        _dataset(images, instances)[0]


@pytest.mark.parametrize("name", ["000000.png", "frame.png", "-1.png"])
def test_image_dataset_rejects_ambiguous_or_invalid_frame_names(tmp_path: Path, name: str) -> None:
    images, _ = _write_frame(tmp_path, name="0.png")
    _write_frame(tmp_path, name=name)
    with pytest.raises(ValueError, match="duplicate frame index|numeric stems"):
        numeric_frame_paths(images / "0000")


def test_image_dataset_rejects_empty_sequence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no image frames"):
        numeric_frame_paths(tmp_path)


def test_image_dataset_discovery_import_does_not_load_tensor_or_engine_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from boxmot.datasets import ImageDataset, ImageSample; "
            "from boxmot.datasets.readers.frames import numeric_frame_paths; "
            "assert 'torch' not in sys.modules; assert 'cv2' not in sys.modules; assert 'PIL' not in sys.modules; "
            "assert not any(name.startswith('boxmot.engine') for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_image_dataset_uses_configured_classes_and_annotation_encoding(tmp_path: Path) -> None:
    """Ignored encoded classes and explicit ignored labels remain separate from tracks."""
    labels = np.full((4, 6), 65535, dtype=np.uint16)
    labels[0, 0] = 0
    labels[1:3, 2:4] = 701
    labels[3, 5] = 902
    images, instances = _write_frame(tmp_path, labels=labels, name="000002.png")
    classes = {"parcel": {"id": 7, "evaluation": "target"}, "excluded": {"id": 9, "evaluation": "ignore"}}

    sample = ImageDataset(
        images,
        instances,
        classes=classes,
        fps=25,
        split="custom",
        instance_options={"class_divisor": 100, "background_id": 65535, "ignore_ids": [0]},
    )[0]

    assert sample.frame.timestamp_s == 0.08
    assert sample.frame.sample_id == "custom:0000:2"
    assert sample.ground_truth.track_ids.tolist() == [701]
    assert sample.ground_truth.class_ids.tolist() == [7]
    assert torch.equal(sample.ignore_mask, torch.from_numpy((labels == 0) | (labels == 902)))


def test_image_dataset_from_inputs_uses_declared_roots_timing_and_mask_options(tmp_path: Path) -> None:
    """Config-backed inputs use the same readers without imposing raw folder names."""
    from boxmot.datasets.inputs import DatasetInputs, ModalityInput, SequenceInputs

    images, instances = _write_frame(tmp_path, labels=np.zeros((4, 6), dtype=np.uint16), name="000002.png")
    inputs = DatasetInputs(
        config_path=None,
        id="custom-images",
        root=tmp_path,
        split="val",
        sequence_names=("camera-a",),
        sequences=(
            SequenceInputs(
                "camera-a",
                {
                    "images": ModalityInput("image-directory", (images / "0000",), {}),
                    "ground_truth": ModalityInput("instance-png", (instances / "0000",), _INSTANCE_OPTIONS),
                },
            ),
        ),
        classes=_CLASSES,
        fps=20,
    )

    sample = ImageDataset.from_inputs(inputs)[0]

    assert sample.frame.sample_id == "val:camera-a:2"
    assert sample.frame.timestamp_s == 0.1
    assert sample.ground_truth is not None


def test_image_dataset_requires_annotation_encoding_options(tmp_path: Path) -> None:
    images, instances = _write_frame(tmp_path, labels=np.zeros((4, 6), dtype=np.uint16))

    with pytest.raises(ValueError, match="class_divisor"):
        ImageDataset(images, instances, classes=_CLASSES, fps=10)
