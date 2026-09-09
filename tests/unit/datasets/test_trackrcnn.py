from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from boxmot.datasets.trackrcnn import TrackRcnnFrame, TrackRcnnSequence


def _row(frame: int, class_id: int = 1, counts: str = "0<") -> str:
    """Encode one full-frame 3x4 mask with the released embedding tail."""
    fields = [str(frame), "0", "0", "4", "3", "0.75", str(class_id), "3", "4", counts] + ["0"] * 128
    return " ".join(fields) + "\n"


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Provide only 2D detections and images, with no sensor-fusion assets."""
    detections = tmp_path / "trackrcnn_detections"
    images = tmp_path / "image_02"
    detections.mkdir()
    (images / "0002").mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (4, 3)).save(images / "0002" / f"{index:06d}.png")
    (detections / "0002.txt").write_text(_row(0, class_id=2) + _row(2, counts="<"))
    return detections, images


def test_trackrcnn_requires_only_images_and_2d_detector_files(tmp_path: Path) -> None:
    detections, images = _fixture(tmp_path)

    sequence = TrackRcnnSequence(detections, images, "0002")
    samples = list(sequence)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["image_02", "trackrcnn_detections"]
    assert len(sequence) == 3
    assert all(isinstance(sample, TrackRcnnFrame) for sample in samples)
    assert sequence.image_size == (3, 4)
    assert [sample.frame_index for sample in samples] == [0, 1, 2]
    assert [sample.detections.sample_id for sample in samples] == ["train:0002:0", "train:0002:1", "train:0002:2"]
    assert samples[0].image_path == images / "0002/000000.png"
    assert samples[0].detections.class_ids.tolist() == [2]
    assert samples[0].detections.geometry.values.tolist() == [[0, 0, 4, 3]]
    assert samples[0].detections.scores.tolist() == [0.75]
    assert samples[0].detections.embeddings is None
    assert samples[0].detections.masks.values.all()
    assert samples[0].detections.masks.values.dtype == torch.bool
    assert samples[0].detections.masks.values.is_contiguous()
    assert len(samples[1].detections) == 0
    assert samples[1].detections.masks.values.shape == (0, 3, 4)
    assert len(samples[2].detections) == 1
    assert not samples[2].detections.masks.values.any()


def test_trackrcnn_preserves_an_entirely_empty_prediction_sequence(tmp_path: Path) -> None:
    detections, images = _fixture(tmp_path)
    (detections / "0002.txt").write_text("")

    sequence = TrackRcnnSequence(detections, images, "0002")

    assert len(sequence) == 3
    for frame in sequence:
        assert len(frame.detections) == 0
        assert frame.detections.geometry.values.shape == (0, 4)
        assert frame.detections.masks.values.shape == (0, 3, 4)


def test_trackrcnn_decodes_only_requested_masks_and_never_rgb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    detections, images = _fixture(tmp_path)
    (detections / "0002.txt").write_text(_row(2, counts="invalid"))

    def reject_rgb(*args: object, **kwargs: object) -> None:
        raise AssertionError("2D detection reader must not decode RGB pixels")

    monkeypatch.setattr(Image.Image, "load", reject_rgb)
    sequence = TrackRcnnSequence(detections, images, "0002")

    assert len(sequence[0].detections) == 0
    with pytest.raises(ValueError, match="0002.txt:1: RLE"):
        sequence[2]


def test_trackrcnn_supports_slices_and_rejects_noninteger_indices(tmp_path: Path) -> None:
    detections, images = _fixture(tmp_path)
    sequence = TrackRcnnSequence(detections, images, "0002")

    assert [frame.frame_index for frame in sequence[1:]] == [1, 2]
    assert sequence[-1].frame_index == 2
    assert sequence[1].image_path == sequence.frame_paths[1]
    with pytest.raises(IndexError):
        sequence[3]
    for index in (True, 1.5, "1"):
        with pytest.raises(TypeError, match="integers or slices"):
            sequence[index]


@pytest.mark.parametrize("sequence_id", ["2", "../../0002", "０００２", 2])
def test_trackrcnn_rejects_ambiguous_sequence_names(tmp_path: Path, sequence_id: object) -> None:
    detections, images = _fixture(tmp_path)

    with pytest.raises(ValueError, match="exact four-digit"):
        TrackRcnnSequence(detections, images, sequence_id)


def test_trackrcnn_rejects_missing_prediction_file(tmp_path: Path) -> None:
    detections, images = _fixture(tmp_path)
    (detections / "0002.txt").unlink()

    with pytest.raises(FileNotFoundError, match="0002.txt"):
        TrackRcnnSequence(detections, images, "0002")


@pytest.mark.parametrize("defect", ["gap", "duplicate", "different_size"])
def test_trackrcnn_rejects_ambiguous_image_timeline(tmp_path: Path, defect: str) -> None:
    detections, images = _fixture(tmp_path)
    if defect == "gap":
        (images / "0002/000001.png").unlink()
    elif defect == "duplicate":
        Image.new("RGB", (4, 3)).save(images / "0002/1.png")
    else:
        Image.new("RGB", (5, 3)).save(images / "0002/000001.png")

    with pytest.raises(ValueError, match="contiguous|duplicate|dimensions"):
        TrackRcnnSequence(detections, images, "0002")


def test_trackrcnn_import_does_not_load_sensor_fusion_modules() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from boxmot.datasets.trackrcnn import TrackRcnnSequence; "
            "assert 'boxmot.datasets.kitti_fusion' not in sys.modules; "
            "assert 'boxmot.structures.spatial' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
