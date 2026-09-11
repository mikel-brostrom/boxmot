"""Declared saved 2D inputs retain native frame identity and omit selected channels."""

from pathlib import Path

import pytest
from PIL import Image

from boxmot.datasets.inputs import ModalityInput, SequenceInputs
from boxmot.datasets.sequence import DetectionSequence
from tests.unit.datasets.test_detection_readers import _fixture, _row


def _inputs(tmp_path: Path, *, options: dict | None = None) -> SequenceInputs:
    detections, images = _fixture(tmp_path)
    detections.write_text(_row(0, class_id=1) + _row(0, class_id=2) + _row(2, class_id=2))
    return SequenceInputs(
        "drive",
        {
            "images": ModalityInput("image-directory", (images,), {}),
            "detections_2d": ModalityInput("trackrcnn", (detections,), options or {}),
            "ground_truth": ModalityInput("kitti-tracking-labels", (tmp_path / "never-read-gt.txt",), {}),
        },
    )


@pytest.mark.parametrize("load_masks", (False, True))
def test_detection_sequence_selects_targets_without_changing_timing_or_reading_gt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_masks: bool
) -> None:
    inputs = _inputs(tmp_path, options={"load_masks": load_masks})

    def reject_rgb(*args: object, **kwargs: object) -> None:
        raise AssertionError("Saved detections do not decode RGB pixels")

    monkeypatch.setattr(Image.Image, "load", reject_rgb)
    sequence = DetectionSequence(
        inputs,
        classes={"car": {"id": 1, "evaluation": "target"}, "pedestrian": {"id": 2, "evaluation": "ignore"}},
        fps=25,
        split="val",
    )
    samples = list(sequence)

    assert [len(sample.detections) for sample in samples] == [1, 0, 0]
    assert [sample.timestamp_s for sample in samples] == [0, 0.04, 0.08]
    assert [sample.frame_index for sample in samples] == [0, 1, 2]
    assert [sample.detections.sample_id for sample in samples] == [f"val:drive:{index}" for index in range(3)]
    assert [sample.image_path for sample in samples] == list(sequence.frame_paths)
    assert all(sample.image_size == sequence.image_size == (3, 4) for sample in samples)
    assert samples[0].detections.class_ids.tolist() == [1]
    assert all(sample.detections.embeddings is None for sample in samples)
    if load_masks:
        assert samples[0].detections.masks.values.shape == (1, 3, 4)
        assert samples[2].detections.masks.values.shape == (0, 3, 4)
    else:
        assert all(sample.detections.masks is None for sample in samples)
    assert [sample.frame_index for sample in sequence[1:]] == [1, 2]
    assert sequence[-1].frame_index == 2
    with pytest.raises(TypeError, match="integers or slices"):
        sequence[True]


@pytest.mark.parametrize(
    "options,match", (({"unknown": True}, "Unsupported trackrcnn"), ({"load_masks": 0}, "boolean"))
)
def test_detection_sequence_validates_options_before_reader_dispatch(tmp_path: Path, options: dict, match: str) -> None:
    inputs = _inputs(tmp_path, options=options)
    with pytest.raises(ValueError, match=match):
        DetectionSequence(inputs, classes={"car": {"id": 1}}, fps=10)


def test_detection_sequence_rejects_unconsumed_sensor_modalities(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs.modalities["poses"] = ModalityInput("camera-to-world-npy", (tmp_path / "poses.npy",), {})
    with pytest.raises(ValueError, match="Unsupported 2D detection sequence modalities: poses"):
        DetectionSequence(inputs, classes={"car": {"id": 1}}, fps=10)
