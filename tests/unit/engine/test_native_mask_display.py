"""Native segmentation remains visible through live tracking output sinks."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxmot import ByteTrack, ByteTrackConfig
from boxmot.detectors import DetectorCapabilities
from boxmot.engine.tracking.workflow import run_track
from boxmot.structures import Boxes, Detections, Frame, MaskBatch
from tests.unit.engine.eval.test_visualization import capture_rendered_images


class _Source:
    """Yield fixed frames while retaining them for mutation checks."""

    def __init__(self, frames: Sequence[Frame]) -> None:
        self.frames = tuple(frames)
        self.closed = False

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)

    def close(self) -> None:
        self.closed = True


class _NativeMaskDetector:
    """Produce a tracked instance after an untracked detection in each batch."""

    capabilities = DetectorCapabilities(provides_masks=True)

    def __init__(self) -> None:
        values = torch.zeros((2, 96, 128), dtype=torch.bool)
        values[0, 30:80, 80:105] = True
        values[1, 30:80, 20:50] = True
        self.masks = MaskBatch(values)

    def predict(self, frames: Sequence[Frame]) -> list[Detections]:
        return [
            Detections(
                geometry=Boxes(torch.tensor([[75, 20, 110, 90], [10, 20, 60, 90]], dtype=torch.float32)),
                scores=torch.tensor([0.2, 0.95], dtype=torch.float32),
                class_ids=torch.tensor([0, 0], dtype=torch.int64),
                sample_id=frame.sample_id,
                masks=self.masks,
            )
            for frame in frames
        ]


@pytest.mark.parametrize("save", [False, True], ids=["display", "display-and-video"])
@pytest.mark.parametrize("background", [0, 50, 150, 255])
def test_native_mask_fill_is_visible_without_an_explicit_segmentor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, save: bool, background: int
) -> None:
    """Draw the first identity's mask distinctly, preserving caller-owned pixels."""
    captured = capture_rendered_images(monkeypatch)
    frames = tuple(
        Frame(
            torch.full((3, 96, 128), background, dtype=torch.uint8),
            sample_id=f"camera/{index}",
            sequence_id="camera",
            frame_index=index,
        )
        for index in range(2)
    )
    originals = tuple(frame.image.clone() for frame in frames)
    source = _Source(frames)
    detector = _NativeMaskDetector()
    original_masks = detector.masks.values.clone()
    run = run_track(
        SimpleNamespace(geometry="aabb", show=True, save=save, project=tmp_path, name="native-masks"),
        detector=detector,
        tracker=ByteTrack(config=ByteTrackConfig(min_hits=1)),
        source=source,
    )

    assert source.closed
    assert run.summary.frames == run.summary.track_rows == 2
    assert run.summary.detections == 4
    assert run.summary.unique_track_ids == 1
    assert len(captured.shown) == 2
    for image in captured.shown:
        # Sample the mask interior, away from the bounding box and its label.
        assert np.ptp(image[55, 30].astype(int)) >= 40
        assert np.ptp(image[55, 10].astype(int)) >= 40
        assert not np.array_equal(image[55, 30], image[55, 10])
        np.testing.assert_array_equal(image[55, 65], [background] * 3)
        # An untracked detection's mask must not acquire the tracked identity.
        np.testing.assert_array_equal(image[55, 90], [background] * 3)
    np.testing.assert_array_equal(captured.shown[0], captured.shown[1])
    for frame, original in zip(frames, originals, strict=True):
        torch.testing.assert_close(frame.image, original)
    torch.testing.assert_close(detector.masks.values, original_masks)
    if save:
        assert len(captured.writers) == 1
        assert captured.writers[0].closed
        assert len(captured.writers[0].images) == 2
        for displayed, encoded in zip(captured.shown, captured.writers[0].images, strict=True):
            np.testing.assert_array_equal(displayed, encoded)
    else:
        assert captured.writers == []
