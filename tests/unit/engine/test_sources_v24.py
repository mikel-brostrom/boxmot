from __future__ import annotations

import cv2
import numpy as np
import torch

from boxmot.engine.tracking.sources import DirectorySource, ImageSource, frame_from_bgr


def test_frame_from_bgr_converts_to_canonical_rgb() -> None:
    bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    bgr[0, 0] = (1, 2, 3)

    frame = frame_from_bgr(
        bgr,
        sample_id="sample",
        sequence_id="sequence",
        frame_index=0,
        timestamp_s=0.0,
        source_uri="memory:test",
    )

    assert frame.image.shape == (3, 2, 3)
    assert frame.image.dtype == torch.uint8
    assert frame.image.is_contiguous()
    assert frame.image[:, 0, 0].tolist() == [3, 2, 1]


def test_image_source_assigns_identity(tmp_path) -> None:
    path = tmp_path / "frame.png"
    assert cv2.imwrite(str(path), np.full((4, 5, 3), 7, dtype=np.uint8))

    frames = list(ImageSource(path))

    assert len(frames) == 1
    assert frames[0].sample_id == "frame.png"
    assert frames[0].frame_index == 0
    assert frames[0].sequence_id
    assert frames[0].source_uri == path.resolve().as_uri()


def test_directory_source_is_sorted_and_applies_stride(tmp_path) -> None:
    for name, value in (("003.png", 3), ("001.png", 1), ("002.png", 2)):
        assert cv2.imwrite(str(tmp_path / name), np.full((2, 2, 3), value, dtype=np.uint8))

    frames = list(DirectorySource(tmp_path, stride=2))

    assert [frame.sample_id for frame in frames] == ["001.png", "003.png"]
    assert [frame.frame_index for frame in frames] == [0, 1]
    assert len({frame.sequence_id for frame in frames}) == 1
