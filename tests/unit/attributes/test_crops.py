import math

import cv2
import numpy as np
import pytest
import torch

from boxmot.attributes.config import AttributeConfig
from boxmot.attributes.crops import build_storyboard, encode_track_crop
from boxmot.attributes.models import AttributeSample
from boxmot.structures import Boxes, Frame, MaskBatch, OrientedBoxes, Tracks


def _config(**kwargs):
    return AttributeConfig("vision", "Describe", {"type": "object"}, **kwargs)


def _tracks(geometry, *, detection_index=0, masks=None):
    return Tracks(
        geometry,
        torch.tensor([4], dtype=torch.int64),
        torch.tensor([0.9], dtype=torch.float32),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([detection_index], dtype=torch.int64),
        "sample",
        masks,
    )


def _frame():
    image = torch.zeros((3, 20, 40), dtype=torch.uint8)
    image[0] = 240
    return Frame(image, "sample")


def _decode(encoded):
    return cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)


def test_crop_clips_half_open_bounds_preserves_color_and_limits_size():
    frame = _frame()
    tracks = _tracks(Boxes(torch.tensor([[-3, -2, 50, 30]], dtype=torch.float32)))
    crop = _decode(encode_track_crop(frame, tracks, 0, _config(max_crop_side=16)))
    assert crop.shape == (8, 16, 3)
    assert crop[4, 8, 2] > 230
    assert crop[4, 8, 0] < 10
    assert torch.count_nonzero(frame.image[1:]) == 0


def test_obb_uses_enclosing_bounds_without_rotation():
    tracks = _tracks(OrientedBoxes(torch.tensor([[10, 10, 8, 4, math.pi / 2]], dtype=torch.float32)))
    crop = _decode(encode_track_crop(_frame(), tracks, 0, _config()))
    assert crop.shape[:2] == (8, 4)


def test_prediction_only_skips_but_nonempty_propagated_mask_is_current_evidence():
    geometry = Boxes(torch.tensor([[1, 1, 10, 15]], dtype=torch.float32))
    tracks = _tracks(geometry, detection_index=-1)
    assert encode_track_crop(_frame(), tracks, 0, _config()) is None
    empty = MaskBatch(torch.zeros((1, 20, 40), dtype=torch.bool))
    assert encode_track_crop(_frame(), _tracks(geometry, detection_index=-1, masks=empty), 0, _config()) is None
    mask = empty.values.clone()
    mask[0, 3:10, 3:8] = True
    assert encode_track_crop(_frame(), _tracks(geometry, detection_index=-1, masks=MaskBatch(mask)), 0, _config())


@pytest.mark.parametrize("background,subject", [(255, 0), (0, 255)])
def test_propagated_crop_uses_current_mask_bounds_instead_of_stale_box(background, subject):
    image = torch.full((3, 20, 40), background, dtype=torch.uint8)
    image[:, 3:10, 20:25] = subject
    frame = Frame(image, "sample")
    mask = torch.zeros((1, 20, 40), dtype=torch.bool)
    mask[0, 3:10, 20:25] = True
    geometry = Boxes(torch.tensor([[0, 0, 4, 4]], dtype=torch.float32))
    propagated = _tracks(geometry, detection_index=-1, masks=MaskBatch(mask))
    crop = _decode(encode_track_crop(frame, propagated, 0, _config()))
    assert crop.shape == (7, 5, 3)
    assert np.abs(crop.astype(int) - subject).max() <= 1
    matched = _tracks(geometry, masks=MaskBatch(mask))
    matched_crop = _decode(encode_track_crop(frame, matched, 0, _config()))
    assert matched_crop.shape == (4, 4, 3)
    assert np.abs(matched_crop.astype(int) - background).max() <= 1


def test_single_pixel_propagated_mask_at_frame_edge_produces_positive_crop():
    frame = _frame()
    mask = torch.zeros((1, 20, 40), dtype=torch.bool)
    mask[0, -1, -1] = True
    geometry = Boxes(torch.tensor([[45, 0, 50, 10]], dtype=torch.float32))
    propagated = _tracks(geometry, detection_index=-1, masks=MaskBatch(mask))
    crop = _decode(encode_track_crop(frame, propagated, 0, _config()))
    assert crop.shape == (1, 1, 3)
    assert crop[0, 0, 2] > 230


def test_crop_skips_outside_frame_and_rejects_misaligned_masks():
    outside = _tracks(Boxes(torch.tensor([[45, 0, 50, 10]], dtype=torch.float32)))
    assert encode_track_crop(_frame(), outside, 0, _config()) is None
    tracks = _tracks(outside.geometry, masks=MaskBatch(torch.zeros((1, 2, 2), dtype=torch.bool)))
    with pytest.raises(ValueError, match="same image size"):
        encode_track_crop(_frame(), tracks, 0, _config())


def test_storyboard_uses_ordered_samples_and_rejects_bad_batches():
    tracks = _tracks(Boxes(torch.tensor([[1, 1, 10, 15]], dtype=torch.float32)))
    encoded = encode_track_crop(_frame(), tracks, 0, _config())
    samples = [AttributeSample(f"sample-{i}", i, float(i)) for i in range(5)]
    assert _decode(build_storyboard(4, samples, [encoded] * 5)).shape == (960, 960, 3)
    with pytest.raises(ValueError, match="chronological"):
        build_storyboard(4, list(reversed(samples)), [encoded] * 5)
    with pytest.raises(ValueError, match="duplicate"):
        build_storyboard(4, [samples[0], samples[0]], [encoded, encoded])
    with pytest.raises(ValueError, match="aligned"):
        build_storyboard(4, samples, [encoded])
    with pytest.raises(ValueError, match="readable"):
        build_storyboard(4, samples[:1], [b"invalid jpeg"])


@pytest.mark.parametrize("bounds,expected_shape", [([0, 0, 40, 20], (8, 16, 3)), ([0, 0, 10, 20], (16, 8, 3))])
def test_single_sample_preserves_bounded_crop_without_padding_or_reencoding(bounds, expected_shape):
    """A small first request must keep the configured crop size in both orientations."""
    tracks = _tracks(Boxes(torch.tensor([bounds], dtype=torch.float32)))
    encoded = encode_track_crop(_frame(), tracks, 0, _config(max_crop_side=16))
    image = build_storyboard(4, [AttributeSample("sample", 0, 0.0)], [encoded])

    assert image == encoded
    assert _decode(image).shape == expected_shape


def test_single_sample_converts_png_to_jpeg_without_padding_or_resizing():
    """Caller-supplied images must match the JPEG media type used by the transport."""
    crop = np.full((24, 16, 3), (0, 0, 255), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", crop)
    assert success

    image = build_storyboard(4, [AttributeSample("sample", 0, 0.0)], [encoded.tobytes()])

    assert image.startswith(b"\xff\xd8\xff")
    decoded = _decode(image)
    assert decoded.shape == crop.shape
    assert decoded[12, 8, 2] > 240


@pytest.mark.parametrize("encoded", [b"", b"invalid jpeg"])
def test_single_sample_rejects_unreadable_crop(encoded):
    """Skipping composition must retain image validation."""
    with pytest.raises(ValueError, match="readable"):
        build_storyboard(4, [AttributeSample("sample", 0, 0.0)], [encoded])
