"""Auxiliary propagated masks use track colors without changing tracking results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest
import torch

from boxmot.engine.tracking import sinks
from boxmot.pipelines import PipelineResult
from boxmot.structures import Boxes, Detections, Frame, MaskBatch, Tracks

IMAGE_SIZE = (64, 96)
BGR = np.array([60, 40, 20], dtype=np.uint8)


def _mask(left: int, right: int) -> np.ndarray:
    mask = np.zeros(IMAGE_SIZE, dtype=bool)
    mask[34:50, left:right] = True
    mask.setflags(write=False)
    return mask


def _sample(*, masks: str | None = None, empty_tracks: bool = False) -> tuple[Frame, PipelineResult]:
    frame = Frame(
        torch.tensor([20, 40, 60], dtype=torch.uint8)[:, None, None].expand(3, *IMAGE_SIZE).contiguous(),
        sample_id="rendering/0",
    )
    boxes = Boxes(torch.tensor([[8, 28, 30, 55], [58, 28, 80, 55]], dtype=torch.float32))
    mask_values = torch.from_numpy(np.stack((_mask(12, 26), _mask(62, 76))))
    detections = Detections(
        boxes,
        torch.tensor([0.9, 0.9], dtype=torch.float32),
        torch.zeros(2, dtype=torch.int64),
        frame.sample_id,
        masks=MaskBatch(mask_values.flip(0)) if masks == "detections" else None,
    )
    tracks = Tracks(
        boxes,
        torch.tensor([11, 3], dtype=torch.int64),
        detections.scores,
        detections.class_ids,
        torch.tensor([1, 0], dtype=torch.int64),
        frame.sample_id,
        masks=MaskBatch(mask_values) if masks == "tracks" else None,
    )
    if empty_tracks:
        tracks = tracks.select(torch.empty(0, dtype=torch.int64))
    return frame, PipelineResult(detections, tracks)


def _blended(track_id: int) -> np.ndarray:
    return np.rint(np.asarray(sinks._track_color(track_id)) * 0.35 + BGR * 0.65).astype(np.uint8)


def test_guidance_colors_follow_ids_and_boxes_remain_on_top() -> None:
    frame, result = _sample()
    left, right = _mask(8, 26), _mask(58, 76)
    masks = MappingProxyType({3: right, np.int64(11): left})

    rendered = sinks.render_result(frame, result, guidance_masks=masks)

    np.testing.assert_array_equal(rendered[40, 20], _blended(11))
    np.testing.assert_array_equal(rendered[40, 68], _blended(3))
    np.testing.assert_array_equal(rendered[40, 8], sinks._track_color(11))
    np.testing.assert_array_equal(rendered[40, 58], sinks._track_color(3))
    np.testing.assert_array_equal(rendered[60, 90], BGR)


def test_lost_identity_masks_render_without_current_box_outputs() -> None:
    frame, result = _sample(empty_tracks=True)

    rendered = sinks.render_result(frame, result, guidance_masks={7: _mask(40, 52)})

    assert len(result.tracks) == 0
    np.testing.assert_array_equal(rendered[40, 45], _blended(7))
    np.testing.assert_array_equal(rendered[40, 20], BGR)


def test_overlapping_guidance_layering_is_independent_of_mapping_order() -> None:
    frame, result = _sample(empty_tracks=True)
    shared = _mask(40, 52)

    first = sinks.render_result(frame, result, guidance_masks={11: shared, 3: shared})
    second = sinks.render_result(frame, result, guidance_masks={3: shared, 11: shared})

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[40, 45], _blended(11))


@pytest.mark.parametrize("mask_source", ["tracks", "detections"])
def test_none_preserves_existing_masks_and_empty_guidance_suppresses_them(mask_source: str) -> None:
    frame, result = _sample(masks=mask_source)

    original = sinks.render_result(frame, result)
    unspecified = sinks.render_result(frame, result, guidance_masks=None)
    empty = sinks.render_result(frame, result, guidance_masks={})
    guided = sinks.render_result(frame, result, guidance_masks={11: _mask(12, 26)})

    np.testing.assert_array_equal(original, unspecified)
    np.testing.assert_array_equal(original[40, 20], _blended(11))
    np.testing.assert_array_equal(original[40, 68], _blended(3))
    np.testing.assert_array_equal(empty[40, 20], BGR)
    np.testing.assert_array_equal(empty[40, 68], BGR)
    np.testing.assert_array_equal(guided[40, 20], _blended(11))
    np.testing.assert_array_equal(guided[40, 68], BGR)


def test_detection_mask_alignment_ignores_invalid_detection_indices() -> None:
    frame, result = _sample(masks="detections")
    tracks = replace(result.tracks, detection_indices=torch.tensor([-1, 0], dtype=torch.int64))

    rendered = sinks.render_result(frame, replace(result, tracks=tracks))

    np.testing.assert_array_equal(rendered[40, 20], BGR)
    np.testing.assert_array_equal(rendered[40, 68], _blended(3))


@pytest.mark.parametrize("mask_source", [None, "tracks", "detections"])
def test_rendering_preserves_frame_and_readonly_mask_storage(mask_source: str) -> None:
    frame, result = _sample(masks=mask_source)
    foreground = _mask(12, 26)
    pixels = frame.image.clone()
    expected_mask = foreground.copy()
    values = result.tracks.masks if mask_source == "tracks" else result.detections.masks
    canonical_masks = None if values is None else values.values.clone()

    rendered = sinks.render_result(frame, result, guidance_masks={11: foreground})
    rendered.fill(0)

    torch.testing.assert_close(frame.image, pixels)
    np.testing.assert_array_equal(foreground, expected_mask)
    assert not foreground.flags.writeable
    assert not np.shares_memory(rendered, frame.image.numpy())
    if values is not None:
        torch.testing.assert_close(values.values, canonical_masks)


@pytest.mark.parametrize(
    "mask, exception, message",
    [
        (np.zeros((64, 95), dtype=bool), ValueError, "frame dimensions"),
        (np.zeros((1, 64, 96), dtype=bool), ValueError, "two-dimensional"),
        (np.zeros(IMAGE_SIZE, dtype=np.uint8), TypeError, "boolean NumPy"),
        (torch.zeros(IMAGE_SIZE, dtype=torch.bool), TypeError, "boolean NumPy"),
    ],
)
def test_guidance_mask_contract_rejects_invalid_storage(mask, exception, message) -> None:
    frame, result = _sample()
    with pytest.raises(exception, match=message):
        sinks.render_result(frame, result, guidance_masks={11: mask})


@pytest.mark.parametrize("track_id", [-1, True, "11", 11.0])
def test_guidance_rejects_noncanonical_track_ids(track_id) -> None:
    frame, result = _sample()
    with pytest.raises(ValueError, match="non-negative integers"):
        sinks.render_result(frame, result, guidance_masks={track_id: _mask(12, 26)})


def test_guidance_renderer_borrows_masks_and_blends_one_overlay(monkeypatch) -> None:
    frame, result = _sample(empty_tracks=True)
    foreground = _mask(12, 26)
    add_weighted = sinks.cv2.addWeighted
    blends = []

    def blend(overlay, alpha, image, beta, gamma, *, dst):
        assert dst is image
        blends.append(overlay)
        return add_weighted(overlay, alpha, image, beta, gamma, dst=dst)

    monkeypatch.setattr(sinks.cv2, "addWeighted", blend)
    monkeypatch.setattr(np, "stack", lambda *a, **kw: pytest.fail("Stacked full-frame masks"))
    monkeypatch.setattr(np, "zeros", lambda *a, **kw: pytest.fail("Allocated a full-frame mask batch"))

    rendered = sinks.render_result(frame, result, guidance_masks={identity: foreground for identity in range(32)})

    assert len(blends) == 1
    assert blends[0].shape == (*IMAGE_SIZE, 3)
    np.testing.assert_array_equal(rendered[40, 20], _blended(31))


def test_shared_sink_reads_current_masks_and_renders_once_for_all_consumers(monkeypatch) -> None:
    frame, result = _sample()
    delivered = [[], []]
    consumers = [
        sinks.RenderingSink(lambda frame, result, image, bucket=bucket: bucket.append(image)) for bucket in delivered
    ]
    current: Mapping[int, np.ndarray] = {11: _mask(12, 26)}
    provider_calls = []
    render_calls = []
    render = sinks.render_result

    def provider() -> Mapping[int, np.ndarray]:
        provider_calls.append(current)
        return current

    def render_once(frame, result, **kwargs):
        assert kwargs["guidance_masks"] is current
        render_calls.append(current)
        return render(frame, result, **kwargs)

    monkeypatch.setattr(sinks, "render_result", render_once)
    shared = sinks.SharedRenderingSink(consumers, guidance_mask_provider=provider)
    shared.write(frame, result)
    current = {}
    shared.write(frame, result)

    assert len(provider_calls) == len(render_calls) == 2
    assert delivered[0][0] is delivered[1][0]
    assert delivered[0][1] is delivered[1][1]
    np.testing.assert_array_equal(delivered[0][0][40, 20], _blended(11))
    np.testing.assert_array_equal(delivered[0][1][40, 20], BGR)


def test_shared_sink_rejects_noncallable_guidance_provider() -> None:
    consumer = sinks.RenderingSink(lambda frame, result, image: None)
    with pytest.raises(TypeError, match="must be callable"):
        sinks.SharedRenderingSink((consumer,), guidance_mask_provider={})
