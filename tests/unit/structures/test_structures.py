from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import get_args

import pytest
import torch

from boxmot.structures import Boxes, Detections, Frame, Geometry, GeometryKind, MaskBatch, OrientedBoxes, Tracks


def _aabb_detections(*, masks: MaskBatch | None = None, embeddings: torch.Tensor | None = None) -> Detections:
    return Detections(
        geometry=Boxes(torch.tensor([[0.0, 1.0, 4.0, 6.0], [10.0, 11.0, 14.0, 16.0]])),
        scores=torch.tensor([0.8, 0.6], dtype=torch.float32),
        class_ids=torch.tensor([2, 3], dtype=torch.int64),
        sample_id="sequence-a:000001",
        instance_ids=("build:sequence-a:000001:0", "build:sequence-a:000001:1"),
        masks=masks,
        embeddings=embeddings,
    )


def test_frame_is_a_frozen_slotted_rgb_chw_value_without_copying() -> None:
    image = torch.zeros((3, 12, 20), dtype=torch.uint8)
    frame = Frame(
        image=image,
        sample_id="camera-a:42",
        sequence_id="camera-a",
        frame_index=42,
        timestamp_s=1.25,
        source_uri="video.mp4",
    )

    assert frame.image is image
    assert frame.image_size == (12, 20)
    assert not hasattr(frame, "__dict__")
    with pytest.raises(FrozenInstanceError):
        frame.sample_id = "replacement"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("image", "error", "message"),
    [
        (torch.zeros((12, 20, 3), dtype=torch.uint8), ValueError, r"shape \[3, H, W\]"),
        (torch.zeros((3, 12, 20), dtype=torch.float32), TypeError, "dtype"),
        (torch.zeros((3, 20, 12), dtype=torch.uint8).transpose(1, 2), ValueError, "contiguous"),
        (torch.empty((3, 12, 20), dtype=torch.uint8, device="meta"), ValueError, "CPU"),
    ],
)
def test_frame_rejects_noncanonical_image_tensors(image: torch.Tensor, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        Frame(image=image, sample_id="sample")


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"sample_id": "  "}, ValueError, "sample_id"),
        ({"sample_id": "sample", "sequence_id": ""}, ValueError, "sequence_id"),
        ({"sample_id": "sample", "frame_index": -1}, ValueError, "non-negative"),
        ({"sample_id": "sample", "frame_index": 1.0}, TypeError, "frame_index"),
        ({"sample_id": "sample", "timestamp_s": 1}, TypeError, "timestamp_s"),
        ({"sample_id": "sample", "timestamp_s": float("inf")}, ValueError, "finite"),
    ],
)
def test_frame_rejects_invalid_metadata(kwargs: dict[str, object], error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        Frame(image=torch.zeros((3, 2, 2), dtype=torch.uint8), **kwargs)  # type: ignore[arg-type]


def test_geometry_validates_shapes_values_and_unwrapped_obb_angles() -> None:
    boxes = Boxes(torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32))
    oriented = OrientedBoxes(torch.tensor([[1.0, 2.0, 3.0, 4.0, 9.5]], dtype=torch.float32))

    assert len(boxes) == len(oriented) == 1
    assert boxes.is_obb is False
    assert oriented.is_obb is True
    assert str(GeometryKind.AABB) == "aabb"
    assert str(GeometryKind.OBB) == "obb"

    with pytest.raises(ValueError, match="x2 > x1"):
        Boxes(torch.tensor([[0.0, 0.0, 0.0, 2.0]], dtype=torch.float32))
    with pytest.raises(ValueError, match="positive"):
        OrientedBoxes(torch.tensor([[1.0, 2.0, -3.0, 4.0, 0.0]], dtype=torch.float32))
    with pytest.raises(ValueError, match="finite"):
        OrientedBoxes(torch.tensor([[1.0, 2.0, 3.0, 4.0, float("nan")]], dtype=torch.float32))
    with pytest.raises(TypeError, match="dtype"):
        Boxes(torch.tensor([[0, 1, 2, 3]], dtype=torch.int64))


def test_image_geometry_alias_remains_limited_to_aabb_and_obb() -> None:
    """Adding camera-space boxes must not expand existing image contracts."""

    assert get_args(Geometry) == (Boxes, OrientedBoxes)


def test_mask_batch_is_full_frame_bool_and_selects_rows() -> None:
    values = torch.zeros((2, 4, 5), dtype=torch.bool)
    values[1, 2, 3] = True
    masks = MaskBatch(values)

    selected = masks.select(torch.tensor([1], dtype=torch.int64))

    assert masks.values is values
    assert masks.image_size == (4, 5)
    assert selected.values.shape == (1, 4, 5)
    assert selected.values[0, 2, 3]
    with pytest.raises(TypeError, match="dtype"):
        MaskBatch(torch.zeros((2, 4, 5), dtype=torch.uint8))
    with pytest.raises(ValueError, match="3 dimensions"):
        MaskBatch(torch.zeros((4, 5), dtype=torch.bool))


def test_detections_enrichment_and_selection_preserve_all_row_alignment() -> None:
    masks = MaskBatch(torch.arange(40).reshape(2, 4, 5).remainder(2).bool().contiguous())
    embeddings = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    detections = _aabb_detections().with_masks(masks).with_embeddings(embeddings)

    selected = detections.select(torch.tensor([1, 0], dtype=torch.int64))

    assert detections.masks is masks
    assert detections.embeddings is embeddings
    assert selected.sample_id == detections.sample_id
    assert selected.instance_ids == tuple(reversed(detections.instance_ids or ()))
    torch.testing.assert_close(selected.geometry.values, detections.geometry.values.flip(0))
    torch.testing.assert_close(selected.scores, detections.scores.flip(0))
    torch.testing.assert_close(selected.class_ids, detections.class_ids.flip(0))
    torch.testing.assert_close(selected.masks.values, masks.values.flip(0))
    torch.testing.assert_close(selected.embeddings, embeddings.flip(0))
    assert selected.geometry.values.is_contiguous()
    assert selected.masks.values.is_contiguous()
    assert selected.embeddings.is_contiguous()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("negative_view", [False, True])
def test_embedding_validation_checks_autograd_storage_and_lazy_views(nonfinite: float, negative_view: bool) -> None:
    """Validation must still inspect mutable tensor storage on every call."""
    values = torch.ones((2, 4), dtype=torch.float32, requires_grad=True)
    embeddings = torch._neg_view(values) if negative_view else values
    detections = _aabb_detections(embeddings=embeddings)
    assert detections.embeddings is embeddings
    with torch.no_grad():
        values[1, 3] = nonfinite
    with pytest.raises(ValueError, match="finite"):
        detections.validate()


def test_detections_boolean_selection_and_empty_batches_retain_geometry_mode() -> None:
    detections = _aabb_detections()

    selected = detections.select(torch.tensor([False, True]))
    empty = selected.select(torch.tensor([False]))

    assert isinstance(selected.geometry, Boxes)
    assert selected.instance_ids == ("build:sequence-a:000001:1",)
    assert isinstance(empty.geometry, Boxes)
    assert empty.geometry.values.shape == (0, 4)
    assert empty.to_aabb_rows().shape == (0, 6)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("scores", torch.tensor([0.8], dtype=torch.float32), ValueError, "aligned"),
        ("class_ids", torch.tensor([2, 3], dtype=torch.int32), TypeError, "dtype"),
        ("instance_ids", ("duplicate", "duplicate"), ValueError, "unique"),
        ("instance_ids", ["one", "two"], TypeError, "tuple"),
        ("embeddings", torch.zeros((1, 4), dtype=torch.float32), ValueError, "aligned"),
        ("embeddings", torch.zeros((2, 0), dtype=torch.float32), ValueError, "positive embedding dimension"),
        ("embeddings", torch.zeros((2, 4), dtype=torch.float64), TypeError, "dtype"),
    ],
)
def test_detections_reject_invalid_or_misaligned_fields(
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    kwargs = {
        "geometry": Boxes(torch.tensor([[0.0, 1.0, 4.0, 6.0], [10.0, 11.0, 14.0, 16.0]])),
        "scores": torch.tensor([0.8, 0.6], dtype=torch.float32),
        "class_ids": torch.tensor([2, 3], dtype=torch.int64),
        "sample_id": "sample",
        "instance_ids": ("one", "two"),
    }
    kwargs[field] = value

    with pytest.raises(error, match=message):
        Detections(**kwargs)  # type: ignore[arg-type]


def test_detections_reject_noncontiguous_canonical_tensor() -> None:
    embeddings = torch.zeros((3, 2), dtype=torch.float32).transpose(0, 1)
    assert embeddings.shape == (2, 3)
    assert not embeddings.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        _aabb_detections(embeddings=embeddings)


def test_detection_row_serializers_are_explicit_for_aabb_and_obb() -> None:
    aabb = _aabb_detections()
    obb = Detections(
        geometry=OrientedBoxes(torch.tensor([[5.0, 6.0, 4.0, 2.0, 0.25]], dtype=torch.float32)),
        scores=torch.tensor([0.7], dtype=torch.float32),
        class_ids=torch.tensor([8], dtype=torch.int64),
        sample_id="obb-sample",
    )

    torch.testing.assert_close(
        aabb.to_aabb_rows(),
        torch.tensor(
            [[0.0, 1.0, 4.0, 6.0, 0.8, 2.0], [10.0, 11.0, 14.0, 16.0, 0.6, 3.0]],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        obb.to_obb_rows(),
        torch.tensor([[5.0, 6.0, 4.0, 2.0, 0.25, 0.7, 8.0]], dtype=torch.float32),
    )
    assert aabb.to_aabb_rows().is_contiguous()
    assert obb.to_obb_rows().is_contiguous()
    with pytest.raises(TypeError, match="oriented detections"):
        obb.to_aabb_rows()
    with pytest.raises(TypeError, match="axis-aligned detections"):
        aabb.to_obb_rows()


def test_tracks_validate_and_select_track_aligned_masks() -> None:
    masks = MaskBatch(torch.zeros((2, 4, 5), dtype=torch.bool))
    masks.values[1, 1, 2] = True
    tracks = Tracks(
        geometry=Boxes(torch.tensor([[0.0, 1.0, 4.0, 6.0], [10.0, 11.0, 14.0, 16.0]])),
        track_ids=torch.tensor([101, 202], dtype=torch.int64),
        scores=torch.tensor([0.8, 0.6], dtype=torch.float32),
        class_ids=torch.tensor([2, 3], dtype=torch.int64),
        detection_indices=torch.tensor([0, -1], dtype=torch.int64),
        sample_id="sample",
        masks=masks,
    )

    selected = tracks.select(torch.tensor([1], dtype=torch.int64))
    replacement_masks = MaskBatch(torch.ones((2, 4, 5), dtype=torch.bool))
    replaced = tracks.with_masks(replacement_masks)

    assert selected.track_ids.tolist() == [202]
    assert selected.detection_indices.tolist() == [-1]
    assert selected.masks is not None and selected.masks.values[0, 1, 2]
    assert replaced.masks is replacement_masks
    assert tracks.masks is masks
    torch.testing.assert_close(
        tracks.to_aabb_rows(),
        torch.tensor(
            [
                [0.0, 1.0, 4.0, 6.0, 101.0, 0.8, 2.0, 0.0],
                [10.0, 11.0, 14.0, 16.0, 202.0, 0.6, 3.0, -1.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_obb_track_serializer_and_wrong_geometry_errors() -> None:
    tracks = Tracks(
        geometry=OrientedBoxes(torch.tensor([[5.0, 6.0, 4.0, 2.0, -0.25]], dtype=torch.float32)),
        track_ids=torch.tensor([7], dtype=torch.int64),
        scores=torch.tensor([0.9], dtype=torch.float32),
        class_ids=torch.tensor([1], dtype=torch.int64),
        detection_indices=torch.tensor([-1], dtype=torch.int64),
        sample_id="sample",
    )

    torch.testing.assert_close(
        tracks.to_obb_rows(),
        torch.tensor([[5.0, 6.0, 4.0, 2.0, -0.25, 7.0, 0.9, 1.0, -1.0]], dtype=torch.float32),
    )
    with pytest.raises(TypeError, match="oriented tracks"):
        tracks.to_aabb_rows()


@pytest.mark.parametrize(
    ("replacement", "error", "message"),
    [
        ({"track_ids": torch.tensor([1, 1], dtype=torch.int64)}, ValueError, "unique"),
        ({"scores": torch.tensor([0.5], dtype=torch.float32)}, ValueError, "aligned"),
        ({"detection_indices": torch.tensor([0, -2], dtype=torch.int64)}, ValueError, "only use -1"),
        ({"masks": MaskBatch(torch.zeros((1, 4, 5), dtype=torch.bool))}, ValueError, "aligned"),
    ],
)
def test_tracks_reject_invalid_or_misaligned_fields(
    replacement: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    kwargs = {
        "geometry": Boxes(torch.tensor([[0.0, 1.0, 4.0, 6.0], [10.0, 11.0, 14.0, 16.0]])),
        "track_ids": torch.tensor([1, 2], dtype=torch.int64),
        "scores": torch.tensor([0.8, 0.6], dtype=torch.float32),
        "class_ids": torch.tensor([2, 3], dtype=torch.int64),
        "detection_indices": torch.tensor([0, 1], dtype=torch.int64),
        "sample_id": "sample",
    }
    kwargs.update(replacement)

    with pytest.raises(error, match=message):
        Tracks(**kwargs)  # type: ignore[arg-type]
