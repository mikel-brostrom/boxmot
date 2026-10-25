"""Detection packing preserves precision and isolates persistent tracker state."""

from collections import deque
from dataclasses import fields

import numpy as np
import pytest
import torch

from boxmot.structures import Boxes, Detections, MaskBatch, OrientedBoxes
from boxmot.trackers.common.config import get_tracker_config_class
from boxmot.trackers.common.detections import _DetectionBatch
from boxmot.trackers.common.detections.layout import AABB_DETECTIONS, OBB_DETECTIONS
from boxmot.trackers.common.registry import get_tracker_class
from boxmot.trackers.common.tracking.collections import TRACK_COLLECTION_ATTRS


@pytest.mark.parametrize("layout", [AABB_DETECTIONS, OBB_DETECTIONS], ids=["aabb", "obb"])
@pytest.mark.parametrize("input_dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize(
    "output_dtype", [np.float16, np.float32, np.float64, None, np.int32, object, "U32", str, bytes]
)
def test_detection_packing_preserves_column_promotion(layout, input_dtype, output_dtype) -> None:
    """Packing retains the old rounding and dtype rules, including custom outputs."""
    rows = np.arange(3 * (layout.det_cols + 1), dtype=input_dtype).reshape(3, -1)
    rows[:, :layout.box_cols] += input_dtype(0.123456789)
    rows[0, 0] = -0.0
    rows[:, layout.conf_idx] = [0.1, 0.9, 0.7]
    batch = _DetectionBatch.from_layout(rows, layout, copy=False).select([2, 0])

    indexed = batch.as_indexed_detections(dtype=output_dtype)
    boxes = batch.as_box_conf_detections(dtype=output_dtype)

    expected_indexed = np.column_stack((batch.boxes, batch.confs, batch.clss, batch.det_inds)).astype(output_dtype)
    expected_boxes = np.column_stack((batch.boxes, batch.confs)).astype(output_dtype)
    np.testing.assert_array_equal(indexed, expected_indexed, strict=True)
    np.testing.assert_array_equal(boxes, expected_boxes, strict=True)
    if indexed.dtype.kind == "f":
        np.testing.assert_array_equal(np.signbit(indexed), np.signbit(expected_indexed))
    assert indexed.flags.c_contiguous and boxes.flags.c_contiguous
    assert not np.shares_memory(indexed, batch.boxes)
    assert not np.shares_memory(boxes, batch.boxes)


@pytest.mark.parametrize("output_dtype", [np.float32, np.float64, np.int64, object, "U32"])
def test_custom_detection_batch_preserves_mixed_dtype_promotion(output_dtype) -> None:
    """Directly constructed batches retain NumPy's intermediate conversion rules."""
    batch = _DetectionBatch(
        boxes=np.array([[2**53 + 1, 2, 3, 4]], dtype=np.int64),
        confs=np.array([0.5], dtype=np.float32),
        clss=np.array([2**53 + 3], dtype=np.int64),
        det_inds=np.array([0], dtype=np.int32),
    )
    for packed, columns in (
        (batch.as_indexed_detections(output_dtype), (batch.boxes, batch.confs, batch.clss, batch.det_inds)),
        (batch.as_box_conf_detections(output_dtype), (batch.boxes, batch.confs)),
    ):
        np.testing.assert_array_equal(packed, np.column_stack(columns).astype(output_dtype), strict=True)


@pytest.mark.parametrize("layout", [AABB_DETECTIONS, OBB_DETECTIONS], ids=["aabb", "obb"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, None])
def test_empty_detection_packing_keeps_shape_and_dtype(layout, dtype) -> None:
    batch = _DetectionBatch.from_layout(np.empty((0, layout.det_cols), dtype=np.float32), layout)
    assert batch.as_indexed_detections(dtype).shape == (0, layout.det_cols + 1)
    assert batch.as_box_conf_detections(dtype).shape == (0, layout.box_cols + 1)
    assert batch.as_indexed_detections(dtype).dtype == np.dtype(dtype)
    assert batch.as_box_conf_detections(dtype).dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_default_detection_batch_owns_all_columns(dtype) -> None:
    """The default still isolates even class/index inputs already stored as int32."""
    rows = np.arange(21, dtype=dtype).reshape(3, 7)
    embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
    masks = np.ones((3, 5, 6), dtype=bool)
    batch = _DetectionBatch.from_layout(rows, AABB_DETECTIONS, embs=embeddings, masks=masks)
    columns = (batch.boxes, batch.confs, batch.clss, batch.det_inds, batch.embs, batch.masks)
    before = [column.copy() for column in columns]

    for column in columns:
        assert not any(np.shares_memory(column, source) for source in (rows, embeddings, masks))
    rows[:] = 0
    embeddings[:] = 0
    masks[:] = False
    for column, expected in zip(columns, before, strict=True):
        np.testing.assert_array_equal(column, expected)


@pytest.mark.parametrize("indices", [[0, 1, 2], [2, 0], [True, True, True], [False, True, False]])
def test_selection_detaches_borrowed_arrays_even_when_every_row_is_selected(indices) -> None:
    """Tracks can retain selected feature rows after callers reuse their buffers."""
    rows = np.arange(21, dtype=np.float32).reshape(3, 7)
    embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
    masks = np.ones((3, 5, 6), dtype=bool)
    borrowed = _DetectionBatch.from_layout(rows, AABB_DETECTIONS, embs=embeddings, masks=masks, copy=False)
    assert np.shares_memory(borrowed.boxes, rows)
    assert np.shares_memory(borrowed.embs, embeddings)
    assert np.shares_memory(borrowed.masks, masks)
    selected = borrowed.select(indices)
    for name in ("boxes", "confs", "clss", "det_inds", "embs", "masks"):
        column, source = getattr(selected, name), getattr(borrowed, name)
        np.testing.assert_array_equal(column, source[indices])
        assert not np.shares_memory(column, source)


def test_default_tracker_detection_batch_still_owns_inputs() -> None:
    """Custom/private callers retain defensive copying unless they opt out."""
    tracker = get_tracker_class("bytetrack")()
    rows = np.array([[1, 2, 11, 22, 0.9, 3]], dtype=np.float32)
    embeddings = np.ones((1, 4), dtype=np.float32)
    masks = np.ones((1, 4, 5), dtype=bool)
    batch = tracker._make_detection_batch(rows, embs=embeddings, masks=masks)
    assert not np.shares_memory(batch.boxes, rows)
    assert not np.shares_memory(batch.embs, embeddings)
    assert not np.shares_memory(batch.masks, masks)


def _array_values(value):
    """Enumerate arrays retained directly or inside track history containers."""
    if isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _array_values(child)
    elif isinstance(value, (list, tuple, deque)):
        for child in value:
            yield from _array_values(child)


def _tracker(name: str, is_obb: bool):
    config_class = get_tracker_config_class(name)
    options = {
        "min_hits": 1,
        "n_init": 1,
        "use_cmc": False,
        "use_embeddings": True,
        "s2ta_mode": "motion",
        "t2ta_mode": "motion",
    }
    field_names = {field.name for field in fields(config_class)}
    config = config_class(**{key: value for key, value in options.items() if key in field_names})
    return get_tracker_class(name)(config=config, **({} if name == "maf_hda" else {"is_obb": is_obb}))


_TRACKER_GEOMETRIES = [
    (name, is_obb)
    for name in (
        "bytetrack", "botsort", "strongsort", "ocsort", "deepocsort",
        "hybridsort", "boosttrack", "occluboost", "sfsort", "maf_hda",
    )
    for is_obb in ((False,) if name == "maf_hda" else (False, True))
]


@pytest.mark.parametrize("name,is_obb", _TRACKER_GEOMETRIES)
@pytest.mark.parametrize("private_kernel", [False, True], ids=["public", "private"])
def test_borrowed_kernel_batches_preserve_outputs_and_input_state_ownership(name, is_obb, private_kernel) -> None:
    """Compare copying batches through births, filtering, updates and caller reuse."""
    actual, reference = _tracker(name, is_obb), _tracker(name, is_obb)
    reference._make_detection_batch = lambda dets, embs=None, masks=None, **_: _DetectionBatch.from_layout(
        dets, reference.detection_layout, embs=embs, masks=masks, copy=True
    )
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    for tracker in (actual, reference):
        tracker._initialize_frame_context(image)
        if private_kernel:
            tracker._encode_kernel_class_id(0)
            tracker._encode_kernel_class_id(1)
    observed_tracks = False
    for frame, scores in enumerate(([0.95, 0.15, 0.03], [0.94, 0.95, 0.03], [0.93, 0.95, 0.95])):
        boxes = np.array([[10, 10, 30, 40], [50, 10, 70, 40], [90, 10, 110, 40]], dtype=np.float32)
        boxes[:, [0, 2]] += frame
        if is_obb:
            boxes = np.column_stack(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2],
                                     np.full(3, 0.2 + frame * 0.03, dtype=np.float32)))
        features = np.eye(3, 4, dtype=np.float32) if actual.requirements.embeddings else None
        masks = np.zeros((3, 96, 128), dtype=bool) if actual.requirements.masks else None
        if masks is not None:
            for index in range(3):
                masks[index, 10:40, 10 + index * 40 + frame:30 + index * 40 + frame] = True
        rows = np.column_stack((boxes, np.asarray(scores, dtype=np.float32), np.array([0, 0, 1], dtype=np.float32)))
        if private_kernel:
            inputs = [value for value in (rows, features, masks) if value is not None]
            before = [value.copy() for value in inputs]
            result = actual._track_detections(rows, image, embs=features, masks=masks)
            expected = reference._track_detections(
                rows.copy(), image, embs=None if features is None else features.copy(),
                masks=None if masks is None else masks.copy(),
            )
            result_parts = result if isinstance(result, tuple) else (result,)
            expected_parts = expected if isinstance(expected, tuple) else (expected,)
            for value, expected_value in zip(result_parts, expected_parts, strict=True):
                np.testing.assert_array_equal(value, expected_value)
        else:
            detections = Detections(
                geometry=(OrientedBoxes if is_obb else Boxes)(torch.from_numpy(boxes)),
                scores=torch.tensor(scores, dtype=torch.float32),
                class_ids=torch.tensor([0, 0, 1], dtype=torch.int64),
                sample_id=f"sample/{frame}",
                embeddings=None if features is None else torch.from_numpy(features),
                masks=None if masks is None else MaskBatch(torch.from_numpy(masks)),
            )
            inputs = [boxes, detections.scores.numpy(), detections.class_ids.numpy()]
            inputs.extend(value for value in (features, masks) if value is not None)
            before = [value.copy() for value in inputs]
            result = actual.update(detections, image)
            expected = reference.update(detections, image)
            for field in ("track_ids", "scores", "class_ids", "detection_indices"):
                torch.testing.assert_close(getattr(result, field), getattr(expected, field), rtol=0, atol=0)
            torch.testing.assert_close(result.geometry.values, expected.geometry.values, rtol=0, atol=0)
            if result.masks is not None:
                torch.testing.assert_close(result.masks.values, expected.masks.values)

        for value, expected_value in zip(inputs, before, strict=True):
            np.testing.assert_array_equal(value, expected_value)
        tracks = {id(track): track for attr in TRACK_COLLECTION_ATTRS for track in getattr(actual, attr, [])}
        observed_tracks |= bool(tracks)
        retained = [array for track in tracks.values() for array in _array_values(vars(track))]
        snapshots = [array.copy() for array in retained]
        for array in retained:
            assert not any(np.shares_memory(array, source) for source in inputs)
        for source in inputs:
            source[:] = 0
        for array, snapshot in zip(retained, snapshots, strict=True):
            np.testing.assert_array_equal(array, snapshot)
    assert observed_tracks
