"""GTA joins replay observations by frame and detection identity."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from boxmot.datasets import DatasetSample
from boxmot.engine.eval.gta import GTA_PARAMETERS, associate_track_rows
from boxmot.structures import Boxes, Detections


def _sample(frame_index: int, features: list[list[float]] | None, *, count: int = 1) -> DatasetSample:
    """Create a cached sample without image decoding or model construction."""
    sample_id = f"validation:sequence:{frame_index}"
    count = count if features is None else len(features)
    return DatasetSample(
        sample_id=sample_id,
        split="validation",
        sequence_id="sequence",
        frame_index=frame_index,
        timestamp_s=None,
        image_size=(40, 60),
        image_ref=None,
        frame=None,
        detections=Detections(
            geometry=Boxes(torch.tensor([[1, 2, 11, 22]] * count, dtype=torch.float32)),
            scores=torch.full((count,), 0.9, dtype=torch.float32),
            class_ids=torch.zeros(count, dtype=torch.int64),
            sample_id=sample_id,
            embeddings=None if features is None else torch.tensor(features, dtype=torch.float32),
        ),
    )


def _row(frame: int, track_id: int, *, class_id: int = 0, detection_index: int = 0) -> list[float]:
    """Keep fractional geometry/scores so precision loss is observable."""
    return [
        frame,
        track_id,
        1.123456789,
        2.234567891,
        10.345678912,
        20.456789123,
        0.912345678,
        class_id,
        detection_index,
    ]


def _assert_only_ids_changed(output: np.ndarray, original: np.ndarray) -> None:
    assert output.shape == original.shape
    np.testing.assert_array_equal(output[:, [0, 2, 3, 4, 5, 6, 7, 8]], original[:, [0, 2, 3, 4, 5, 6, 7, 8]])


def test_gta_joins_features_by_frame_and_detection_index_preserving_exact_rows() -> None:
    rows = np.asarray([_row(3, 20), _row(1, 10, detection_index=1)], dtype=np.float64)
    original = rows.copy()
    samples = [_sample(2, [[4, 0], [0, 1]]), _sample(0, [[0, 1], [2, 0]])]

    output = associate_track_rows(rows, iter(samples))

    assert output[0, 1] == output[1, 1]
    _assert_only_ids_changed(output, original)
    np.testing.assert_array_equal(rows, original)
    assert samples[0].detections.embeddings[0, 0] == 4


def test_gta_never_merges_classes_or_simultaneous_observations() -> None:
    rows = np.asarray([_row(1, 10), _row(1, 20, detection_index=1), _row(2, 30, class_id=1)], dtype=np.float64)
    output = associate_track_rows(rows, [_sample(0, [[1, 0], [1, 0]]), _sample(1, [[1, 0]])])

    assert len(set(output[:, 1])) == 3
    _assert_only_ids_changed(output, rows)


def test_gta_splits_mixed_identity_track_and_attaches_unmatched_rows_by_nearest_frame() -> None:
    rows = np.asarray([_row(frame, 10, detection_index=-1 if frame == 51 else 0) for frame in range(1, 102)])
    samples = [_sample(frame - 1, [[1, 0] if frame <= 50 else [0, 1]]) for frame in range(1, 102) if frame != 51]
    rows = rows[::-1].copy()

    output = associate_track_rows(rows, samples)

    by_frame = {int(row[0]): int(row[1]) for row in output}
    assert len(set(by_frame.values())) == 2
    assert by_frame[1] == by_frame[50] == by_frame[51]
    assert by_frame[52] == by_frame[101] != by_frame[51]
    _assert_only_ids_changed(output, rows)


def test_unmatched_rows_prevent_merge_collisions_and_all_unmatched_tracks_stay_distinct() -> None:
    rows = np.asarray(
        [
            _row(1, 10),
            _row(2, 10, detection_index=-1),
            _row(2, 20),
            _row(3, 30, detection_index=-1),
            _row(4, 30, detection_index=-1),
            _row(5, 40, detection_index=-1),
        ],
        dtype=np.float64,
    )
    output = associate_track_rows(rows, [_sample(0, [[1, 0]]), _sample(1, [[1, 0]])])

    assert output[0, 1] == output[1, 1]
    assert output[0, 1] != output[2, 1]
    assert output[3, 1] == output[4, 1]
    assert len(set(output[:, 1])) == 4
    assert len({(row[0], row[1]) for row in output}) == len(rows)
    _assert_only_ids_changed(output, rows)


@pytest.mark.parametrize("rows", [np.empty((0, 9)), np.asarray([_row(1, 10, detection_index=-1)])])
def test_gta_does_not_read_embeddings_without_matched_observations(rows: np.ndarray) -> None:
    def unreadable_dataset():
        raise AssertionError("No appearance observation needs to be loaded")
        yield

    output = associate_track_rows(rows, unreadable_dataset())

    _assert_only_ids_changed(output, rows)
    assert output is not rows


@pytest.mark.parametrize("features", [None, [[0, 0]]])
def test_gta_rejects_missing_or_zero_embeddings(features: list[list[float]] | None) -> None:
    with pytest.raises(ValueError, match="embeddings"):
        associate_track_rows(np.asarray([_row(1, 10)]), [_sample(0, features)])


def test_gta_revalidates_mutated_nonfinite_embeddings() -> None:
    sample = _sample(0, [[1, 0]])
    sample.detections.embeddings[0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite, nonzero embeddings"):
        associate_track_rows(np.asarray([_row(1, 10)]), [sample])


def test_gta_rejects_out_of_range_detection_indices() -> None:
    with pytest.raises(ValueError, match="detection index 1 is out of range"):
        associate_track_rows(np.asarray([_row(1, 10, detection_index=1)]), [_sample(0, [[1, 0]])])


def test_gta_rejects_missing_frames_and_inconsistent_embedding_dimensions() -> None:
    rows = np.asarray([_row(1, 10), _row(2, 20)])
    with pytest.raises(ValueError, match="frame numbers"):
        associate_track_rows(rows, [_sample(0, [[1, 0]])])
    with pytest.raises(ValueError, match="consistent embedding dimension"):
        associate_track_rows(rows, [_sample(0, [[1, 0]]), _sample(1, [[1, 0, 0]])])


def test_gta_rejects_ambiguous_dataset_frame_identity() -> None:
    rows = np.asarray([_row(1, 10)])
    sample = _sample(0, [[1, 0]])
    with pytest.raises(ValueError, match="duplicate frame"):
        associate_track_rows(rows, [sample, sample])
    with pytest.raises(ValueError, match="exactly one sequence"):
        associate_track_rows(rows, [sample, replace(_sample(1, [[1, 0]]), sequence_id="other")])
    with pytest.raises(ValueError, match="identities do not match"):
        associate_track_rows(rows, [replace(sample, sample_id="wrong")])


@pytest.mark.parametrize(
    "rows",
    [np.zeros((1, 10)), np.asarray([_row(0, 10)]), np.asarray([_row(1, 10), _row(1, 10)])],
)
def test_gta_rejects_invalid_track_rows(rows: np.ndarray) -> None:
    with pytest.raises(ValueError, match="GTA"):
        associate_track_rows(rows, [])


def test_gta_parameters_are_json_scalars() -> None:
    assert all(type(value) in {int, float} for value in GTA_PARAMETERS.values())


def test_gta_reports_actual_work_for_each_class_without_console_output(capsys: pytest.CaptureFixture) -> None:
    rows = np.asarray([_row(1, 10), _row(2, 20), _row(3, 30, class_id=1), _row(4, 40, class_id=1)])
    samples = [_sample(frame, [[1, 0]]) for frame in range(4)]
    events: list[tuple[str, int, int | None]] = []

    output = associate_track_rows(rows, samples, progress_fn=lambda *event: events.append(event))
    without_progress = associate_track_rows(rows, samples)

    np.testing.assert_array_equal(output, without_progress)
    assert [(completed, total) for stage, completed, total in events if stage == "Join embeddings"] == [
        (completed, 4) for completed in range(5)
    ]
    for class_id in (0, 1):
        prefix = f"Class {class_id}: "
        assert [(current, total) for stage, current, total in events if stage == prefix + "Split tracklets"] == [
            (0, 2),
            (1, 2),
            (2, 2),
        ]
        assert (prefix + "Batch 1/1: Compute distances", 1, 1) in events
        assert (prefix + "Batch 1/1: Merge candidates", 1, None) in events
        assert (prefix + "Batch 1/1: Merge candidates", 1, 1) in events
        assert (prefix + "Global: Compute distances", 0, 0) in events
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("rows", [np.empty((0, 9)), np.asarray([_row(1, 10, detection_index=-1)])])
def test_gta_reports_empty_embedding_work_without_reading_dataset(rows: np.ndarray) -> None:
    events: list[tuple[str, int, int | None]] = []

    def unreadable_dataset():
        raise AssertionError("No appearance observation needs to be loaded")
        yield

    associate_track_rows(rows, unreadable_dataset(), progress_fn=lambda *event: events.append(event))

    assert events[0] == ("Join embeddings", 0, 0)
    assert all(total == 0 for _, _, total in events)
