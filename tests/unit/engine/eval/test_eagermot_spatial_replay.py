"""Preserve independent spatial identities while visualizing the evaluated KITTI masks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from click.testing import CliRunner

from boxmot import EagerMot
from boxmot.datasets.kitti_fusion import KittiFusionSequence
from boxmot.engine.cli import boxmot
from boxmot.engine.eval.eagermot_kitti import (
    KITTI_PROFILES,
    _track_frame,
    evaluate_eagermot_kitti,
    load_kitti_profiles,
    prepare_eagermot_kitti,
)
from boxmot.structures import CameraModel, MultimodalTracks, Tracks3D
from tests.unit.engine.eval.test_eagermot_kitti import _arguments, _fixture
from tests.unit.engine.eval.test_eagermot_visualization import _capture_visualizations, _RecordingVisualization


def _trackers() -> dict[int, EagerMot]:
    """Keep fixture projections eligible for image-only recovery after a 3D dropout."""
    return {class_id: EagerMot(**{**profile, "iou_threshold": 0.01}) for class_id, profile in KITTI_PROFILES.items()}


@pytest.mark.parametrize("reverse_spatial", [False, True])
def test_class_replay_preserves_shared_ids_and_independent_sensor_row_indices(
    tmp_path: Path, reverse_spatial: bool
) -> None:
    data = _fixture(tmp_path)
    sequence = KittiFusionSequence("0002", **data.reader_paths)
    frame = sequence[0]
    assert frame.detections.class_ids.tolist() == [2, 1]
    assert frame.detections_3d.class_ids.tolist() == [1, 2]
    if reverse_spatial:
        frame = replace(frame, detections_3d=frame.detections_3d.select(torch.tensor([1, 0])))

    result = _track_frame(frame, _trackers())

    assert isinstance(result, MultimodalTracks)
    image, spatial = result.image_tracks, result.spatial_tracks
    assert image.class_ids.tolist() == spatial.class_ids.tolist() == [1, 2]
    assert image.detection_indices.tolist() == [1, 0]
    assert spatial.detection_indices.tolist() == ([1, 0] if reverse_spatial else [0, 1])
    torch.testing.assert_close(image.track_ids, spatial.track_ids)
    assert len(image.track_ids.unique()) == 2
    assert image.sample_id == spatial.sample_id == frame.detections.sample_id
    torch.testing.assert_close(
        image.masks.values, frame.detections.masks.values.index_select(0, image.detection_indices)
    )
    torch.testing.assert_close(
        spatial.geometry.values, frame.detections_3d.geometry.values.index_select(0, spatial.detection_indices)
    )


def test_image_supported_spatial_prediction_preserves_missing_3d_index(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    sequence = KittiFusionSequence("0002", **data.reader_paths)
    trackers = _trackers()
    first_frame, empty_frame = sequence[0], sequence[1]
    initial = _track_frame(first_frame, trackers)
    image_supported = replace(
        empty_frame,
        detections=replace(first_frame.detections, sample_id=empty_frame.detections.sample_id),
    )
    assert len(image_supported.detections_3d) == 0

    recovered = _track_frame(image_supported, trackers)

    assert len(recovered.image_tracks) == len(recovered.spatial_tracks) == 2
    assert recovered.spatial_tracks.detection_indices.tolist() == [-1, -1]
    assert recovered.image_tracks.detection_indices.tolist() == [1, 0]
    torch.testing.assert_close(recovered.image_tracks.track_ids, initial.image_tracks.track_ids)
    torch.testing.assert_close(recovered.spatial_tracks.track_ids, initial.spatial_tracks.track_ids)
    torch.testing.assert_close(recovered.image_tracks.masks.values, initial.image_tracks.masks.values)
    assert recovered.sample_id == "train:0002:1"


def test_spatial_only_support_retains_tracks_without_fabricating_image_masks(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    sequence = KittiFusionSequence("0002", **data.reader_paths)
    trackers = _trackers()
    first_frame, empty_frame = sequence[0], sequence[1]
    initial = _track_frame(first_frame, trackers)
    spatial_supported = replace(
        empty_frame,
        detections_3d=replace(first_frame.detections_3d, sample_id=empty_frame.detections.sample_id),
    )

    recovered = _track_frame(spatial_supported, trackers)

    assert len(recovered.image_tracks) == 0
    assert recovered.image_tracks.masks.values.shape == (0, 24, 48)
    assert len(recovered.spatial_tracks) == 2
    assert recovered.spatial_tracks.detection_indices.tolist() == [0, 1]
    torch.testing.assert_close(recovered.spatial_tracks.track_ids, initial.spatial_tracks.track_ids)
    unsupported = replace(
        sequence[2],
        detections=replace(empty_frame.detections, sample_id="train:0002:2"),
        detections_3d=replace(empty_frame.detections_3d, sample_id="train:0002:2"),
    )
    prediction_only = _track_frame(unsupported, trackers)
    assert len(prediction_only.image_tracks) == len(prediction_only.spatial_tracks) == 0
    assert all(len(tracker._tracks) == 1 for tracker in trackers.values())


class _SpatialVisualization(_RecordingVisualization):
    """Record the optional spatial payload separately from the image result."""

    def __init__(self, output: Path, **options: Any) -> None:
        super().__init__(output, **options)
        self.payloads: list[dict[str, Any]] = []

    def __call__(self, replayed: Any, **payload: Any) -> None:
        super().__call__(replayed)
        self.payloads.append(payload)


def test_show_3d_payload_is_opt_in_and_preserves_mots_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _fixture(tmp_path)
    renderers = _capture_visualizations(monkeypatch, _SpatialVisualization)
    plain = CliRunner().invoke(boxmot, [*_arguments(data), "--save"])
    assert plain.exit_code == 0, (plain.output, plain.exception)

    spatial = CliRunner().invoke(boxmot, [*_arguments(data), "--save", "--show-3d"])

    assert spatial.exit_code == 0, (spatial.output, spatial.exception)
    assert len(renderers) == 2
    baseline, enhanced = renderers
    assert baseline.payloads == [{}, {}, {}]
    assert baseline.closed and enhanced.closed
    assert [event.sample.frame_index for event in enhanced.frames] == [0, 1, 2]
    assert [len(payload["spatial_tracks"]) for payload in enhanced.payloads] == [2, 0, 2]
    for event, payload in zip(enhanced.frames, enhanced.payloads, strict=True):
        assert set(payload) == {"spatial_tracks", "camera"}
        tracks, camera = payload["spatial_tracks"], payload["camera"]
        assert isinstance(tracks, Tracks3D)
        assert isinstance(camera, CameraModel)
        assert camera.image_size == event.sample.image_size
        assert tracks.sample_id == event.sample.sample_id
        torch.testing.assert_close(tracks.track_ids, event.result.tracks.track_ids)
        torch.testing.assert_close(tracks.class_ids, event.result.tracks.class_ids)
        assert tracks.detection_indices.tolist() == ([0, 1] if len(tracks) else [])
    first, second = data.project / "val", data.project / "val2"
    assert (first / "mots/0002.txt").read_bytes() == (second / "mots/0002.txt").read_bytes()
    assert json.loads((first / "metrics.json").read_text()) == json.loads((second / "metrics.json").read_text())
    assert json.loads((first / "run.json").read_text())["visualization"]["show_3d"] is False
    assert json.loads((second / "run.json").read_text())["visualization"]["show_3d"] is True


def test_python_show_3d_requires_an_output_mode_before_creating_results(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    inputs = prepare_eagermot_kitti(
        SimpleNamespace(
            dataset=data.dataset,
            split="val",
            sequence_names=("0002",),
        )
    )
    output = data.project / "invalid"
    previous_threads = torch.get_num_threads()

    with pytest.raises(ValueError, match="show.*save"):
        evaluate_eagermot_kitti(inputs, load_kitti_profiles(), output, show_3d=True)

    assert not output.exists()
    assert torch.get_num_threads() == previous_threads
