"""Render the evaluated KITTI masks on every source frame without GUI dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.eval.mots_io import read_mots_results
from tests.unit.engine.eval.test_eagermot_kitti import _arguments, _fixture, mask_utils


class _RecordingVisualization:
    """Capture canonical replay events while exercising the real sensor evaluation."""

    def __init__(self, output: Path, **options: Any) -> None:
        self.output = output
        self.options = options
        self.frames: list[Any] = []
        self.entered = False
        self.closed = False

    def __enter__(self) -> _RecordingVisualization:
        self.entered = True
        return self

    def __exit__(self, *_exc: object) -> None:
        self.closed = True

    @property
    def video_paths(self) -> tuple[Path, ...]:
        """Expose the sink's reported artifact paths without opening a codec."""
        return (self.output / "videos/0002.mp4",) if self.options["save"] else ()

    def __call__(self, replayed: Any) -> None:
        self.frames.append(replayed)


def _capture_visualizations(
    monkeypatch: pytest.MonkeyPatch,
    renderer_type: type[_RecordingVisualization] = _RecordingVisualization,
) -> list[_RecordingVisualization]:
    """Replace the optional renderer at its canonical lazy import boundary."""
    import boxmot.engine.eval.visualization as visualization

    renderers = []

    def create(output: Path, **options: Any) -> _RecordingVisualization:
        renderer = renderer_type(output, **options)
        renderers.append(renderer)
        return renderer

    monkeypatch.setattr(visualization, "ReplayVisualization", create)
    return renderers


@pytest.mark.parametrize("flags", [("--show",), ("--save",), ("--show", "--save")])
def test_visualization_receives_evaluated_masks_ids_and_all_native_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: tuple[str, ...]
) -> None:
    data = _fixture(tmp_path)
    # Make the first pedestrian mask overlap a car pixel. Final MOTS preparation
    # must resolve it identically for visualization and the written evaluation.
    detections_path = data.reader_paths["detections_2d"]
    lines = detections_path.read_text().splitlines()
    fields = lines[0].split()
    mask = mask_utils.decode({"size": [24, 48], "counts": fields[9].encode("ascii")})
    mask[8, 14] = 1
    fields[9] = mask_utils.encode(np.asfortranarray(mask))["counts"].decode("ascii")
    lines[0] = " ".join(fields)
    detections_path.write_text("\n".join(lines) + "\n")
    renderers = _capture_visualizations(monkeypatch)
    previous_threads = torch.get_num_threads()

    invocation = CliRunner().invoke(boxmot, [*_arguments(data), *flags])

    assert invocation.exit_code == 0, (invocation.output, invocation.exception)
    assert torch.get_num_threads() == previous_threads
    assert len(renderers) == 1
    renderer = renderers[0]
    assert renderer.entered and renderer.closed
    assert renderer.options == {
        "show": "--show" in flags,
        "save": "--save" in flags,
        "class_names": {1: "car", 2: "pedestrian"},
        "video_fps": 10.0,
    }
    assert [event.sample.frame_index for event in renderer.frames] == [0, 1, 2]
    assert [event.sample.timestamp_s for event in renderer.frames] == [0.0, 0.1, 0.2]
    assert [len(event.result.tracks) for event in renderer.frames] == [2, 0, 2]
    first, empty, recovered = renderer.frames
    assert bool((first.result.detections.masks.values.sum(dim=0) > 1).any())
    assert not bool((first.result.tracks.masks.values.sum(dim=0) > 1).any())
    assert empty.result.tracks.masks.values.shape == (0, 24, 48)
    torch.testing.assert_close(first.result.tracks.track_ids, recovered.result.tracks.track_ids)
    output = data.project / "val"
    written = read_mots_results(output / "mots/0002.txt")
    for event in renderer.frames:
        sample, tracks = event.sample, event.result.tracks
        assert sample.split == "val"
        assert sample.sequence_id == "0002"
        assert sample.image_size == (24, 48)
        assert sample.frame.image.shape == (3, 24, 48)
        assert sample.frame.image.dtype == torch.uint8
        assert sample.frame.image.is_contiguous()
        assert sample.frame.timestamp_s == sample.timestamp_s
        assert sample.frame.frame_index == sample.frame_index
        assert sample.frame.sample_id == sample.sample_id == tracks.sample_id
        assert sample.image_ref == (data.reader_paths["images"] / f"{sample.frame_index:06d}.png").as_uri()
        rows = written.get(sample.frame_index, ())
        assert tracks.track_ids.tolist() == [row.track_id for row in rows]
        assert tracks.class_ids.tolist() == [row.class_id for row in rows]
        for index, row in enumerate(rows):
            np.testing.assert_array_equal(tracks.masks.values[index].numpy(), mask_utils.decode(row.rle).astype(bool))
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["videos"] == (["videos/0002.mp4"] if "--save" in flags else [])


def test_visualization_reloads_selected_class_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _fixture(tmp_path)
    profiles = tmp_path / "current-best.yaml"
    profiles.write_text("car: {det_thresh: 0.99}\npedestrian: {}\n")
    renderers = _capture_visualizations(monkeypatch)

    invocation = CliRunner().invoke(boxmot, [*_arguments(data), "--class-config", str(profiles), "--save"])

    assert invocation.exit_code == 0, (invocation.output, invocation.exception)
    assert [event.result.tracks.class_ids.tolist() for event in renderers[0].frames] == [[2], [], [2]]
    manifest = json.loads((data.project / "val/run.json").read_text())
    assert manifest["tracker_profiles"]["1"]["det_thresh"] == 0.99
    assert manifest["tracker_profiles"]["2"]["det_thresh"] == 0.9


def test_default_evaluation_never_decodes_rgb_or_constructs_visualization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import boxmot.datasets.readers.images as images

    data = _fixture(tmp_path)
    renderers = _capture_visualizations(monkeypatch)

    def reject_rgb(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise AssertionError("Default sensor evaluation must not decode RGB images")

    monkeypatch.setattr(images, "read_rgb_chw_uint8", reject_rgb)

    invocation = CliRunner().invoke(boxmot, _arguments(data))

    assert invocation.exit_code == 0, (invocation.output, invocation.exception)
    assert renderers == []
    manifest = json.loads((data.project / "val/run.json").read_text())
    assert manifest["visualization"] == {"show": False, "save": False, "show_3d": False, "video_fps": 10.0}
    assert "videos" not in manifest


def test_visualization_failure_closes_renderer_restores_threads_and_marks_failed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingVisualization(_RecordingVisualization):
        """Fail after recording a real blank frame to exercise resource cleanup."""

        def __call__(self, replayed: Any) -> None:
            super().__call__(replayed)
            if replayed.sample.frame_index == 1:
                raise RuntimeError("visualization callback failed")

    data = _fixture(tmp_path)
    renderers = _capture_visualizations(monkeypatch, FailingVisualization)
    previous_threads = torch.get_num_threads()

    invocation = CliRunner().invoke(boxmot, [*_arguments(data), "--show"])

    assert invocation.exit_code == 1
    assert "visualization callback failed" in invocation.output
    assert invocation.output.count("Traceback (most recent call last)") == 1
    assert renderers[0].entered and renderers[0].closed
    assert [event.sample.frame_index for event in renderers[0].frames] == [0, 1]
    assert torch.get_num_threads() == previous_threads
    manifest = json.loads((data.project / "val/run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"] == "visualization callback failed"
    assert not (data.project / "val/metrics.json").exists()
