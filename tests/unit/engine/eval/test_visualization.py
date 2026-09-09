"""Visual replay preserves gaps without adding tracker observations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boxmot.engine.eval import visualization
from boxmot.engine.eval.visualization import ReplayVisualization


def _replay(index, timestamp, sequence="seq"):
    return SimpleNamespace(
        sample=SimpleNamespace(sequence_id=sequence, frame_index=index, timestamp_s=timestamp, frame=object()),
        result=index + 1,
    )


@pytest.fixture
def rendering(monkeypatch):
    writers = []
    shown = []
    destroyed = []
    clock = [0.0]
    keys = []

    class Writer:
        def __init__(self, path, codec, fps, dimensions):
            self.path, self.fps, self.dimensions = path, fps, dimensions
            self.frames = []
            self.images = []
            self.released = False
            writers.append(self)

        def isOpened(self):
            return True

        def write(self, image):
            assert image.shape[:2] == self.dimensions[::-1]
            self.frames.append(int(image[0, 0, 0]))
            self.images.append(image.copy())

        def release(self):
            self.released = True

    def wait(delay):
        clock[0] += delay / 1000
        return keys.pop(0) if keys else -1

    monkeypatch.setattr(visualization.cv2, "VideoWriter", Writer)
    monkeypatch.setattr(
        visualization, "render_result", lambda frame, result, **kw: np.full((8, 10, 3), result, np.uint8)
    )
    monkeypatch.setattr(visualization.cv2, "putText", lambda *args: None)
    monkeypatch.setattr(visualization.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(visualization.cv2, "resizeWindow", lambda *args: None)
    monkeypatch.setattr(visualization.cv2, "imshow", lambda name, image: shown.append((clock[0], int(image[0, 0, 0]))))
    monkeypatch.setattr(visualization.cv2, "destroyWindow", lambda name: destroyed.append(name))
    monkeypatch.setattr(visualization.cv2, "waitKey", wait)
    monkeypatch.setattr(visualization.time, "perf_counter", lambda: clock[0])
    return SimpleNamespace(writers=writers, shown=shown, destroyed=destroyed, clock=clock, keys=keys)


def test_saved_video_holds_previous_observation_through_gap(tmp_path, rendering):
    with ReplayVisualization(tmp_path, show=False, save=True) as consumer:
        consumer(_replay(0, 10.0))
        consumer(_replay(1, 10.1))
        consumer(_replay(2, 10.2))
    writer = rendering.writers[0]
    assert writer.frames == [1, 1, 1, 2, 2, 2, 3]
    assert writer.fps == 30.0
    assert writer.released
    assert consumer.video_paths == (tmp_path / "videos" / "seq.mp4",)
    assert rendering.clock[0] == 0.0  # Saving does not sleep to source time.
    consumer.close()
    assert writer.frames == [1, 1, 1, 2, 2, 2, 3]


def test_missing_times_use_one_output_frame_per_observation(tmp_path, rendering):
    with ReplayVisualization(tmp_path, show=False, save=True) as consumer:
        for index in range(4):
            consumer(_replay(index, None))
    assert rendering.writers[0].frames == [1, 2, 3, 4]


def test_ten_fps_grid_writes_each_kitti_observation_once(tmp_path: Path, rendering: SimpleNamespace) -> None:
    """A matching output rate preserves one video frame per 10 Hz image."""
    with ReplayVisualization(tmp_path, show=False, save=True, video_fps=10.0) as consumer:
        for index in range(6):
            consumer(_replay(index, 10.0 + index / 10.0))
    writer = rendering.writers[0]
    assert writer.fps == 10.0
    assert writer.frames == [1, 2, 3, 4, 5, 6]
    assert writer.released


@pytest.mark.parametrize("shape", ((9, 11), (8, 11), (9, 10)))
def test_odd_video_dimensions_pad_edges_without_changing_render_or_preview(
    tmp_path: Path, rendering: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, shape: tuple[int, int]
) -> None:
    """Encoding retains every source pixel and pads only the bottom/right edges."""
    height, width = shape
    originals = [np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3) + index for index in range(2)]
    original_pixels = [image.copy() for image in originals]
    previews = []
    monkeypatch.setattr(visualization, "render_result", lambda frame, result, **kwargs: originals[result - 1])
    monkeypatch.setattr(visualization.cv2, "imshow", lambda name, image: previews.append(image.copy()))
    with ReplayVisualization(tmp_path, show=True, save=True, video_fps=10.0) as consumer:
        consumer(_replay(0, 0.0))
        consumer(_replay(1, 0.1))

    writer = rendering.writers[0]
    assert writer.dimensions == (width + width % 2, height + height % 2)
    assert len(writer.images) == len(previews) == 2
    for original, expected, preview, encoded in zip(originals, original_pixels, previews, writer.images, strict=True):
        np.testing.assert_array_equal(original, expected)
        np.testing.assert_array_equal(preview, expected)
        np.testing.assert_array_equal(encoded[:height, :width], expected)
        np.testing.assert_array_equal(encoded[-1, :width], expected[-1])
        np.testing.assert_array_equal(encoded[:height, -1], expected[:, -1])
        np.testing.assert_array_equal(encoded[-1, -1], expected[-1, -1])


@pytest.mark.parametrize("rate", (0, -10, float("nan"), float("inf"), float("-inf"), True, False, "10", None))
def test_invalid_video_rates_fail_before_opening_outputs(
    tmp_path: Path, rendering: SimpleNamespace, rate: object
) -> None:
    with pytest.raises(ValueError, match="video_fps.*finite positive"):
        ReplayVisualization(tmp_path, show=False, save=True, video_fps=rate)
    assert not rendering.writers


def test_video_and_preview_restart_the_clock_at_sequence_boundary(tmp_path, rendering):
    with ReplayVisualization(tmp_path, show=True, save=True) as consumer:
        consumer(_replay(0, 10.0, "a"))
        consumer(_replay(1, 10.1, "a"))
        consumer(_replay(0, 0.0, "b"))
        consumer(_replay(1, 0.1, "b"))
    assert [writer.frames for writer in rendering.writers] == [[1, 1, 1, 2], [1, 1, 1, 2]]
    assert all(writer.released for writer in rendering.writers)
    assert rendering.shown[1][0] == pytest.approx(0.1, abs=0.001)
    assert rendering.shown[3][0] - rendering.shown[2][0] == pytest.approx(0.1, abs=0.001)
    assert rendering.destroyed == ["BoxMOT evaluation"]


def test_quit_closes_preview_but_keeps_accepting_and_saving_results(tmp_path, rendering):
    rendering.keys.extend([-1, ord("q")])
    with ReplayVisualization(tmp_path, show=True, save=True) as consumer:
        consumer(_replay(0, 0.0))
        consumer(_replay(1, 0.1))
        consumer(_replay(2, 0.2))
    assert len(rendering.shown) == 1
    assert rendering.destroyed == ["BoxMOT evaluation"]
    assert rendering.writers[0].frames == [1, 1, 1, 2, 2, 2, 3]


@pytest.mark.parametrize("timestamp", [0.0, -1.0, None, float("nan"), float("inf")])
def test_bad_capture_time_fails_and_releases_outputs(tmp_path, rendering, timestamp):
    with pytest.raises(ValueError), ReplayVisualization(tmp_path, show=False, save=True) as consumer:
        consumer(_replay(0, 0.0))
        consumer(_replay(1, timestamp))
    assert rendering.writers[0].released


def test_non_grid_capture_time_is_delivered_at_next_output_tick(tmp_path, rendering):
    with ReplayVisualization(tmp_path, show=False, save=True) as consumer:
        consumer(_replay(0, 0.0))
        consumer(_replay(1, 0.05))
        consumer(_replay(2, 0.10))
    assert rendering.writers[0].frames == [1, 1, 2, 3]


def test_original_images_are_required(tmp_path, rendering):
    replayed = _replay(0, 0.0)
    replayed.sample.frame = None
    with ReplayVisualization(tmp_path, show=True, save=True) as consumer:
        with pytest.raises(ValueError, match="original frame pixels"):
            consumer(replayed)
    assert not rendering.writers


def test_rendering_failure_releases_video_and_window(tmp_path, rendering, monkeypatch):
    with (
        pytest.raises(RuntimeError, match="render failed"),
        ReplayVisualization(tmp_path, show=True, save=True) as consumer,
    ):
        consumer(_replay(0, 0.0))

        def fail(*args, **kwargs):
            raise RuntimeError("render failed")

        monkeypatch.setattr(visualization, "render_result", fail)
        consumer(_replay(1, 0.1))
    assert rendering.writers[0].released
    assert rendering.destroyed == ["BoxMOT evaluation"]
