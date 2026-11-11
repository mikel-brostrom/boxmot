from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from boxmot.engine.tracking import sources
from boxmot.engine.tracking.sources import (
    DirectorySource,
    ImageSource,
    VideoSource,
    create_frame_source,
    frame_from_bgr,
)


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


def test_directory_source_ignores_appledouble_sidecars(tmp_path) -> None:
    assert cv2.imwrite(str(tmp_path / "001.jpg"), np.zeros((2, 2, 3), dtype=np.uint8))
    assert cv2.imwrite(str(tmp_path / ".002.jpg"), np.ones((2, 2, 3), dtype=np.uint8))
    (tmp_path / "._001.jpg").write_bytes(b"AppleDouble metadata, not a JPEG")

    frames = list(DirectorySource(tmp_path))

    assert [frame.sample_id for frame in frames] == [".002.jpg", "001.jpg"]


class _Capture:
    """Small OpenCV capture double with independently controlled PTS and FPS."""

    def __init__(self, positions_ms: list[float], fps: float) -> None:
        self.positions_ms = positions_ms
        self.fps = fps
        self.index = -1
        self.released = False

    def isOpened(self) -> bool:
        return not self.released

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.index += 1
        if self.index >= len(self.positions_ms):
            return False, None
        return True, np.full((2, 3, 3), self.index, dtype=np.uint8)

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return self.positions_ms[self.index]
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        raise AssertionError(f"Unexpected capture property: {property_id}")

    def release(self) -> None:
        self.released = True


@pytest.mark.parametrize("stride, expected", [(1, [0.0, 0.04, 0.11, 0.14, 0.23]), (2, [0.0, 0.11, 0.23])])
def test_video_source_preserves_zero_origin_and_variable_media_pts(
    monkeypatch: pytest.MonkeyPatch, stride: int, expected: list[float]
) -> None:
    capture = _Capture([0.0, 40.0, 110.0, 140.0, 230.0], fps=25.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)
    frames = list(VideoSource("video.mp4", stride=stride))
    assert [frame.timestamp_s for frame in frames] == pytest.approx(expected)
    assert [frame.frame_index for frame in frames] == list(range(len(expected)))
    assert capture.released


@pytest.mark.parametrize("positions", [[0.0] * 5, [np.nan, np.nan, -1.0, 0.0, 0.0]])
def test_video_source_falls_back_to_fps_including_stride(
    monkeypatch: pytest.MonkeyPatch, positions: list[float]
) -> None:
    capture = _Capture(positions, fps=25.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)
    frames = list(VideoSource("video.mp4", stride=2))
    assert [frame.timestamp_s for frame in frames] == pytest.approx([0.0, 0.08, 0.16])


@pytest.mark.parametrize("fps", [0.0, np.nan])
def test_webcam_without_initial_capture_timing_stays_untimed(monkeypatch: pytest.MonkeyPatch, fps: float) -> None:
    capture = _Capture([0.0, 0.0, 40.0, 90.0], fps=fps)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)
    frames = list(VideoSource(0, camera_warmup_s=0))
    assert [frame.timestamp_s for frame in frames] == [None] * 4
    assert all(frame.source_uri == "camera:0" for frame in frames)


def test_positive_media_pts_can_establish_timing_without_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _Capture([1000.0, 1040.0, 1190.0], fps=0.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)
    frames = list(VideoSource("stream.mp4"))
    assert [frame.timestamp_s for frame in frames] == pytest.approx([1.0, 1.04, 1.19])


@pytest.mark.parametrize("invalid_position", [1000.0, 0.0, np.nan])
def test_timed_source_without_fps_rejects_broken_media_pts(
    monkeypatch: pytest.MonkeyPatch, invalid_position: float
) -> None:
    capture = _Capture([1000.0, invalid_position], fps=0.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)
    frames = iter(VideoSource("stream.mp4"))
    assert next(frames).timestamp_s == 1.0
    with pytest.raises(ValueError, match="timestamps.*FPS"):
        next(frames)
    assert capture.released


def test_video_source_keeps_nominal_timestamps_monotonic_after_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    captures = [_Capture([0.0, 40.0], 25.0), _Capture([0.0, 40.0], 25.0), _Capture([], 25.0)]
    pending = iter(captures)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: next(pending))
    frames = list(VideoSource("https://example.test/video", reconnect_attempts=1, reconnect_backoff_s=0.0))
    assert [frame.timestamp_s for frame in frames] == pytest.approx([0.0, 0.04, 0.08, 0.12])
    assert [frame.frame_index for frame in frames] == [0, 1, 2, 3]
    assert all(capture.released for capture in captures)


class _Clock:
    """Advance only when a fake camera reads or reconnects."""

    def __init__(self) -> None:
        self.now = 10_000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _ClockedCapture(_Capture):
    """Camera reads consume elapsed time independently of reported FPS."""

    def __init__(
        self,
        clock: _Clock,
        positions_ms: list[float],
        fps: float,
        *,
        pixel_values: list[int] | None = None,
    ) -> None:
        super().__init__(positions_ms, fps)
        self.clock = clock
        self.pixel_values = pixel_values

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.clock.now += 0.5
        ok, image = super().read()
        if ok and self.pixel_values is not None:
            image.fill(self.pixel_values[self.index])
        return ok, image


@pytest.fixture
def camera_clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(sources.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(sources.time, "sleep", clock.sleep)
    return clock


def test_camera_warmup_reads_past_dark_startup_before_first_emitted_frame(
    monkeypatch: pytest.MonkeyPatch, camera_clock: _Clock
) -> None:
    capture = _ClockedCapture(
        camera_clock,
        [0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0],
        fps=2.0,
        pixel_values=[0, 10, 20, 100, 180, 200],
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)
    frames = list(VideoSource(0))

    assert [int(frame.image[0, 0, 0]) for frame in frames] == [180, 200]
    assert [frame.frame_index for frame in frames] == [0, 1]
    assert [frame.timestamp_s for frame in frames] == pytest.approx([2.0, 2.5])
    assert frames[0].sample_id.endswith(":000000000004")
    assert capture.released


@pytest.mark.parametrize("fps", [0.0, 2.0, 240.0])
def test_camera_warmup_uses_elapsed_reads_independently_of_reported_fps(
    monkeypatch: pytest.MonkeyPatch, camera_clock: _Clock, fps: float
) -> None:
    capture = _ClockedCapture(camera_clock, [0.0] * 4, fps=fps)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)

    frames = list(VideoSource(0, camera_warmup_s=1.0))

    assert [int(frame.image[0, 0, 0]) for frame in frames] == [2, 3]


def test_camera_without_timing_after_warmup_stays_untimed(
    monkeypatch: pytest.MonkeyPatch, camera_clock: _Clock
) -> None:
    capture = _ClockedCapture(camera_clock, [0.0] * 5 + [2500.0, 3000.0], fps=0.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)

    frames = list(VideoSource(0))

    assert len(frames) == 3
    assert [frame.timestamp_s for frame in frames] == [None] * 3


def test_camera_fps_fallback_counts_discarded_frames_without_using_wall_time(
    monkeypatch: pytest.MonkeyPatch, camera_clock: _Clock
) -> None:
    capture = _ClockedCapture(camera_clock, [np.nan] * 6, fps=2.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)

    frames = list(VideoSource(0))

    assert [frame.timestamp_s for frame in frames] == pytest.approx([2.0, 2.5])


def test_camera_stride_counts_warmup_frames_but_emitted_indices_start_at_zero(
    monkeypatch: pytest.MonkeyPatch, camera_clock: _Clock
) -> None:
    capture = _ClockedCapture(camera_clock, [index * 500.0 for index in range(10)], fps=2.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)

    frames = list(VideoSource(0, stride=3))

    assert [int(frame.image[0, 0, 0]) for frame in frames] == [6, 9]
    assert [frame.frame_index for frame in frames] == [0, 1]
    assert [frame.timestamp_s for frame in frames] == pytest.approx([3.0, 4.5])
    assert [frame.sample_id.rsplit(":", 1)[1] for frame in frames] == ["000000000006", "000000000009"]


def test_camera_warmup_restarts_after_reconnect_without_including_backoff(
    monkeypatch: pytest.MonkeyPatch, camera_clock: _Clock
) -> None:
    captures = [
        _ClockedCapture(camera_clock, [0.0, 500.0, 1000.0], fps=2.0),
        _ClockedCapture(camera_clock, [0.0, 500.0, 1000.0], fps=2.0),
        _ClockedCapture(camera_clock, [], fps=2.0),
    ]
    pending = iter(captures)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: next(pending))

    frames = list(VideoSource(0, camera_warmup_s=1.0, reconnect_attempts=1, reconnect_backoff_s=20.0))

    assert [int(frame.image[0, 0, 0]) for frame in frames] == [2, 2]
    assert [frame.frame_index for frame in frames] == [0, 1]
    assert [frame.timestamp_s for frame in frames] == pytest.approx([1.0, 2.5])
    assert all(capture.released for capture in captures)


def test_camera_exhausted_during_warmup_releases_without_emitting_startup_frames(
    monkeypatch: pytest.MonkeyPatch, camera_clock: _Clock
) -> None:
    capture = _ClockedCapture(camera_clock, [0.0, 500.0], fps=2.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)

    assert list(VideoSource(0)) == []
    assert capture.released


def test_interrupt_during_camera_warmup_releases_capture(monkeypatch: pytest.MonkeyPatch, camera_clock: _Clock) -> None:
    capture = _ClockedCapture(camera_clock, [0.0], fps=2.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)
    original_read = capture.read

    def interrupt_after_first_read() -> tuple[bool, np.ndarray | None]:
        if capture.index >= 0:
            raise KeyboardInterrupt
        return original_read()

    monkeypatch.setattr(capture, "read", interrupt_after_first_read)
    with pytest.raises(KeyboardInterrupt):
        next(iter(VideoSource(0)))
    assert capture.released


@pytest.mark.parametrize("source", ["video.mp4", "https://example.test/video"])
def test_files_and_urls_preserve_first_frame_without_warmup_clock(monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    capture = _Capture([0.0, 40.0], fps=25.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)
    monkeypatch.setattr(sources.time, "monotonic", lambda: pytest.fail("non-camera source warmed up"))

    frames = list(VideoSource(source))

    assert [int(frame.image[0, 0, 0]) for frame in frames] == [0, 1]
    assert [frame.timestamp_s for frame in frames] == pytest.approx([0.0, 0.04])


@pytest.mark.parametrize("source", [0, "0"])
@pytest.mark.parametrize("warmup_s, expected_indices", [(2.0, [4, 5]), (0.0, list(range(6)))])
def test_camera_factory_applies_default_warmup_and_explicit_opt_out(
    monkeypatch: pytest.MonkeyPatch,
    camera_clock: _Clock,
    source: int | str,
    warmup_s: float,
    expected_indices: list[int],
) -> None:
    capture = _ClockedCapture(camera_clock, [0.0] * 6, fps=2.0)
    opened = []

    def open_camera(value: int) -> _ClockedCapture:
        opened.append(value)
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", open_camera)
    options = {} if warmup_s == 2.0 else {"camera_warmup_s": warmup_s}

    frames = list(create_frame_source(source, **options))

    assert opened == [0]
    assert [int(frame.image[0, 0, 0]) for frame in frames] == expected_indices


@pytest.mark.parametrize("warmup_s", [-1.0, np.nan, np.inf, True, "2"])
def test_invalid_camera_warmup_rejected_before_opening_capture(
    monkeypatch: pytest.MonkeyPatch, warmup_s: object
) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: pytest.fail("opened invalid camera configuration"))

    with pytest.raises((TypeError, ValueError), match="camera_warmup_s"):
        create_frame_source(0, camera_warmup_s=warmup_s)
