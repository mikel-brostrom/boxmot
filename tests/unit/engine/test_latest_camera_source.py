"""Camera capture stays current while consumers pause, with deterministic handoffs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Condition, Event, get_ident
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from boxmot.engine.tracking import sources
from boxmot.engine.tracking.sources import VideoSource, create_frame_source


class Capture:
    """An explicitly paced camera that deliberately reuses its image buffer."""

    def __init__(self, *, fps: float = 25.0, clock: SimpleNamespace | None = None) -> None:
        self.fps = fps
        self.clock = clock
        self.commands: Queue[tuple[int, float] | BaseException | None] = Queue()
        self.image = np.zeros((3, 4, 3), dtype=np.uint8)
        self.position_ms = 0.0
        self.read_count = 0
        self.condition = Condition()
        self.released = Event()
        self.in_read = False
        self.call_threads: set[int] = set()
        self.settings: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        self.call_threads.add(get_ident())
        return True

    def set(self, property_id: int, value: float) -> bool:
        self.call_threads.add(get_ident())
        self.settings.append((property_id, value))
        return False  # Correctness must not rely on driver buffer-size support.

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.call_threads.add(get_ident())
        self.in_read = True
        with self.condition:
            self.read_count += 1
            self.condition.notify_all()
        try:
            command = self.commands.get(timeout=5)
            if isinstance(command, BaseException):
                raise command
            if command is None:
                return False, None
            value, self.position_ms = command
            self.image.fill(value)
            if self.clock is not None:
                self.clock.now += 0.5
            return True, self.image
        finally:
            self.in_read = False

    def get(self, property_id: int) -> float:
        self.call_threads.add(get_ident())
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return self.position_ms
        assert property_id == cv2.CAP_PROP_FPS
        return self.fps

    def release(self) -> None:
        self.call_threads.add(get_ident())
        assert not self.in_read, "capture released concurrently with read"
        self.released.set()

    def frame(self, value: int, position_ms: float = 0.0) -> None:
        self.commands.put((value, position_ms))

    def wait_for_read(self, count: int) -> None:
        """A following read proves that the preceding frame was published."""
        with self.condition:
            assert self.condition.wait_for(lambda: self.read_count >= count, timeout=5)


@pytest.mark.parametrize(
    "fps, positions, expected_timestamp",
    [
        (25.0, [0.0, 0.0, 0.0, 0.0], 0.12),
        (25.0, [0.0, 40.0, 110.0, 230.0], 0.23),
        (0.0, [0.0, 40.0, 110.0, 230.0], None),
    ],
)
def test_slow_consumer_gets_latest_owned_pixels_with_capture_metadata(
    monkeypatch: pytest.MonkeyPatch, fps: float, positions: list[float], expected_timestamp: float | None
) -> None:
    capture = Capture(fps=fps)
    capture.frame(0, positions[0])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)
    converted: list[str] = []
    convert = sources.frame_from_bgr

    def record_conversion(image, **kwargs):
        converted.append(kwargs["sample_id"])
        return convert(image, **kwargs)

    monkeypatch.setattr(sources, "frame_from_bgr", record_conversion)
    source = VideoSource(0, camera_warmup_s=0, latest_camera_frame=True)
    iterator = iter(source)
    try:
        selected = next(iterator)
        capture.wait_for_read(2)
        for index in range(1, 4):
            capture.frame(index, positions[index])
        capture.wait_for_read(5)
        latest = next(iterator)

        assert int(selected.image[0, 0, 0]) == 0
        assert int(latest.image[0, 0, 0]) == 3
        assert selected.frame_index == 0 and latest.frame_index == 3
        assert latest.sample_id.endswith(":000000000003")
        if expected_timestamp is None:
            assert latest.timestamp_s is None
        else:
            assert latest.timestamp_s == pytest.approx(expected_timestamp)
        assert latest.source_uri == "camera:0" and latest.sequence_id == selected.sequence_id
        assert converted == [selected.sample_id, latest.sample_id]
        assert capture.settings == [(cv2.CAP_PROP_BUFFERSIZE, 1)]
        assert len(capture.call_threads) == 1 and get_ident() not in capture.call_threads
    finally:
        capture.commands.put(None)
        iterator.close()
        source.close()
    assert capture.released.is_set()
    assert source._latest_reader is None


def test_raw_frame_is_owned_while_consumer_converts_and_camera_reuses_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = Capture()
    capture.frame(0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)
    source = VideoSource(0, camera_warmup_s=0, latest_camera_frame=True)
    iterator = iter(source)
    converting, resume_conversion = Event(), Event()
    convert = sources.frame_from_bgr

    def paused_conversion(image, **kwargs):
        converting.set()
        assert resume_conversion.wait(timeout=5)
        return convert(image, **kwargs)

    try:
        next(iterator)
        capture.frame(1)
        capture.wait_for_read(3)
        monkeypatch.setattr(sources, "frame_from_bgr", paused_conversion)
        with ThreadPoolExecutor(max_workers=1) as consumer:
            pending = consumer.submit(next, iterator)
            try:
                assert converting.wait(timeout=5)
                capture.frame(2)
                capture.wait_for_read(4)
            finally:
                resume_conversion.set()
            selected = pending.result(timeout=5)
        assert int(selected.image[0, 0, 0]) == 1
        assert selected.sample_id.endswith(":000000000001")
    finally:
        resume_conversion.set()
        capture.commands.put(None)
        iterator.close()
        source.close()


def test_latest_capture_preserves_warmup_stride_and_skipped_source_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(sources.time, "monotonic", lambda: clock.now)
    capture = Capture(fps=2.0, clock=clock)
    for index in range(3):
        capture.frame(index, index * 500.0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)
    source = VideoSource(0, camera_warmup_s=1, stride=2, latest_camera_frame=True)
    iterator = iter(source)
    try:
        first = next(iterator)
        capture.wait_for_read(4)
        for index in range(3, 7):
            capture.frame(index, index * 500.0)
        capture.wait_for_read(8)
        latest = next(iterator)
        assert [int(frame.image[0, 0, 0]) for frame in (first, latest)] == [2, 6]
        assert [frame.frame_index for frame in (first, latest)] == [0, 2]
        assert [frame.timestamp_s for frame in (first, latest)] == pytest.approx([1.0, 3.0])
        assert latest.sample_id.endswith(":000000000006")
    finally:
        capture.commands.put(None)
        iterator.close()
        source.close()


@pytest.mark.parametrize("end", [None, RuntimeError("camera read failed")])
def test_final_pending_frame_is_yielded_before_eof_or_reader_error(
    monkeypatch: pytest.MonkeyPatch, end: BaseException | None
) -> None:
    capture = Capture()
    capture.frame(0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)
    source = VideoSource(0, camera_warmup_s=0, latest_camera_frame=True)
    iterator = iter(source)
    try:
        next(iterator)
        capture.frame(1, 40.0)
        capture.commands.put(end)
        assert capture.released.wait(timeout=5)
        final = next(iterator)
        assert final.frame_index == 1 and int(final.image[0, 0, 0]) == 1
        if end is None:
            with pytest.raises(StopIteration):
                next(iterator)
        else:
            with pytest.raises(RuntimeError, match="camera read failed"):
                next(iterator)
    finally:
        iterator.close()
        source.close()
    assert source._latest_reader is None


def test_reconnect_rearms_warmup_and_preserves_capture_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(sources.time, "monotonic", lambda: clock.now)
    captures = [Capture(fps=2.0, clock=clock) for _ in range(3)]
    for index in range(3):
        captures[0].frame(index, index * 500.0)
        captures[1].frame(index + 10, index * 500.0)
    captures[1].commands.put(None)
    captures[2].commands.put(None)
    pending = iter(captures)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: next(pending))
    source = VideoSource(0, camera_warmup_s=1, latest_camera_frame=True, reconnect_attempts=1, reconnect_backoff_s=0)
    iterator = iter(source)
    try:
        first = next(iterator)
        captures[0].commands.put(None)
        assert captures[2].released.wait(timeout=5)
        remaining = list(iterator)
        assert len(remaining) == 1
        latest = remaining[0]
        assert [int(frame.image[0, 0, 0]) for frame in (first, latest)] == [2, 12]
        assert [frame.frame_index for frame in (first, latest)] == [0, 1]
        assert [frame.timestamp_s for frame in (first, latest)] == pytest.approx([1.0, 2.5])
        assert latest.sample_id.endswith(":000000000005")
        assert all(capture.settings == [(cv2.CAP_PROP_BUFFERSIZE, 1)] for capture in captures)
    finally:
        for capture in captures:
            capture.commands.put(None)
        iterator.close()
        source.close()
    assert all(capture.released.is_set() for capture in captures)


def test_close_interrupts_reconnect_backoff_and_joins_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = Capture()
    capture.frame(0)
    opened = []

    def open_capture(_source):
        opened.append(get_ident())
        assert len(opened) == 1, "cancelled reconnect opened another camera"
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", open_capture)
    source = VideoSource(0, camera_warmup_s=0, latest_camera_frame=True, reconnect_attempts=3, reconnect_backoff_s=3600)
    iterator = iter(source)
    try:
        next(iterator)
        capture.commands.put(None)
        assert capture.released.wait(timeout=5)
        source.close()
        assert source._latest_reader is None
    finally:
        iterator.close()
        source.close()


def test_blocked_read_is_not_released_or_restarted_from_another_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sources._LatestCameraReader, "_JOIN_TIMEOUT_S", 0.0)
    capture = Capture()
    capture.frame(0)
    opened = []

    def open_capture(_source):
        opened.append(get_ident())
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", open_capture)
    source = VideoSource(0, camera_warmup_s=0, latest_camera_frame=True)
    iterator = iter(source)
    reader = None
    try:
        next(iterator)
        capture.wait_for_read(2)
        reader = source._latest_reader
        source.close()
        assert reader is not None and reader.alive
        assert source._latest_reader is reader
        assert not capture.released.is_set()
        with pytest.raises(RuntimeError, match="still stopping"):
            next(iter(source))
        assert len(opened) == 1
    finally:
        capture.commands.put(None)
        assert capture.released.wait(timeout=5)
        if reader is not None:
            reader._thread.join(timeout=5)
        iterator.close()
        source.close()
    assert source._latest_reader is None
    assert capture.call_threads == set(opened)


@pytest.mark.parametrize("source_name", ["video.mp4", "https://example.test/video"])
def test_latest_camera_option_keeps_files_and_urls_sequential(
    monkeypatch: pytest.MonkeyPatch, source_name: str
) -> None:
    capture = Capture()
    for index in range(3):
        capture.frame(index, index * 40.0)
    capture.commands.put(None)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: capture)
    frames = list(VideoSource(source_name, latest_camera_frame=True))
    assert [frame.frame_index for frame in frames] == [0, 1, 2]
    assert [int(frame.image[0, 0, 0]) for frame in frames] == [0, 1, 2]
    assert capture.call_threads == {get_ident()}
    assert not capture.settings


@pytest.mark.parametrize("camera", [0, "0"])
def test_source_factory_enables_latest_frames_only_when_requested(camera: int | str) -> None:
    source = create_frame_source(camera, latest_camera_frame=True)
    assert isinstance(source, VideoSource) and source.source == 0 and source.latest_camera_frame
    assert not create_frame_source(camera).latest_camera_frame
