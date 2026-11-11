"""Engine-owned tracking frame sources.

Domain components intentionally know nothing about files, cameras, streams, or
OpenCV's BGR convention.  This module is the boundary that turns those inputs
into canonical RGB :class:`boxmot.structures.Frame` objects.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from glob import glob
from math import isfinite
from numbers import Real
from pathlib import Path
from threading import Condition, Event, Thread, current_thread
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
import torch
from typing_extensions import Self

from boxmot.engine.tracking.timestamps import SourceTimestamps
from boxmot.structures import Frame

IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})


def is_appledouble_file(path: str | Path) -> bool:
    """Return whether a discovered file is a macOS AppleDouble sidecar."""

    return Path(path).name.startswith("._")


@runtime_checkable
class FrameSource(Protocol):
    """Iterable source of canonical frames."""

    def __iter__(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *_exc: object) -> None: ...


def _sequence_id(source: str) -> str:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"source-{digest}"


def frame_from_bgr(
    image: np.ndarray,
    *,
    sample_id: str,
    sequence_id: str,
    frame_index: int,
    timestamp_s: float | None,
    source_uri: str,
) -> Frame:
    """Convert an OpenCV BGR image into a canonical RGB frame."""

    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        shape = getattr(image, "shape", None)
        dtype = getattr(image, "dtype", None)
        raise ValueError(f"OpenCV frames must be uint8 HWC images with three channels; got {shape}, {dtype}.")
    rgb = np.ascontiguousarray(image[..., ::-1])
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    return Frame(
        image=tensor,
        sample_id=sample_id,
        sequence_id=sequence_id,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        source_uri=source_uri,
    )


class _BaseFrameSource:
    """Context-manager defaults shared by concrete sources."""

    def close(self) -> None:
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class ImageSource(_BaseFrameSource):
    """A single image file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def __iter__(self) -> Iterator[Frame]:
        image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image: {self.path}")
        uri = self.path.as_uri()
        yield frame_from_bgr(
            image,
            sample_id=self.path.name,
            sequence_id=_sequence_id(uri),
            frame_index=0,
            timestamp_s=None,
            source_uri=uri,
        )


@dataclass(frozen=True, slots=True)
class _CapturedVideoFrame:
    """Capture metadata stays attached to its pixels before RGB conversion."""

    image: np.ndarray
    source_index: int
    frame_index: int
    timestamp_s: float | None


class _LatestCameraReader:
    """Drain a camera on its owning thread with one replaceable pending frame."""

    _JOIN_TIMEOUT_S = 1.0

    def __init__(self, read_frames: Callable[[Event], Iterator[_CapturedVideoFrame]]) -> None:
        self._read_frames = read_frames
        self._stop = Event()
        self._condition = Condition()
        self._pending: _CapturedVideoFrame | None = None
        self._error: BaseException | None = None
        self._done = False
        self._thread = Thread(target=self._run, name="boxmot-camera-reader", daemon=True)

    @property
    def alive(self) -> bool:
        """Whether capture ownership still belongs to the reader thread."""
        return self._thread.is_alive()

    def start(self) -> None:
        """Start capture after the owner has registered this reader for cleanup."""
        self._thread.start()

    def _run(self) -> None:
        frames = self._read_frames(self._stop)
        try:
            for frame in frames:
                if self._stop.is_set():
                    break
                # Some capture backends reuse their read buffer. Own the raw
                # pixels before the next read, including while prompts pause.
                owned = replace(frame, image=frame.image.copy())
                with self._condition:
                    if self._stop.is_set():
                        break
                    self._pending = owned
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
        finally:
            try:
                frames.close()
            except BaseException as error:
                with self._condition:
                    if self._error is None:
                        self._error = error
            finally:
                with self._condition:
                    self._done = True
                    self._condition.notify_all()

    def next_frame(self) -> _CapturedVideoFrame | None:
        """Wait without polling; deliver the final pending frame before EOF/errors."""
        with self._condition:
            self._condition.wait_for(lambda: self._pending is not None or self._done or self._stop.is_set())
            if self._stop.is_set():
                return None
            if self._pending is not None:
                frame, self._pending = self._pending, None
                return frame
            if self._error is not None:
                raise self._error
            return None

    def close(self) -> None:
        """Stop and join briefly; an OS-blocked read retains capture ownership."""
        self._stop.set()
        with self._condition:
            self._pending = None
            self._condition.notify_all()
        if self._thread.ident is not None and self._thread is not current_thread():
            self._thread.join(timeout=self._JOIN_TIMEOUT_S)


class VideoSource(_BaseFrameSource):
    """A video, URL, or webcam with bounded reconnect handling.

    Increasing media PTS supplies timestamps, including a zero origin when FPS
    is available. Missing or non-increasing PTS falls back to nominal FPS,
    accounting for skipped source frames. Timing availability is fixed by the
    first eligible capture: zero PTS with unknown FPS remains untimed throughout.
    A stream established from positive PTS without FPS must keep increasing.
    Processing time and reconnect delays never become capture timestamps.
    Local cameras discard startup frames for ``camera_warmup_s`` after their
    first successful read, allowing automatic exposure to adjust before a
    consumer pauses or tracks the first emitted frame. Files and URLs retain
    their first frame. Warmup repeats after a camera reconnect.

    With ``latest_camera_frame=True``, local cameras keep reading while the
    consumer works and replace any unconsumed frame. Capture indices and media
    timestamps retain the resulting gaps. Files and URLs remain sequential.
    Close waits at most one second for an active read; a permanently blocked
    driver keeps its daemon thread and capture until that read returns. The
    source cannot restart while this earlier reader is still stopping.
    """

    def __init__(
        self,
        source: str | Path | int,
        *,
        stride: int = 1,
        reconnect_attempts: int = 0,
        reconnect_backoff_s: float = 0.5,
        camera_warmup_s: float = 2.0,
        latest_camera_frame: bool = False,
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be at least one")
        if reconnect_attempts < 0 or reconnect_backoff_s < 0:
            raise ValueError("reconnect settings must be non-negative")
        if isinstance(camera_warmup_s, bool) or not isinstance(camera_warmup_s, Real):
            raise TypeError("camera_warmup_s must be a finite, non-negative number")
        if not isfinite(camera_warmup_s) or camera_warmup_s < 0:
            raise ValueError("camera_warmup_s must be a finite, non-negative number")
        if not isinstance(latest_camera_frame, bool):
            raise TypeError("latest_camera_frame must be a bool")
        self.source = int(source) if isinstance(source, int) else str(source)
        self.stride = int(stride)
        self.reconnect_attempts = int(reconnect_attempts)
        self.reconnect_backoff_s = float(reconnect_backoff_s)
        self.camera_warmup_s = float(camera_warmup_s)
        self.latest_camera_frame = latest_camera_frame
        self._capture: cv2.VideoCapture | None = None
        self._latest_reader: _LatestCameraReader | None = None

    @property
    def source_uri(self) -> str:
        if isinstance(self.source, int):
            return f"camera:{self.source}"
        if "://" in self.source:
            return self.source
        return Path(self.source).expanduser().resolve().as_uri()

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            raise OSError(f"Could not open video source: {self.source}")
        if self.latest_camera_frame and isinstance(self.source, int):
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                pass  # Backend support varies; continuous draining does the work.
        return capture

    def __iter__(self) -> Iterator[Frame]:
        self.close()
        if self._latest_reader is not None:
            raise RuntimeError("The previous camera read is still stopping; the source cannot restart yet.")
        sequence_id = _sequence_id(self.source_uri)
        if self.latest_camera_frame and isinstance(self.source, int):
            reader = _LatestCameraReader(self._read_frames)
            self._latest_reader = reader
            reader.start()
            try:
                while (captured := reader.next_frame()) is not None:
                    yield self._canonical_frame(captured, sequence_id)
            finally:
                reader.close()
                if self._latest_reader is reader and not reader.alive:
                    self._latest_reader = None
        else:
            frames = self._read_frames()
            try:
                for captured in frames:
                    yield self._canonical_frame(captured, sequence_id)
            finally:
                frames.close()

    def _canonical_frame(self, captured: _CapturedVideoFrame, sequence_id: str) -> Frame:
        """Convert only frames consumed by tracking, leaving skipped captures raw."""
        return frame_from_bgr(
            captured.image,
            sample_id=f"{sequence_id}:{captured.source_index:012d}",
            sequence_id=sequence_id,
            frame_index=captured.frame_index,
            timestamp_s=captured.timestamp_s,
            source_uri=self.source_uri,
        )

    def _read_frames(self, stop: Event | None = None) -> Iterator[_CapturedVideoFrame]:
        """Own all capture calls on the iterator's thread, including release."""
        capture = self._open()
        self._capture = capture
        source_index = 0
        emitted_index = 0
        reconnects = 0
        warming_up = isinstance(self.source, int) and self.camera_warmup_s > 0.0
        warmup_started: float | None = None
        timestamps: SourceTimestamps | None = None
        try:
            while stop is None or not stop.is_set():
                ok, image = capture.read()
                if stop is not None and stop.is_set():
                    break
                if not ok:
                    if reconnects >= self.reconnect_attempts:
                        break
                    reconnects += 1
                    capture.release()
                    self._capture = None
                    delay = self.reconnect_backoff_s * reconnects
                    if stop is None:
                        time.sleep(delay)
                    elif stop.wait(delay):
                        break
                    capture = self._open()
                    self._capture = capture
                    warming_up = isinstance(self.source, int) and self.camera_warmup_s > 0.0
                    warmup_started = None
                    continue
                reconnects = 0
                current_index = source_index
                source_index += 1
                if warming_up:
                    now = time.monotonic()
                    if warmup_started is None:
                        warmup_started = now
                    if now - warmup_started < self.camera_warmup_s:
                        continue
                    warming_up = False
                if current_index % self.stride:
                    continue
                position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                if timestamps is None:
                    timestamps = SourceTimestamps(float(capture.get(cv2.CAP_PROP_FPS)))
                timestamp_s = timestamps.resolve(position_ms, frame_index=current_index)
                yield _CapturedVideoFrame(image, current_index, emitted_index, timestamp_s)
                emitted_index += 1
        finally:
            if self._capture is capture:
                capture.release()
                self._capture = None

    def close(self) -> None:
        if self._latest_reader is not None:
            self._latest_reader.close()
            if not self._latest_reader.alive:
                self._latest_reader = None
            return
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class DirectorySource(_BaseFrameSource):
    """Deterministic, non-recursive image/video directory source."""

    def __init__(self, path: str | Path, *, stride: int = 1) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)
        self.stride = int(stride)
        if self.stride < 1:
            raise ValueError("stride must be at least one")
        self._active: FrameSource | None = None

    def __iter__(self) -> Iterator[Frame]:
        directory_uri = self.path.as_uri()
        image_sequence = _sequence_id(directory_uri)
        image_index = 0
        for child in sorted(item for item in self.path.iterdir() if item.is_file() and not is_appledouble_file(item)):
            if child.suffix.lower() in IMAGE_EXTENSIONS:
                if image_index % self.stride == 0:
                    image = cv2.imread(str(child), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f"Could not decode image: {child}")
                    yield frame_from_bgr(
                        image,
                        sample_id=child.relative_to(self.path).as_posix(),
                        sequence_id=image_sequence,
                        frame_index=image_index // self.stride,
                        timestamp_s=None,
                        source_uri=child.as_uri(),
                    )
                image_index += 1
                continue
            elif child.suffix.lower() in VIDEO_EXTENSIONS:
                self._active = VideoSource(child, stride=self.stride)
            else:
                continue
            yield from self._active
            self._active.close()
            self._active = None

    def close(self) -> None:
        if self._active is not None:
            self._active.close()
            self._active = None


class CompositeSource(_BaseFrameSource):
    """Ordered composition used for glob expansion."""

    def __init__(self, sources: list[FrameSource]) -> None:
        self.sources = sources

    def __iter__(self) -> Iterator[Frame]:
        for source in self.sources:
            self._active = source
            yield from source
            source.close()
        self._active = None

    def close(self) -> None:
        active = getattr(self, "_active", None)
        if active is not None:
            active.close()
        for source in self.sources:
            source.close()


def create_frame_source(
    source: str | Path | int,
    *,
    stride: int = 1,
    reconnect_attempts: int = 0,
    reconnect_backoff_s: float = 0.5,
    camera_warmup_s: float = 2.0,
    latest_camera_frame: bool = False,
) -> FrameSource:
    """Create an input source; local cameras warm up before yielding frames.

    ``camera_warmup_s=0`` disables camera startup draining. Files and URLs do
    not warm up, so recorded frames are preserved regardless of this setting.
    ``latest_camera_frame=True`` continuously drains local cameras and retains
    only their newest pending frame; files and URLs still yield every frame.
    """

    if isinstance(source, int) or (isinstance(source, str) and source.isdecimal()):
        camera = int(source)
        return VideoSource(
            camera,
            stride=stride,
            reconnect_attempts=reconnect_attempts,
            reconnect_backoff_s=reconnect_backoff_s,
            camera_warmup_s=camera_warmup_s,
            latest_camera_frame=latest_camera_frame,
        )
    source_text = str(source)
    if any(token in source_text for token in "*?["):
        matches = sorted(glob(source_text))
        if not matches:
            raise FileNotFoundError(source_text)
        return CompositeSource(
            [create_frame_source(match, stride=stride, camera_warmup_s=camera_warmup_s) for match in matches]
        )
    if "://" in source_text:
        return VideoSource(
            source_text,
            stride=stride,
            reconnect_attempts=reconnect_attempts,
            reconnect_backoff_s=reconnect_backoff_s,
            camera_warmup_s=camera_warmup_s,
        )
    path = Path(source_text).expanduser().resolve()
    if path.is_dir():
        return DirectorySource(path, stride=stride)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return ImageSource(path)
    if path.is_file():
        return VideoSource(path, stride=stride, camera_warmup_s=camera_warmup_s)
    raise FileNotFoundError(path)


__all__ = (
    "CompositeSource",
    "DirectorySource",
    "FrameSource",
    "ImageSource",
    "VideoSource",
    "create_frame_source",
    "frame_from_bgr",
    "is_appledouble_file",
)
