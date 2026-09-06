"""Engine-owned tracking frame sources.

Domain components intentionally know nothing about files, cameras, streams, or
OpenCV's BGR convention.  This module is the boundary that turns those inputs
into canonical RGB :class:`boxmot.structures.Frame` objects.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from glob import glob
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
import torch
from typing_extensions import Self

from boxmot.structures import Frame

IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})


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


class VideoSource(_BaseFrameSource):
    """A video, URL, or webcam with bounded reconnect handling."""

    def __init__(
        self,
        source: str | Path | int,
        *,
        stride: int = 1,
        reconnect_attempts: int = 0,
        reconnect_backoff_s: float = 0.5,
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be at least one")
        if reconnect_attempts < 0 or reconnect_backoff_s < 0:
            raise ValueError("reconnect settings must be non-negative")
        self.source = int(source) if isinstance(source, int) else str(source)
        self.stride = int(stride)
        self.reconnect_attempts = int(reconnect_attempts)
        self.reconnect_backoff_s = float(reconnect_backoff_s)
        self._capture: cv2.VideoCapture | None = None

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
        return capture

    def __iter__(self) -> Iterator[Frame]:
        self.close()
        self._capture = self._open()
        sequence_id = _sequence_id(self.source_uri)
        source_index = 0
        emitted_index = 0
        reconnects = 0
        try:
            while True:
                assert self._capture is not None
                ok, image = self._capture.read()
                if not ok:
                    if reconnects >= self.reconnect_attempts:
                        break
                    reconnects += 1
                    self._capture.release()
                    time.sleep(self.reconnect_backoff_s * reconnects)
                    self._capture = self._open()
                    continue
                reconnects = 0
                current_index = source_index
                source_index += 1
                if current_index % self.stride:
                    continue
                position_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
                timestamp_s = position_ms / 1000.0 if position_ms > 0 else None
                yield frame_from_bgr(
                    image,
                    sample_id=f"{sequence_id}:{current_index:012d}",
                    sequence_id=sequence_id,
                    frame_index=emitted_index,
                    timestamp_s=timestamp_s,
                    source_uri=self.source_uri,
                )
                emitted_index += 1
        finally:
            self.close()

    def close(self) -> None:
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
        for child in sorted(item for item in self.path.iterdir() if item.is_file()):
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
) -> FrameSource:
    """Create the appropriate engine source without importing model code."""

    if isinstance(source, int) or (isinstance(source, str) and source.isdecimal()):
        camera = int(source)
        return VideoSource(
            camera,
            stride=stride,
            reconnect_attempts=reconnect_attempts,
            reconnect_backoff_s=reconnect_backoff_s,
        )
    source_text = str(source)
    if any(token in source_text for token in "*?["):
        matches = sorted(glob(source_text))
        if not matches:
            raise FileNotFoundError(source_text)
        return CompositeSource([create_frame_source(match, stride=stride) for match in matches])
    if "://" in source_text:
        return VideoSource(
            source_text,
            stride=stride,
            reconnect_attempts=reconnect_attempts,
            reconnect_backoff_s=reconnect_backoff_s,
        )
    path = Path(source_text).expanduser().resolve()
    if path.is_dir():
        return DirectorySource(path, stride=stride)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return ImageSource(path)
    if path.is_file():
        return VideoSource(path, stride=stride)
    raise FileNotFoundError(path)


__all__ = (
    "CompositeSource",
    "DirectorySource",
    "FrameSource",
    "ImageSource",
    "VideoSource",
    "create_frame_source",
    "frame_from_bgr",
)
