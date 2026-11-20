"""Killable network capture with a single replaceable shared-memory frame."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class CaptureDisconnected(RuntimeError):
    """Capture failed or stopped producing fresh frames."""


@dataclass(frozen=True)
class CapturedFrame:
    """Capture-owned pixels and provenance, independent of processing speed."""

    image: Any
    index: int
    timestamp_s: float | None
    captured_at: float
    captured_wall_time: float = 0.0


class FrameSlot:
    """One fixed-size frame with bounded lock waits, even if its writer dies."""

    def __init__(self, context: Any, max_bytes: int) -> None:
        self.pixels = context.RawArray("B", max_bytes)
        # index, height, width, timestamp, monotonic time, wall-clock time
        self.metadata = context.RawArray("d", [-1, 0, 0, float("nan"), 0, 0])
        self.lock = context.Lock()

    def put(self, frame: CapturedFrame) -> None:
        image = np.ascontiguousarray(frame.image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.nbytes > len(self.pixels):
            raise ValueError("Camera frame exceeds the configured BGR frame capacity.")
        if not self.lock.acquire(timeout=0.1):
            return
        try:
            np.frombuffer(self.pixels, dtype=np.uint8, count=image.size)[:] = image.reshape(-1)
            self.metadata[:] = (
                frame.index,
                image.shape[0],
                image.shape[1],
                frame.timestamp_s if frame.timestamp_s is not None else float("nan"),
                frame.captured_at,
                frame.captured_wall_time,
            )
        finally:
            self.lock.release()

    def get(self, previous_index: int) -> CapturedFrame | None:
        if not self.lock.acquire(timeout=0.1):
            return None
        try:
            index, height, width, timestamp, captured, wall = self.metadata[:]
            if index <= previous_index:
                return None
            image = np.frombuffer(self.pixels, dtype=np.uint8, count=int(height * width * 3)).copy()
            return CapturedFrame(
                image.reshape(int(height), int(width), 3),
                int(index),
                timestamp if np.isfinite(timestamp) else None,
                captured,
                wall,
            )
        finally:
            self.lock.release()


def _capture(url: str, timeout_s: float, slot: FrameSlot, stop: Any, timestamp_source: str) -> None:
    """Own all OpenCV operations in a process that the supervisor can terminate."""
    # OpenCV/FFmpeg can log credentialed URLs directly to stderr on failure.
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 2)
    import cv2

    from boxmot.engine.tracking.timestamps import LiveTimestamps

    capture = None
    try:
        capture = cv2.VideoCapture(
            url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                int(timeout_s * 1000),
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                int(timeout_s * 1000),
            ],
        )
        if not capture.isOpened():
            return
        timestamps = LiveTimestamps(timestamp_source)
        index = 0
        while not stop.is_set():
            ok, image = capture.read()
            if not ok:
                return
            captured_at, wall_time = time.monotonic(), time.time()
            slot.put(
                CapturedFrame(
                    image,
                    index,
                    timestamps.resolve(
                        captured_at=captured_at,
                        position_ms=float(capture.get(cv2.CAP_PROP_POS_MSEC)) if timestamp_source == "media" else None,
                    ),
                    captured_at,
                    wall_time,
                )
            )
            index += 1
    finally:
        if capture is not None:
            capture.release()


class NetworkCapture:
    """Bounded parent-side waiting and termination, including blocked native reads."""

    def __init__(
        self,
        url: str,
        timeout_s: float = 10,
        *,
        max_frame_bytes: int = 32 * 1024 * 1024,
        timestamp_source: str = "capture",
        target: Callable[..., None] = _capture,
    ) -> None:
        context = mp.get_context("spawn")
        self._slot = FrameSlot(context, max_frame_bytes)
        self._stop = context.Event()
        self._process = context.Process(
            target=target, args=(url, timeout_s, self._slot, self._stop, timestamp_source), daemon=True
        )
        self.timeout_s = timeout_s
        self._last_received = time.monotonic()
        self._previous_index = -1
        self._closed = False
        self._process.start()

    def next(self) -> CapturedFrame | None:
        """Wait briefly so the supervisor can check signals and ownership."""
        frame = self._slot.get(self._previous_index)
        if frame is None:
            if not self._process.is_alive() or time.monotonic() - self._last_received > self.timeout_s + 2:
                raise CaptureDisconnected("Camera disconnected or capture timed out.")
            self._stop.wait(0.02)
            return None
        self._last_received = time.monotonic()
        self._previous_index = frame.index
        if self._last_received - frame.captured_at > self.timeout_s:
            raise CaptureDisconnected("Camera supplied a stale buffered frame.")
        return frame

    def close(self) -> None:
        """Release or forcibly terminate capture without waiting on native code."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._process.join(timeout=0.2)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=0.5)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=0.5)
        if not self._process.is_alive():
            self._process.close()
