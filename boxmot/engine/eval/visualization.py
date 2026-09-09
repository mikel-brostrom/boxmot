"""Display and save cached tracking replay using its capture timeline."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from typing_extensions import Self

from boxmot.engine.eval.replay import _validate_sequence_id
from boxmot.engine.tracking.sinks import render_result
from boxmot.structures import CameraModel, Tracks3D

if TYPE_CHECKING:
    from boxmot.engine.eval.replay import ReplayFrame


class ReplayVisualization:
    """Render final replay in the caller thread with optional paced preview.

    Videos use the configured output grid (30 FPS by default) and hold the
    previous rendered observation across capture gaps. Repeated output images
    never become tracker updates. Sequences without timestamps use one output
    frame per observation. Odd image dimensions are padded at the video boundary.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        show: bool,
        save: bool,
        class_names: Mapping[int, str] | None = None,
        video_fps: float = 30.0,
    ) -> None:
        if (
            isinstance(video_fps, bool)
            or not isinstance(video_fps, Real)
            or not math.isfinite(video_fps)
            or video_fps <= 0
        ):
            raise ValueError("Replay video_fps must be a finite positive number.")
        self.output_dir = Path(output_dir)
        self.show = bool(show)
        self.save = bool(save)
        self.class_names = class_names
        self.video_fps = float(video_fps)
        self._video_paths: list[Path] = []
        self._sequence: str | None = None
        self._writer: cv2.VideoWriter | None = None
        self._previous_image: np.ndarray | None = None
        self._origin: float | None = None
        self._previous_time = 0.0
        self._next_tick = 0
        self._observations = 0
        self._clock_origin = 0.0
        self._window_open = False
        self._window_name = "BoxMOT evaluation"
        self._closed = False

    @property
    def video_paths(self) -> tuple[Path, ...]:
        """Return videos opened during this replay, in sequence order."""
        return tuple(self._video_paths)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _start_sequence(self, sequence: str, timestamp: float | None) -> None:
        self._finish_sequence()
        self._sequence = _validate_sequence_id(sequence)
        self._origin = timestamp
        self._previous_time = 0.0
        self._next_tick = 0
        self._observations = 0
        self._clock_origin = time.perf_counter()

    def _write_until(self, end_time: float) -> None:
        """Write the previous observation at output ticks before end_time."""
        if self._writer is None or self._previous_image is None:
            return
        end_tick = math.ceil(end_time * self.video_fps - 1e-9)
        if self._next_tick >= end_tick:
            return
        image = self._previous_image
        height, width = image.shape[:2]
        if height % 2 or width % 2:
            # mp4v needs even dimensions. Preserve the rendered/preview image
            # and extend its final row/column only for the encoded video.
            image = cv2.copyMakeBorder(image, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)
        while self._next_tick < end_tick:
            self._writer.write(image)
            self._next_tick += 1

    def _finish_sequence(self) -> None:
        try:
            # The last observation gets one output-frame duration; no future
            # capture interval is inferred after the source ends.
            self._write_until(self._previous_time + 1.0 / self.video_fps)
        finally:
            if self._writer is not None:
                self._writer.release()
            self._writer = None
            self._previous_image = None
            self._sequence = None

    def _open_video(self, image: np.ndarray) -> None:
        assert self._sequence is not None
        path = self.output_dir / "videos" / f"{self._sequence}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = image.shape[:2]
        dimensions = (width + width % 2, height + height % 2)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.video_fps, dimensions)
        if not writer.isOpened():
            writer.release()
            raise OSError(f"Could not open replay video: {path}.")
        self._writer = writer
        self._video_paths.append(path)

    def _check_key(self, delay_ms: int) -> None:
        if cv2.waitKey(delay_ms) & 0xFF in (ord("q"), 27):
            self.show = False
            self._close_window()

    def _close_window(self) -> None:
        if self._window_open:
            cv2.destroyWindow(self._window_name)
            self._window_open = False

    def _display(self, image: np.ndarray, elapsed: float) -> None:
        if not self.show:
            return
        if not self._window_open:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            scale = min(1.0, 1280.0 / width, 800.0 / height)
            cv2.resizeWindow(self._window_name, round(width * scale), round(height * scale))
            self._window_open = True
        # Event polling keeps q/Escape responsive during a long capture gap.
        while self.show:
            remaining = self._clock_origin + elapsed - time.perf_counter()
            if remaining <= 0:
                break
            self._check_key(max(1, min(50, math.ceil(remaining * 1000))))
        if self.show:
            cv2.imshow(self._window_name, image)
            self._check_key(1)

    def __call__(
        self,
        replayed: ReplayFrame,
        *,
        spatial_tracks: Tracks3D | None = None,
        camera: CameraModel | None = None,
    ) -> None:
        """Render one timed observation, optionally overlaying camera-space cuboids."""
        if self._closed:
            raise RuntimeError("Replay visualization is closed.")
        if not self.show and not self.save:
            return
        sample = replayed.sample
        if sample.frame is None:
            raise ValueError("Replay visualization requires original frame pixels.")
        timestamp = sample.timestamp_s
        if timestamp is not None and not math.isfinite(timestamp):
            raise ValueError("Replay visualization requires finite capture timestamps.")
        if sample.sequence_id != self._sequence:
            self._start_sequence(sample.sequence_id, timestamp)
        if (timestamp is None) != (self._origin is None):
            raise ValueError("Capture timestamps must be present for every frame in a sequence or absent for all.")
        elapsed = self._observations / self.video_fps if timestamp is None else timestamp - self._origin
        if self._observations and elapsed <= self._previous_time:
            raise ValueError("Replay capture timestamps must increase strictly.")
        image = render_result(
            sample.frame,
            replayed.result,
            class_names=self.class_names,
            spatial_tracks=spatial_tracks,
            camera=camera,
        )
        if self._previous_image is not None and image.shape != self._previous_image.shape:
            raise ValueError("Replay frame dimensions must remain constant within a sequence.")
        delta = None if not self._observations else elapsed - self._previous_time
        timing = f"{sample.sequence_id} | frame {sample.frame_index + 1}"
        if timestamp is not None:
            timing += f" | t={timestamp:.3f}s"
            if delta is not None:
                timing += f" | dt={delta * 1000:.1f}ms"
        cv2.putText(image, timing, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, timing, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if self.save:
            if self._writer is None:
                self._open_video(image)
            self._write_until(elapsed)
        self._previous_image = image
        self._previous_time = elapsed
        self._observations += 1
        self._display(image, elapsed)

    def close(self) -> None:
        """Flush the last observation and release every opened output."""
        if not self._closed:
            self._closed = True
            try:
                self._finish_sequence()
            finally:
                self._close_window()
