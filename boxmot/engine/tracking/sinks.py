"""Engine-owned consumers for canonical tracking results."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Callable, Protocol, runtime_checkable

import cv2
import numpy as np
from typing_extensions import Self

from boxmot.engine.tracking.profiling import timed_runtime_stage
from boxmot.pipelines import PipelineResult
from boxmot.structures import Boxes, Frame, OrientedBoxes


class _StopTrackingRequested(Exception):
    """Internal control signal raised when an interactive sink requests exit."""


@runtime_checkable
class TrackSink(Protocol):
    """Consumer of a caller-owned frame and its pipeline result."""

    def write(self, frame: Frame, result: PipelineResult) -> None: ...

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *_exc: object) -> None: ...


class _BaseSink:
    def close(self) -> None:
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class NullSink(_BaseSink):
    """Discard all results."""

    def write(self, frame: Frame, result: PipelineResult) -> None:
        del frame, result


class RenderingSink(_BaseSink):
    """Render each result and pass the BGR image to a caller-owned consumer."""

    def __init__(
        self,
        consumer: Callable[[Frame, PipelineResult, np.ndarray], None],
        *,
        class_names: Mapping[int, str] | None = None,
        line_width: int = 2,
    ) -> None:
        if not callable(consumer):
            raise TypeError("consumer must be callable")
        self.consumer = consumer
        self.class_names = class_names
        self.line_width = int(line_width)

    def write(self, frame: Frame, result: PipelineResult) -> None:
        self.write_rendered(
            frame,
            result,
            render_result(
                frame,
                result,
                class_names=self.class_names,
                line_width=self.line_width,
            ),
        )

    def write_rendered(
        self,
        frame: Frame,
        result: PipelineResult,
        image: np.ndarray,
    ) -> None:
        """Consume an image already rendered by a shared engine sink."""

        self.consumer(frame, result, image)


class DisplaySink(RenderingSink):
    """Render tracking results in an OpenCV window."""

    def __init__(
        self,
        *,
        window_name: str = "BoxMOT",
        class_names: Mapping[int, str] | None = None,
        line_width: int = 2,
    ) -> None:
        if not window_name:
            raise ValueError("window_name must not be empty")
        self.window_name = window_name
        self._opened = False
        super().__init__(self._display, class_names=class_names, line_width=line_width)

    def _display(self, frame: Frame, result: PipelineResult, image: np.ndarray) -> None:
        del frame, result
        cv2.imshow(self.window_name, image)
        self._opened = True
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise _StopTrackingRequested

    def close(self) -> None:
        if self._opened:
            cv2.destroyWindow(self.window_name)
            self._opened = False


class CompositeSink(_BaseSink):
    """Fan out each result to multiple sinks in declaration order."""

    def __init__(self, sinks: Sequence[TrackSink]) -> None:
        self.sinks = tuple(sinks)

    def write(self, frame: Frame, result: PipelineResult) -> None:
        for sink in self.sinks:
            sink.write(frame, result)

    def close(self) -> None:
        errors: list[BaseException] = []
        for sink in reversed(self.sinks):
            try:
                sink.close()
            except BaseException as exc:  # noqa: BLE001 - close every sink before propagating
                errors.append(exc)
        if errors:
            raise errors[0]


def _json_tracks(result: PipelineResult) -> list[dict[str, object]]:
    tracks = result.tracks
    geometry = tracks.geometry.values.tolist()
    rows: list[dict[str, object]] = []
    for index, values in enumerate(geometry):
        rows.append(
            {
                "geometry": values,
                "geometry_mode": "obb" if isinstance(tracks.geometry, OrientedBoxes) else "aabb",
                "track_id": int(tracks.track_ids[index]),
                "score": float(tracks.scores[index]),
                "class_id": int(tracks.class_ids[index]),
                "detection_index": int(tracks.detection_indices[index]),
            }
        )
    return rows


class JsonLinesSink(_BaseSink):
    """Write one JSON object per frame."""

    def __init__(self, destination: str | Path | IO[str]) -> None:
        self._owns_handle = not hasattr(destination, "write")
        self._handle: IO[str]
        if self._owns_handle:
            path = Path(destination)  # type: ignore[arg-type]
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a", encoding="utf-8")
        else:
            self._handle = destination  # type: ignore[assignment]

    def write(self, frame: Frame, result: PipelineResult) -> None:
        payload = {
            "sample_id": frame.sample_id,
            "sequence_id": frame.sequence_id,
            "frame_index": frame.frame_index,
            "timestamp_s": frame.timestamp_s,
            "tracks": _json_tracks(result),
        }
        self._handle.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._owns_handle and not self._handle.closed:
            self._handle.close()


class MotSink(_BaseSink):
    """Write MOTChallenge AABB rows; OBB results are rejected explicitly."""

    def __init__(self, destination: str | Path | IO[str]) -> None:
        self._owns_handle = not hasattr(destination, "write")
        if self._owns_handle:
            path = Path(destination)  # type: ignore[arg-type]
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle: IO[str] = path.open("a", encoding="utf-8")
        else:
            self._handle = destination  # type: ignore[assignment]

    def write(self, frame: Frame, result: PipelineResult) -> None:
        if not isinstance(result.tracks.geometry, Boxes):
            raise ValueError("MotSink accepts axis-aligned tracks only")
        frame_number = (frame.frame_index or 0) + 1
        boxes = result.tracks.geometry.values
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = (float(value) for value in box)
            row = (
                frame_number,
                int(result.tracks.track_ids[index]),
                x1,
                y1,
                x2 - x1,
                y2 - y1,
                float(result.tracks.scores[index]),
                int(result.tracks.class_ids[index]),
                -1,
            )
            self._handle.write(",".join(str(value) for value in row) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._owns_handle and not self._handle.closed:
            self._handle.close()


def _frame_bgr(frame: Frame) -> np.ndarray:
    rgb = frame.image.permute(1, 2, 0).numpy()
    return np.ascontiguousarray(rgb[..., ::-1])


def render_result(
    frame: Frame,
    result: PipelineResult,
    *,
    class_names: Mapping[int, str] | None = None,
    line_width: int = 2,
) -> np.ndarray:
    """Render canonical tracks while exposing engine-owned render timing."""

    with timed_runtime_stage("rendering"):
        return _render_result(
            frame,
            result,
            class_names=class_names,
            line_width=line_width,
        )


def _render_result(
    frame: Frame,
    result: PipelineResult,
    *,
    class_names: Mapping[int, str] | None = None,
    line_width: int = 2,
) -> np.ndarray:
    """Render canonical tracks into a new OpenCV BGR frame."""

    image = _frame_bgr(frame).copy()
    tracks = result.tracks
    masks = tracks.masks
    if masks is None and result.detections.masks is not None and len(tracks):
        det_indices = tracks.detection_indices.numpy()
        valid = (det_indices >= 0) & (det_indices < len(result.detections))
        aligned = np.zeros((len(tracks), image.shape[0], image.shape[1]), dtype=bool)
        aligned[valid] = result.detections.masks.values.numpy()[det_indices[valid]]
        mask_values = aligned
    else:
        mask_values = None if masks is None else masks.values.numpy()

    if mask_values is not None:
        overlay = image.copy()
        for index, mask in enumerate(mask_values):
            color = _track_color(int(tracks.track_ids[index]))
            overlay[mask] = color
        image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)

    if isinstance(tracks.geometry, Boxes):
        for index, box in enumerate(tracks.geometry.values.numpy()):
            x1, y1, x2, y2 = np.rint(box).astype(int)
            color = _track_color(int(tracks.track_ids[index]))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)
            _draw_label(image, tracks, index, (x1, y1), color, class_names)
    else:
        for index, box in enumerate(tracks.geometry.values.numpy()):
            cx, cy, width, height, angle = (float(value) for value in box)
            corners = cv2.boxPoints(((cx, cy), (width, height), float(np.degrees(angle))))
            corners = np.rint(corners).astype(np.int32)
            color = _track_color(int(tracks.track_ids[index]))
            cv2.polylines(image, [corners], True, color, line_width)
            anchor = tuple(int(value) for value in corners[0])
            _draw_label(image, tracks, index, anchor, color, class_names)
    return image


def _track_color(track_id: int) -> tuple[int, int, int]:
    return ((37 * track_id) % 205 + 50, (17 * track_id) % 205 + 50, (29 * track_id) % 205 + 50)


def _draw_label(image, tracks, index, anchor, color, class_names) -> None:
    class_id = int(tracks.class_ids[index])
    class_label = str(class_id) if class_names is None else class_names.get(class_id, str(class_id))
    label = f"{class_label} #{int(tracks.track_ids[index])} {float(tracks.scores[index]):.2f}"
    cv2.putText(image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


class VideoSink(_BaseSink):
    """Render and append frames to a video file."""

    def __init__(
        self,
        destination: str | Path,
        *,
        fps: float,
        class_names: Mapping[int, str] | None = None,
        line_width: int = 2,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.path = Path(destination)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.class_names = class_names
        self.line_width = int(line_width)
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame: Frame, result: PipelineResult) -> None:
        self.write_rendered(
            frame,
            result,
            render_result(
                frame,
                result,
                class_names=self.class_names,
                line_width=self.line_width,
            ),
        )

    def write_rendered(
        self,
        frame: Frame,
        result: PipelineResult,
        rendered: np.ndarray,
    ) -> None:
        """Append an image already rendered by a shared engine sink."""

        del frame, result
        if self._writer is None:
            height, width = rendered.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                raise OSError(f"Could not open video sink: {self.path}")
        self._writer.write(rendered)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class SharedRenderingSink(_BaseSink):
    """Render once and fan the same image out to engine rendering sinks."""

    def __init__(
        self,
        sinks: Sequence[RenderingSink | VideoSink],
        *,
        class_names: Mapping[int, str] | None = None,
        line_width: int = 2,
    ) -> None:
        self.sinks = tuple(sinks)
        if not self.sinks:
            raise ValueError("SharedRenderingSink requires at least one sink")
        self.class_names = class_names
        self.line_width = int(line_width)

    def write(self, frame: Frame, result: PipelineResult) -> None:
        rendered = render_result(
            frame,
            result,
            class_names=self.class_names,
            line_width=self.line_width,
        )
        for sink in self.sinks:
            sink.write_rendered(frame, result, rendered)

    def close(self) -> None:
        errors: list[BaseException] = []
        for sink in reversed(self.sinks):
            try:
                sink.close()
            except BaseException as exc:  # noqa: BLE001 - close every sink before propagating
                errors.append(exc)
        if errors:
            raise errors[0]


__all__ = (
    "CompositeSink",
    "DisplaySink",
    "JsonLinesSink",
    "MotSink",
    "NullSink",
    "RenderingSink",
    "SharedRenderingSink",
    "TrackSink",
    "VideoSink",
    "render_result",
)
