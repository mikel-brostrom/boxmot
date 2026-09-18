"""Engine-owned consumers for canonical tracking results."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import IO, Callable, Protocol, runtime_checkable

import cv2
import numpy as np
from typing_extensions import Self

from boxmot.engine.tracking.profiling import timed_runtime_stage
from boxmot.pipelines import PipelineResult
from boxmot.structures import Boxes, CameraModel, Frame, OrientedBoxes, Tracks3D


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
    spatial_tracks: Tracks3D | None = None,
    camera: CameraModel | None = None,
    guidance_masks: Mapping[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Render tracks and optional propagated masks using the same identity colors.

    When supplied, ``guidance_masks`` is the authoritative mask layer, including
    identities currently lost from box outputs. An empty mapping suppresses
    standalone segmentation masks while temporal guidance has none to show.
    """

    if (spatial_tracks is None) != (camera is None):
        raise ValueError("Rendering 3D tracks requires both spatial_tracks and camera.")
    if spatial_tracks is not None:
        if spatial_tracks.sample_id != frame.sample_id or result.tracks.sample_id != frame.sample_id:
            raise ValueError("Rendered image and spatial tracks must belong to the current frame.")
        if camera.image_size != frame.image_size:
            raise ValueError("3D visualization camera dimensions must match the image.")

    with timed_runtime_stage("rendering"):
        image = _render_result(
            frame,
            result,
            class_names=class_names,
            line_width=line_width,
            guidance_masks=guidance_masks,
        )
        if spatial_tracks is not None:
            from boxmot.engine.tracking.spatial_visualization import draw_spatial_tracks

            image = draw_spatial_tracks(
                image,
                spatial_tracks,
                camera,
                class_names=class_names,
                image_track_ids=frozenset(result.tracks.track_ids.tolist()),
                line_width=line_width,
            )
        return image


def _render_result(
    frame: Frame,
    result: PipelineResult,
    *,
    class_names: Mapping[int, str] | None = None,
    line_width: int = 2,
    guidance_masks: Mapping[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Render canonical tracks into a new OpenCV BGR frame."""

    image = _frame_bgr(frame)
    tracks = result.tracks
    mask_layers: Iterable[tuple[int, np.ndarray]]
    if guidance_masks is not None:
        if not isinstance(guidance_masks, Mapping):
            raise TypeError("guidance_masks must map track IDs to boolean mask arrays.")
        for track_id, mask in guidance_masks.items():
            if isinstance(track_id, bool) or not isinstance(track_id, Integral) or track_id < 0:
                raise ValueError("Guidance mask IDs must be non-negative integers.")
            if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
                raise TypeError("Guidance masks must be boolean NumPy arrays.")
            if mask.ndim != 2 or mask.shape != image.shape[:2]:
                raise ValueError("Guidance masks must be two-dimensional and match the frame dimensions.")
        # Stable layering is independent of mapping insertion or current box order.
        mask_layers = sorted(guidance_masks.items())
    elif tracks.masks is not None:
        mask_layers = zip(tracks.track_ids.tolist(), tracks.masks.values.numpy())
    elif result.detections.masks is not None:
        detection_masks = result.detections.masks.values.numpy()
        mask_layers = (
            (track_id, detection_masks[index])
            for track_id, index in zip(tracks.track_ids.tolist(), tracks.detection_indices.tolist())
            if 0 <= index < len(result.detections)
        )
    else:
        mask_layers = ()

    overlay = None
    for track_id, mask in mask_layers:
        if overlay is None:
            overlay = image.copy()
        overlay[mask] = _track_color(int(track_id))
    if overlay is not None:
        cv2.addWeighted(overlay, 0.35, image, 0.65, 0, dst=image)

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
        guidance_mask_provider: Callable[[], Mapping[int, np.ndarray]] | None = None,
    ) -> None:
        self.sinks = tuple(sinks)
        if not self.sinks:
            raise ValueError("SharedRenderingSink requires at least one sink")
        if guidance_mask_provider is not None and not callable(guidance_mask_provider):
            raise TypeError("guidance_mask_provider must be callable.")
        self.class_names = class_names
        self.line_width = int(line_width)
        self.guidance_mask_provider = guidance_mask_provider

    def write(self, frame: Frame, result: PipelineResult) -> None:
        rendered = render_result(
            frame,
            result,
            class_names=self.class_names,
            line_width=self.line_width,
            guidance_masks=None if self.guidance_mask_provider is None else self.guidance_mask_provider(),
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
