"""Engine-owned consumers for canonical tracking results."""

from __future__ import annotations

import colorsys
import json
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from typing import IO, Callable, Protocol, runtime_checkable

import cv2
import numpy as np
from typing_extensions import Self

from boxmot.engine.tracking.profiling import timed_runtime_stage
from boxmot.pipelines import PipelineResult
from boxmot.structures import Boxes, CameraModel, Frame, OrientedBoxes, Tracks, Tracks3D
from boxmot.utils import logger as LOGGER


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
        show_labels: bool = True,
    ) -> None:
        if not callable(consumer):
            raise TypeError("consumer must be callable")
        self.consumer = consumer
        self.class_names = class_names
        self.line_width = int(line_width)
        self.show_labels = bool(show_labels)

    def write(self, frame: Frame, result: PipelineResult) -> None:
        self.write_rendered(
            frame,
            result,
            render_result(
                frame,
                result,
                class_names=self.class_names,
                line_width=self.line_width,
                show_labels=self.show_labels,
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


def _missing_gui_support(error: cv2.error) -> bool:
    """Recognize HighGUI stubs, which also use OpenCV's generic error code."""
    return error.code == cv2.Error.StsNotImplemented or (
        error.code == cv2.Error.StsError and "the function is not implemented" in str(error).lower()
    )


class DisplaySink(RenderingSink):
    """Render in an OpenCV window, disabling preview if GUI support is missing."""

    def __init__(
        self,
        *,
        window_name: str = "BoxMOT",
        class_names: Mapping[int, str] | None = None,
        line_width: int = 2,
        show_labels: bool = True,
    ) -> None:
        if not window_name:
            raise ValueError("window_name must not be empty")
        self.window_name = window_name
        self._opened = False
        self._disabled = False
        super().__init__(self._display, class_names=class_names, line_width=line_width, show_labels=show_labels)

    def _display(self, frame: Frame, result: PipelineResult, image: np.ndarray) -> None:
        """Show one frame, allowing other outputs to continue on headless builds."""
        del frame, result
        if self._disabled:
            return
        try:
            cv2.imshow(self.window_name, image)
            self._opened = True
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            if not _missing_gui_support(exc):
                raise
            self._disabled = True
            self.close()
            LOGGER.warning(
                "OpenCV GUI support is unavailable; disabling preview. "
                "Tracking and requested saved outputs will continue; press Ctrl-C to stop. "
                "Restore the project's opencv-python GUI package and keep vLLM in a separate "
                "environment to avoid its opencv-python-headless dependency overwriting cv2."
            )
            return
        if key in (ord("q"), 27):
            raise _StopTrackingRequested

    def close(self) -> None:
        """Close an opened preview, including partially available GUI backends."""
        if self._opened:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error as exc:
                if not _missing_gui_support(exc):
                    raise
            finally:
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
        attribute = result.attributes.get(int(tracks.track_ids[index]))
        if attribute is not None:
            rows[-1]["attributes"] = attribute.to_dict()
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
    show_labels: bool = True,
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
            show_labels=show_labels,
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
    show_labels: bool = True,
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

    label_anchors: list[tuple[int, int]] = []
    if isinstance(tracks.geometry, Boxes):
        for index, box in enumerate(tracks.geometry.values.numpy()):
            x1, y1, x2, y2 = np.rint(box).astype(int)
            color = _track_color(int(tracks.track_ids[index]))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width, lineType=cv2.LINE_8)
            label_anchors.append((int(x1), int(y1)))
    else:
        for index, box in enumerate(tracks.geometry.values.numpy()):
            cx, cy, width, height, angle = (float(value) for value in box)
            corners = cv2.boxPoints(((cx, cy), (width, height), float(np.degrees(angle))))
            corners = np.rint(corners).astype(np.int32)
            color = _track_color(int(tracks.track_ids[index]))
            cv2.polylines(image, [corners], True, color, line_width, lineType=cv2.LINE_8)
            top_corner = corners[np.argmin(corners[:, 1])]
            label_anchors.append((int(top_corner[0]), int(top_corner[1])))
    # Draw badges last so another track's box cannot cross their text.
    if show_labels:
        for index, anchor in enumerate(label_anchors):
            _draw_label(
                image, tracks, index, anchor, class_names, line_width=line_width,
                attribute=result.attributes.get(int(tracks.track_ids[index])),
            )
    return image


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Keep every identity, including zero, vivid in boxes and translucent masks."""

    hue = (track_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(blue * 255), int(green * 255), int(red * 255)


def _label_text_color(background: tuple[int, int, int]) -> tuple[int, int, int]:
    """Choose black or white text with the stronger contrast against a BGR fill."""

    channels = (value / 255.0 for value in reversed(background))
    red, green, blue = (
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    )
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    black_contrast = (luminance + 0.05) / 0.05
    white_contrast = 1.05 / (luminance + 0.05)
    return (0, 0, 0) if black_contrast >= white_contrast else (255, 255, 255)


@lru_cache(maxsize=256)
def _label_text_mask(label: str, font_scale: float) -> np.ndarray:
    """Return read-only text coverage cropped to its actual antialiased pixels."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(label, font, font_scale, 1)
    guard = 3
    mask = np.zeros((height + baseline + 2 * guard, width + 2 * guard), dtype=np.uint8)
    cv2.putText(mask, label, (guard, guard + height), font, font_scale, 255, 1, cv2.LINE_AA)
    x, y, width, height = cv2.boundingRect(mask)
    ink = mask[y : y + height, x : x + width].copy()
    ink.setflags(write=False)
    return ink


def _attribute_label_parts(attribute: object | None) -> list[str]:
    """Format only explicitly selected scalar dot paths from a completed result."""
    if attribute is None or getattr(attribute, "status", None) != "completed":
        return []
    parts: list[str] = []
    for path in getattr(attribute, "display_fields", ()):
        value = getattr(attribute, "values", None)
        for key in path.split("."):
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            if value is None or isinstance(value, (str, int, float, bool)):
                text = "unknown" if value is None else str(value)
                text = " ".join(text.split())
                parts.append(f"{path}: {text[:80]}")
    return parts


def _wrap_label_line(text: str, font_scale: float, max_width: int) -> list[str]:
    """Wrap a field at spaces, splitting unusually long words only when needed."""
    lines: list[str] = []
    while text:
        if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] <= max_width:
            lines.append(text)
            break
        low, high = 1, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            width = cv2.getTextSize(text[:middle], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
            if width <= max_width:
                low = middle
            else:
                high = middle - 1
        split = text.rfind(" ", 0, low + 1)
        if split <= 0:
            split = low
        lines.append(text[:split].rstrip())
        text = text[split:].lstrip()
    return lines


def _label_block_mask(lines: Sequence[str], font_scale: float, max_width: int) -> np.ndarray:
    """Keep each selected attribute on readable, wrapped lines below its identity."""
    masks = [
        _label_text_mask(wrapped, font_scale)
        for line in lines
        for wrapped in _wrap_label_line(line, font_scale, max_width)
    ]
    gap = max(2, round(4 * font_scale))
    width = max(mask.shape[1] for mask in masks)
    height = sum(mask.shape[0] for mask in masks) + gap * (len(masks) - 1)
    block = np.zeros((height, width), dtype=np.uint8)
    top = 0
    for mask in masks:
        line_height, line_width = mask.shape
        block[top : top + line_height, :line_width] = mask
        top += line_height + gap
    return block


def _draw_label(
    image: np.ndarray,
    tracks: Tracks,
    index: int,
    anchor: tuple[int, int],
    class_names: Mapping[int, str] | None,
    *,
    line_width: int,
    attribute: object | None = None,
) -> None:
    """Align the badge to the outer stroke and its first ink just inside the border."""

    class_id = int(tracks.class_ids[index])
    class_label = str(class_id) if class_names is None else class_names.get(class_id, str(class_id))
    label = f"{class_label} #{int(tracks.track_ids[index])} {float(tracks.scores[index]):.2f}"
    parts = _attribute_label_parts(attribute)
    font_scale = 0.5
    padding = 2
    # LINE_8 outlines extend beyond their coordinates; filled rectangles do not.
    # Odd widths above one round outward (widths 3 and 4 both extend 2 pixels).
    stroke_radius = (line_width + 1) // 2 if line_width > 1 else 0
    text_inset = 2 * stroke_radius + 1
    image_height, image_width = image.shape[:2]
    available_width = image_width - text_inset - padding
    available_height = image_height - 2 * padding
    if available_width <= 0 or available_height <= 0:
        return

    text_mask = (
        _label_block_mask([label, *parts], font_scale, available_width)
        if parts
        else _label_text_mask(label, font_scale)
    )
    text_height, text_width = text_mask.shape
    fit = min(1.0, available_width / text_width, available_height / text_height)
    if fit < 1.0:
        # Leave room for pixel rounding and the fixed text stroke after scaling.
        font_scale *= fit * 0.95
        text_mask = (
            _label_block_mask([label, *parts], font_scale, available_width)
            if parts
            else _label_text_mask(label, font_scale)
        )
        text_height, text_width = text_mask.shape
    badge_width = text_width + text_inset + padding
    badge_height = text_height + 2 * padding
    if badge_width > image_width or badge_height > image_height:
        return

    left = min(max(anchor[0] - stroke_radius, 0), image_width - badge_width)
    # Overlap the rounded corner up to its centerline for a continuous join.
    top = anchor[1] - badge_height + 1
    if top < 0:
        top = anchor[1] - stroke_radius
    top = min(max(top, 0), image_height - badge_height)
    color = _track_color(int(tracks.track_ids[index]))
    cv2.rectangle(
        image,
        (left, top),
        (left + badge_width - 1, top + badge_height - 1),
        color,
        cv2.FILLED,
    )
    # Cropping actual glyph coverage removes font bearings without clipping
    # leading antialiasing or descenders at image boundaries.
    alpha = text_mask.astype(np.uint16)[..., None]
    foreground = np.asarray(_label_text_color(color), dtype=np.uint16)
    background = np.asarray(color, dtype=np.uint16)
    text_pixels = ((foreground * alpha + background * (255 - alpha) + 127) // 255).astype(np.uint8)
    text_left, text_top = left + text_inset, top + padding
    image[text_top : text_top + text_height, text_left : text_left + text_width] = text_pixels


class VideoSink(_BaseSink):
    """Render and append frames to a video file."""

    def __init__(
        self,
        destination: str | Path,
        *,
        fps: float,
        class_names: Mapping[int, str] | None = None,
        line_width: int = 2,
        show_labels: bool = True,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.path = Path(destination)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.class_names = class_names
        self.line_width = int(line_width)
        self.show_labels = bool(show_labels)
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
                show_labels=self.show_labels,
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
        show_labels: bool = True,
        guidance_mask_provider: Callable[[], Mapping[int, np.ndarray]] | None = None,
    ) -> None:
        self.sinks = tuple(sinks)
        if not self.sinks:
            raise ValueError("SharedRenderingSink requires at least one sink")
        if guidance_mask_provider is not None and not callable(guidance_mask_provider):
            raise TypeError("guidance_mask_provider must be callable.")
        self.class_names = class_names
        self.line_width = int(line_width)
        self.show_labels = bool(show_labels)
        self.guidance_mask_provider = guidance_mask_provider

    def write(self, frame: Frame, result: PipelineResult) -> None:
        rendered = render_result(
            frame,
            result,
            class_names=self.class_names,
            line_width=self.line_width,
            show_labels=self.show_labels,
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
