from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol, TypeAlias

import cv2
import numpy as np

from boxmot.core.box_schema import BoxSchema, BoxType, get_box_schema
from boxmot.engine.service.config import OBB_ASSOCIATION_FUNCTIONS, ServiceSettings
from boxmot.engine.service.models import FrameRequest
from boxmot.engine.tracking.inputs import TrackerInputAdapter
from boxmot.trackers.registry import create_tracker
from boxmot.utils import logger as LOGGER

StreamKey: TypeAlias = tuple[str, str]
TrackValue: TypeAlias = float | int


class TrackerProtocol(Protocol):
    """Small tracker surface required by the service manager."""

    def update(
        self,
        dets: np.ndarray,
        img: np.ndarray | None = None,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> np.ndarray: ...

    def reset(self) -> None: ...


TrackerFactory: TypeAlias = Callable[[int], TrackerProtocol]
ReIDModelFactory: TypeAlias = Callable[[ServiceSettings], object]


class ServiceRequestError(Exception):
    """Base class for errors caused by a service request."""


class DetectionValidationError(ServiceRequestError):
    """A detection matrix does not satisfy the public BoxMOT contract."""


class ImageValidationError(ServiceRequestError):
    """An encoded frame does not satisfy the service image contract."""


class FrameConflictError(ServiceRequestError):
    """A request conflicts with the state or ordering of its stream."""


class StreamCapacityError(ServiceRequestError):
    """The process cannot create another in-memory tracker stream."""


class TrackerExecutionError(Exception):
    """A tracker failed and its session was discarded."""


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Immutable result retained for safe retry replay."""

    frame_id: int
    next_frame_id: int
    box_type: BoxType
    tracks: tuple[tuple[TrackValue, ...], ...]
    replayed: bool = False


@dataclass(slots=True)
class StreamState:
    """Mutable state owned by exactly one stream/session key."""

    tracker: TrackerProtocol
    input_adapter: TrackerInputAdapter
    width: int
    height: int
    frame_rate: int
    box_type: BoxType
    image: np.ndarray | None
    lock: asyncio.Lock
    last_seen: float
    last_frame_id: int | None = None
    last_request_digest: bytes | None = None
    last_result: FrameResult | None = None
    observed_class_ids: set[int] = field(default_factory=set)


class TrackerManager:
    """Own tracker instances and serialize updates within each stream."""

    def __init__(
        self,
        settings: ServiceSettings,
        *,
        tracker_factory: TrackerFactory | None = None,
        reid_model_factory: ReIDModelFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self._reid_model = None
        if settings.requires_image and tracker_factory is None:
            model_factory = reid_model_factory or self._build_reid_model
            self._reid_model = model_factory(settings)
        self._tracker_factory = tracker_factory or self._build_tracker
        self._clock = clock
        self._states: dict[StreamKey, StreamState] = {}
        self._registry_lock = asyncio.Lock()
        self._update_slots = asyncio.Semaphore(settings.max_concurrent_updates)

    def _build_tracker(self, frame_rate: int) -> TrackerProtocol:
        tracker_kwargs = {"asso_func": self.settings.asso_func}
        if self.settings.tracker_type in {"bytetrack", "botsort"}:
            tracker_kwargs["frame_rate"] = frame_rate
        return create_tracker(
            self.settings.tracker_type,
            per_class=True,
            tracker_backend="python",
            tracker_kwargs=tracker_kwargs,
            reid_model=self._reid_model,
            warmup_model=False,
        )

    @staticmethod
    def _build_reid_model(settings: ServiceSettings) -> object:
        """Load and warm one ReID backend shared by all streams in this process."""

        from boxmot.reid.core import ReID
        from boxmot.utils.misc import resolve_model_path

        weights = resolve_model_path(Path(settings.reid_weights))
        if not weights.is_file():
            raise FileNotFoundError(
                f"ReID weights were not found at {weights}. Mount the checkpoint and set "
                "BOXMOT_SERVICE_REID_WEIGHTS to its path."
            )
        model = ReID(weights=weights, device=settings.device, half=settings.half).model
        model.warmup()
        return model

    async def process(self, key: StreamKey, request: FrameRequest) -> FrameResult:
        """Validate and process one sequential frame for ``key``."""

        self._validate_association_contract(request)
        detections, schema = self._prepare_detections(request)
        await self._update_slots.acquire()
        try:
            return await self._process_with_slot(key, request, detections, schema)
        finally:
            self._update_slots.release()

    def _validate_association_contract(self, request: FrameRequest) -> None:
        """Reject association modes that cannot operate on the request geometry."""

        if request.box_type is BoxType.OBB and self.settings.asso_func not in OBB_ASSOCIATION_FUNCTIONS:
            available = ", ".join(OBB_ASSOCIATION_FUNCTIONS)
            raise DetectionValidationError(
                f"Association function {self.settings.asso_func!r} is not supported for OBB tracking; "
                f"choose one of: {available}."
            )

    async def _process_with_slot(
        self,
        key: StreamKey,
        request: FrameRequest,
        detections: np.ndarray,
        schema: BoxSchema,
    ) -> FrameResult:
        """Decode and update while holding one process-wide frame slot."""

        image, image_digest = await self._prepare_image(request)
        request_digest = self._request_digest(request, detections, image_digest)
        state = await self._get_or_create_state(key, request)

        async with state.lock:
            if not await self._is_current_state(key, state):
                raise FrameConflictError("The tracker session was reset while this request was waiting.")
            self._validate_stream_contract(state, request)

            if state.last_frame_id is not None and request.frame_id == state.last_frame_id:
                if request_digest != state.last_request_digest or state.last_result is None:
                    raise FrameConflictError(f"Frame {request.frame_id} was already processed with different input.")
                state.last_seen = self._clock()
                return replace(state.last_result, replayed=True)

            if state.last_frame_id is not None and request.frame_id != state.last_frame_id + 1:
                raise FrameConflictError(
                    f"Expected frame {state.last_frame_id + 1}, got {request.frame_id}. "
                    "Send every frame, including frames with no detections."
                )

            frame_class_ids = self._frame_class_ids(detections, schema)
            if len(state.observed_class_ids | frame_class_ids) > self.settings.max_classes_per_stream:
                raise DetectionValidationError(
                    f"A tracker session may contain at most {self.settings.max_classes_per_stream} distinct class IDs."
                )

            try:
                raw_tracks, was_cancelled = await self._update_without_releasing_lock(
                    state,
                    detections,
                    state.image if image is None else image,
                )
                tracks = self._serialize_tracks(raw_tracks, schema)
            except Exception as exc:
                await self._discard_state(key, state)
                await self._reset_state(state, reason=f"failed stream {key[0]}/{key[1]}")
                LOGGER.exception("Tracker update failed; discarded stream %s/%s", *key)
                raise TrackerExecutionError(
                    "Tracker update failed and the session was discarded; retry with a new session ID."
                ) from exc

            result = FrameResult(
                frame_id=request.frame_id,
                next_frame_id=request.frame_id + 1,
                box_type=request.box_type,
                tracks=tracks,
            )
            state.last_frame_id = request.frame_id
            state.last_request_digest = request_digest
            state.last_result = result
            state.last_seen = self._clock()
            state.observed_class_ids.update(frame_class_ids)
            if was_cancelled:
                # The tracker has advanced and the result is cached, so an exact
                # client retry is safe even though the disconnected request no
                # longer receives this response.
                raise asyncio.CancelledError
            return result

    async def delete(self, key: StreamKey) -> bool:
        """Delete and reset a tracker session, returning whether it existed."""

        async with self._registry_lock:
            state = self._states.get(key)
        if state is None:
            return False

        async with state.lock:
            async with self._registry_lock:
                if self._states.get(key) is not state:
                    return False
                self._states.pop(key, None)
            await self._reset_state(state, reason=f"deleted stream {key[0]}/{key[1]}")
        return True

    async def stats(self) -> dict[str, int]:
        """Return current capacity after evicting idle, unlocked streams."""

        await self._evict_expired(self._clock())
        async with self._registry_lock:
            return {
                "active_streams": len(self._states),
                "max_streams": self.settings.max_streams,
            }

    async def close(self) -> None:
        """Remove all stream state during application shutdown."""

        async with self._registry_lock:
            states = list(self._states.values())
            self._states.clear()
        for state in states:
            async with state.lock:
                await self._reset_state(state, reason="application shutdown")

    def _prepare_detections(self, request: FrameRequest) -> tuple[np.ndarray, BoxSchema]:
        schema = get_box_schema(request.box_type)
        row_count = len(request.detections)
        if row_count > self.settings.max_detections_per_frame:
            raise DetectionValidationError(
                f"At most {self.settings.max_detections_per_frame} detections are allowed per frame."
            )
        if row_count == 0:
            return schema.empty_detections(), schema

        try:
            values = np.asarray(request.detections, dtype=np.float64)
        except (OverflowError, TypeError, ValueError) as exc:
            raise DetectionValidationError("Detections must be a rectangular numeric matrix.") from exc
        expected_shape = (row_count, schema.detection_cols)
        if values.shape != expected_shape:
            raise DetectionValidationError(
                f"{request.box_type.value.upper()} detections must have shape "
                f"(N, {schema.detection_cols}), got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise DetectionValidationError("Detections must contain only finite numbers.")
        if np.abs(values).max(initial=0.0) > np.finfo(np.float32).max:
            raise DetectionValidationError("Detection values exceed the supported float32 range.")

        geometry = values[:, : schema.geometry_cols]
        if schema.is_obb:
            if np.any(geometry[:, 2:4] <= 0.0):
                raise DetectionValidationError("OBB width and height must be positive.")
        elif np.any(geometry[:, 2] <= geometry[:, 0]) or np.any(geometry[:, 3] <= geometry[:, 1]):
            raise DetectionValidationError("AABB detections must satisfy x2 > x1 and y2 > y1.")

        confidences = values[:, schema.detection_conf_index]
        if np.any((confidences < 0.0) | (confidences > 1.0)):
            raise DetectionValidationError("Detection confidence must be between 0 and 1.")

        class_ids = values[:, schema.detection_class_index]
        if np.any(class_ids < 0.0) or not np.equal(class_ids, np.floor(class_ids)).all():
            raise DetectionValidationError("Detection class IDs must be non-negative integers.")
        if np.any(class_ids > 16_777_215):
            raise DetectionValidationError("Detection class IDs must not exceed 16777215.")
        if np.unique(class_ids).size > self.settings.max_classes_per_stream:
            raise DetectionValidationError(
                f"A frame may contain at most {self.settings.max_classes_per_stream} distinct class IDs."
            )

        return values.astype(np.float32), schema

    async def _prepare_image(self, request: FrameRequest) -> tuple[np.ndarray | None, bytes]:
        """Decode and validate an optional frame without blocking the event loop."""

        if request.image_base64 is None:
            if self.settings.requires_image:
                raise ImageValidationError(
                    "image_base64 is required by the GPU service profile, including on frames with no detections."
                )
            return None, b""
        if request.width * request.height > self.settings.max_frame_pixels:
            raise ImageValidationError(f"Frame dimensions exceed the {self.settings.max_frame_pixels}-pixel limit.")
        decode_task = asyncio.create_task(asyncio.to_thread(self._decode_image, request))
        was_cancelled = False
        while not decode_task.done():
            try:
                await asyncio.shield(decode_task)
            except asyncio.CancelledError:
                was_cancelled = True
        result = decode_task.result()
        if was_cancelled:
            raise asyncio.CancelledError
        return result

    def _decode_image(self, request: FrameRequest) -> tuple[np.ndarray, bytes]:
        try:
            encoded = request.image_base64.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ImageValidationError("image_base64 must contain ASCII base64 data.") from exc
        max_encoded_length = 4 * ((self.settings.max_image_bytes + 2) // 3)
        if len(encoded) > max_encoded_length:
            raise ImageValidationError(f"Encoded image exceeds the {self.settings.max_image_bytes}-byte limit.")
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImageValidationError("image_base64 is not valid base64 data.") from exc
        if not image_bytes:
            raise ImageValidationError("image_base64 must encode a non-empty JPEG or PNG image.")
        if len(image_bytes) > self.settings.max_image_bytes:
            raise ImageValidationError(f"Encoded image exceeds the {self.settings.max_image_bytes}-byte limit.")

        encoded_width, encoded_height = self._encoded_image_dimensions(image_bytes)
        if encoded_width * encoded_height > self.settings.max_frame_pixels:
            raise ImageValidationError(f"Encoded image exceeds the {self.settings.max_frame_pixels}-pixel limit.")
        if (encoded_width, encoded_height) != (request.width, request.height):
            raise ImageValidationError(
                "Encoded image dimensions must match the declared frame dimensions: "
                f"expected {(request.width, request.height)}, got {(encoded_width, encoded_height)}."
            )

        decode_flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), decode_flags)
        if image is None:
            raise ImageValidationError("image_base64 must encode a valid JPEG or PNG image.")
        expected_shape = (request.height, request.width, 3)
        if image.shape != expected_shape:
            raise ImageValidationError(f"Decoded image must have shape {expected_shape}, got {image.shape}.")
        image_digest = hashlib.blake2b(image_bytes, digest_size=16).digest()
        return np.ascontiguousarray(image), image_digest

    @staticmethod
    def _encoded_image_dimensions(image_bytes: bytes) -> tuple[int, int]:
        """Read JPEG/PNG dimensions before OpenCV allocates the decoded frame."""

        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            if len(image_bytes) < 24 or image_bytes[12:16] != b"IHDR":
                raise ImageValidationError("image_base64 contains an invalid PNG header.")
            width, height = struct.unpack(">II", image_bytes[16:24])
            if width == 0 or height == 0:
                raise ImageValidationError("image_base64 contains invalid PNG dimensions.")
            return width, height

        if image_bytes.startswith(b"\xff\xd8"):
            sof_markers = {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }
            offset = 2
            while offset + 3 < len(image_bytes):
                if image_bytes[offset] != 0xFF:
                    offset += 1
                    continue
                while offset < len(image_bytes) and image_bytes[offset] == 0xFF:
                    offset += 1
                if offset >= len(image_bytes):
                    break
                marker = image_bytes[offset]
                offset += 1
                if marker in {0x01, *range(0xD0, 0xDA)}:
                    continue
                if offset + 2 > len(image_bytes):
                    break
                segment_length = int.from_bytes(image_bytes[offset : offset + 2], "big")
                if segment_length < 2 or offset + segment_length > len(image_bytes):
                    break
                if marker in sof_markers:
                    if segment_length < 7:
                        break
                    height = int.from_bytes(image_bytes[offset + 3 : offset + 5], "big")
                    width = int.from_bytes(image_bytes[offset + 5 : offset + 7], "big")
                    if width == 0 or height == 0:
                        break
                    return width, height
                offset += segment_length
            raise ImageValidationError("image_base64 contains an invalid JPEG header.")

        raise ImageValidationError("image_base64 must encode a JPEG or PNG image.")

    @staticmethod
    def _frame_class_ids(detections: np.ndarray, schema: BoxSchema) -> set[int]:
        return set(detections[:, schema.detection_class_index].astype(np.int64).tolist())

    async def _get_or_create_state(self, key: StreamKey, request: FrameRequest) -> StreamState:
        await self._evict_expired(self._clock())
        async with self._registry_lock:
            now = self._clock()
            state = self._states.get(key)
            if state is not None:
                return state
            if request.frame_id != 0:
                raise FrameConflictError(
                    "The tracker session does not exist or has expired. Start a new session with frame 0."
                )
            if len(self._states) >= self.settings.max_streams:
                raise StreamCapacityError(f"Tracker capacity reached ({self.settings.max_streams} active streams).")

            try:
                tracker = self._tracker_factory(request.frame_rate)
            except Exception as exc:
                LOGGER.exception("Failed to create tracker for stream %s/%s", *key)
                raise TrackerExecutionError(
                    "Tracker creation failed; verify the selected tracker and service profile."
                ) from exc

            input_adapter = TrackerInputAdapter(tracker)
            image = None
            if input_adapter.uses_img:
                image = np.broadcast_to(
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    (request.height, request.width, 3),
                )
            state = StreamState(
                tracker=tracker,
                input_adapter=input_adapter,
                width=request.width,
                height=request.height,
                frame_rate=request.frame_rate,
                box_type=request.box_type,
                image=image,
                lock=asyncio.Lock(),
                last_seen=now,
            )
            self._states[key] = state
            return state

    @staticmethod
    def _validate_stream_contract(state: StreamState, request: FrameRequest) -> None:
        actual = (request.width, request.height, request.frame_rate, request.box_type)
        expected = (state.width, state.height, state.frame_rate, state.box_type)
        if actual != expected:
            raise FrameConflictError(
                "Stream width, height, frame_rate, and box_type cannot change within a session. "
                "Use a new session ID for a different stream contract."
            )

    async def _is_current_state(self, key: StreamKey, state: StreamState) -> bool:
        async with self._registry_lock:
            return self._states.get(key) is state

    async def _discard_state(self, key: StreamKey, state: StreamState) -> None:
        async with self._registry_lock:
            if self._states.get(key) is state:
                self._states.pop(key, None)

    async def _evict_expired(self, now: float) -> None:
        async with self._registry_lock:
            cutoff = now - self.settings.stream_ttl_seconds
            expired = [
                (key, state)
                for key, state in self._states.items()
                if state.last_seen < cutoff and not state.lock.locked()
            ]
            for key, _ in expired:
                self._states.pop(key, None)
        for key, state in expired:
            await self._reset_state(state, reason=f"expired stream {key[0]}/{key[1]}")

    async def _reset_state(self, state: StreamState, *, reason: str) -> None:
        try:
            await asyncio.to_thread(state.tracker.reset)
        except Exception:
            LOGGER.exception("Failed to reset tracker after %s", reason)

    @staticmethod
    async def _update_without_releasing_lock(
        state: StreamState,
        detections: np.ndarray,
        image: np.ndarray | None,
    ) -> tuple[np.ndarray, bool]:
        """Drain a tracker thread before allowing cancellation to release its lock."""

        update_task = asyncio.create_task(
            asyncio.to_thread(
                state.input_adapter.update,
                detections,
                img=image,
            )
        )
        was_cancelled = False
        while not update_task.done():
            try:
                await asyncio.shield(update_task)
            except asyncio.CancelledError:
                was_cancelled = True
        return update_task.result(), was_cancelled

    @staticmethod
    def _request_digest(
        request: FrameRequest,
        detections: np.ndarray,
        image_digest: bytes,
    ) -> bytes:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(
            f"{request.frame_id}:{request.width}:{request.height}:{request.frame_rate}:{request.box_type.value}".encode()
        )
        digest.update(detections.tobytes(order="C"))
        digest.update(image_digest)
        return digest.digest()

    @staticmethod
    def _serialize_tracks(raw_tracks: np.ndarray, schema: BoxSchema) -> tuple[tuple[TrackValue, ...], ...]:
        tracks = np.asarray(raw_tracks, dtype=np.float32)
        if tracks.size == 0:
            tracks = schema.empty_tracks()
        if tracks.ndim != 2 or tracks.shape[1] != schema.track_cols:
            raise ValueError(
                f"Tracker returned shape {tracks.shape}; expected (N, {schema.track_cols}) for {schema.box_type.value}."
            )
        if not np.isfinite(tracks).all():
            raise ValueError("Tracker returned non-finite values.")

        integer_columns = {
            schema.track_id_index,
            schema.track_class_index,
            schema.track_detection_index,
        }
        for column in integer_columns:
            values = tracks[:, column]
            if not np.equal(values, np.floor(values)).all():
                raise ValueError("Tracker returned non-integer IDs or detection indices.")
        rows: list[tuple[TrackValue, ...]] = []
        for row in tracks:
            rows.append(
                tuple(int(value) if column in integer_columns else float(value) for column, value in enumerate(row))
            )
        return tuple(rows)


__all__ = (
    "DetectionValidationError",
    "FrameConflictError",
    "FrameResult",
    "ImageValidationError",
    "ReIDModelFactory",
    "ServiceRequestError",
    "StreamCapacityError",
    "TrackerExecutionError",
    "TrackerFactory",
    "TrackerManager",
)
