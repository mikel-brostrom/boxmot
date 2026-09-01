from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol, TypeAlias

import numpy as np

from boxmot.core.box_schema import BoxSchema, BoxType, get_box_schema
from boxmot.engine.service.config import ServiceSettings
from boxmot.engine.service.models import FrameRequest
from boxmot.trackers.registry import create_tracker
from boxmot.utils import logger as LOGGER

StreamKey: TypeAlias = tuple[str, str]
TrackValue: TypeAlias = float | int


class TrackerProtocol(Protocol):
    """Small tracker surface required by the service manager."""

    def update(self, dets: np.ndarray, img: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


TrackerFactory: TypeAlias = Callable[[int], TrackerProtocol]


class ServiceRequestError(Exception):
    """Base class for errors caused by a service request."""


class DetectionValidationError(ServiceRequestError):
    """A detection matrix does not satisfy the public BoxMOT contract."""


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
    width: int
    height: int
    frame_rate: int
    box_type: BoxType
    image: np.ndarray
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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self._tracker_factory = tracker_factory or self._build_tracker
        self._clock = clock
        self._states: dict[StreamKey, StreamState] = {}
        self._registry_lock = asyncio.Lock()
        self._update_slots = asyncio.Semaphore(settings.max_concurrent_updates)

    def _build_tracker(self, frame_rate: int) -> TrackerProtocol:
        tracker_kwargs = {"frame_rate": frame_rate} if self.settings.tracker_type == "bytetrack" else None
        return create_tracker(
            self.settings.tracker_type,
            per_class=True,
            tracker_backend="python",
            tracker_kwargs=tracker_kwargs,
        )

    async def process(self, key: StreamKey, request: FrameRequest) -> FrameResult:
        """Validate and process one sequential frame for ``key``."""

        detections, schema = self._prepare_detections(request)
        request_digest = self._request_digest(request, detections)
        state = await self._get_or_create_state(key, request)

        async with state.lock:
            if not await self._is_current_state(key, state):
                raise FrameConflictError("The tracker session was reset while this request was waiting.")
            self._validate_stream_contract(state, request)

            if state.last_frame_id is not None and request.frame_id == state.last_frame_id:
                if request_digest != state.last_request_digest or state.last_result is None:
                    raise FrameConflictError(
                        f"Frame {request.frame_id} was already processed with different input."
                    )
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
                    "A tracker session may contain at most "
                    f"{self.settings.max_classes_per_stream} distinct class IDs."
                )

            try:
                await self._update_slots.acquire()
                try:
                    raw_tracks, was_cancelled = await self._update_without_releasing_lock(
                        state,
                        detections,
                    )
                finally:
                    self._update_slots.release()
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
                "A frame may contain at most "
                f"{self.settings.max_classes_per_stream} distinct class IDs."
            )

        return values.astype(np.float32), schema

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
                    "The tracker session does not exist or has expired. "
                    "Start a new session with frame 0."
                )
            if len(self._states) >= self.settings.max_streams:
                raise StreamCapacityError(
                    f"Tracker capacity reached ({self.settings.max_streams} active streams)."
                )

            image = np.broadcast_to(
                np.zeros((1, 1, 3), dtype=np.uint8),
                (request.height, request.width, 3),
            )
            state = StreamState(
                tracker=self._tracker_factory(request.frame_rate),
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
    ) -> tuple[np.ndarray, bool]:
        """Drain a tracker thread before allowing cancellation to release its lock."""

        update_task = asyncio.create_task(
            asyncio.to_thread(state.tracker.update, detections, state.image)
        )
        was_cancelled = False
        while not update_task.done():
            try:
                await asyncio.shield(update_task)
            except asyncio.CancelledError:
                was_cancelled = True
        return update_task.result(), was_cancelled

    @staticmethod
    def _request_digest(request: FrameRequest, detections: np.ndarray) -> bytes:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(
            f"{request.frame_id}:{request.width}:{request.height}:{request.frame_rate}:{request.box_type.value}".encode()
        )
        digest.update(detections.tobytes(order="C"))
        return digest.digest()

    @staticmethod
    def _serialize_tracks(raw_tracks: np.ndarray, schema: BoxSchema) -> tuple[tuple[TrackValue, ...], ...]:
        tracks = np.asarray(raw_tracks, dtype=np.float32)
        if tracks.size == 0:
            tracks = schema.empty_tracks()
        if tracks.ndim != 2 or tracks.shape[1] != schema.track_cols:
            raise ValueError(
                f"Tracker returned shape {tracks.shape}; expected (N, {schema.track_cols}) "
                f"for {schema.box_type.value}."
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
                tuple(
                    int(value) if column in integer_columns else float(value)
                    for column, value in enumerate(row)
                )
            )
        return tuple(rows)


__all__ = (
    "DetectionValidationError",
    "FrameConflictError",
    "FrameResult",
    "ServiceRequestError",
    "StreamCapacityError",
    "TrackerExecutionError",
    "TrackerFactory",
    "TrackerManager",
)
