from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path, Response, status

from boxmot.engine.service.config import ServiceSettings
from boxmot.engine.service.manager import (
    FrameConflictError,
    ServiceRequestError,
    StreamCapacityError,
    TrackerExecutionError,
    TrackerFactory,
    TrackerManager,
)
from boxmot.engine.service.models import FrameRequest, FrameResponse, ReadinessResponse

SessionIdentifier = Annotated[
    str,
    Path(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    ),
]


def create_app(
    settings: ServiceSettings | None = None,
    *,
    tracker_factory: TrackerFactory | None = None,
) -> FastAPI:
    """Create a tracker service with isolated, in-memory stream state."""

    resolved_settings = settings or ServiceSettings.from_env()
    manager = TrackerManager(resolved_settings, tracker_factory=tracker_factory)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await manager.close()

    application = FastAPI(
        title="BoxMOT Tracker Service",
        version="1.0.0",
        description="Stateful multi-object tracking from externally supplied detections.",
        lifespan=lifespan,
    )
    application.state.tracker_manager = manager

    @application.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/readyz", response_model=ReadinessResponse)
    async def readiness() -> ReadinessResponse:
        capacity = await manager.stats()
        return ReadinessResponse(
            status="ready",
            profile=resolved_settings.profile,
            tracker=resolved_settings.tracker_type,
            device=resolved_settings.device,
            requires_image=resolved_settings.requires_image,
            **capacity,
        )

    @application.post(
        "/v1/streams/{stream_id}/sessions/{session_id}/frames",
        response_model=FrameResponse,
    )
    async def track_frame(
        stream_id: SessionIdentifier,
        session_id: SessionIdentifier,
        frame: FrameRequest,
    ) -> FrameResponse:
        try:
            result = await manager.process((stream_id, session_id), frame)
        except FrameConflictError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except StreamCapacityError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
                headers={"Retry-After": "1"},
            ) from exc
        except ServiceRequestError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        except TrackerExecutionError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

        columns = (
            ["cx", "cy", "w", "h", "angle", "id", "confidence", "class_id", "detection_index"]
            if result.box_type.value == "obb"
            else ["x1", "y1", "x2", "y2", "id", "confidence", "class_id", "detection_index"]
        )
        return FrameResponse(
            frame_id=result.frame_id,
            next_frame_id=result.next_frame_id,
            box_type=result.box_type,
            track_columns=columns,
            tracks=[list(row) for row in result.tracks],
            replayed=result.replayed,
        )

    @application.delete(
        "/v1/streams/{stream_id}/sessions/{session_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_session(
        stream_id: SessionIdentifier,
        session_id: SessionIdentifier,
    ) -> Response:
        await manager.delete((stream_id, session_id))
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return application


app = create_app()


__all__ = ("app", "create_app")
