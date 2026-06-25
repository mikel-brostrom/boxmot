from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

_VIDEO_WRITERS: dict[str, cv2.VideoWriter] = {}


def _resolve_fps(source: Any) -> float:
    from boxmot.engine.workflows.support import resolve_output_fps

    return resolve_output_fps(source)


def append_frame(path: str | Path, rendered: np.ndarray, source: Any, fps: float | None = None) -> None:
    key = str(Path(path))
    if key not in _VIDEO_WRITERS:
        if fps is None:
            fps = _resolve_fps(source)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        h, w = rendered.shape[:2]
        _VIDEO_WRITERS[key] = cv2.VideoWriter(
            key,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
    _VIDEO_WRITERS[key].write(rendered)


def close(path: str | Path | None = None) -> None:
    if path is not None:
        key = str(Path(path))
        writer = _VIDEO_WRITERS.pop(key, None)
        if writer is not None:
            writer.release()
        return

    for writer in _VIDEO_WRITERS.values():
        writer.release()
    _VIDEO_WRITERS.clear()


__all__ = ("append_frame", "close")
