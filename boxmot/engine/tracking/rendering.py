from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from boxmot.trackers.results import TrackResults

try:
    from ultralytics.utils.plotting import colors
except ImportError:
    colors = None


Drawer = Callable[[np.ndarray, np.ndarray], np.ndarray]


def track_color(track_id: int) -> tuple[int, int, int]:
    if colors is not None:
        color = colors(track_id, True)
        return int(color[0]), int(color[1]), int(color[2])
    base = (int(track_id) * 123457) % 255
    return int(base), int((base * 3) % 255), int((base * 7) % 255)


def draw_tracks(frame: np.ndarray, tracks: TrackResults, masks: np.ndarray | None = None) -> np.ndarray:
    drawn = frame.copy()
    if tracks.size == 0:
        return drawn

    if masks is not None:
        _draw_masks(drawn, tracks, masks)

    for i in range(len(tracks)):
        track_id = int(tracks.id[i])
        conf = float(tracks.conf[i])
        color = track_color(track_id)

        if tracks.is_obb:
            cx, cy, width, height, angle = tracks.xywha[i]
            rect = (
                (float(cx), float(cy)),
                (max(float(width), 1.0), max(float(height), 1.0)),
                float(np.degrees(angle)),
            )
            corners = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(drawn, [corners], True, color, 2)
            label_point = tuple(corners[0])
        else:
            x1, y1, x2, y2 = tracks.xyxy[i].round().astype(int)
            cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
            label_point = (x1, max(0, y1 - 6))

        cv2.putText(
            drawn,
            f"{track_id} {conf:.2f}",
            label_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return drawn


def _draw_masks(drawn: np.ndarray, tracks: TrackResults, masks: np.ndarray) -> None:
    h_frame, w_frame = drawn.shape[:2]
    for i in range(len(tracks)):
        mask = masks[i]
        if mask.sum() == 0:
            continue
        color = track_color(int(tracks.id[i]))
        h_mask, w_mask = mask.shape[:2]
        if (h_mask, w_mask) != (h_frame, w_frame):
            mask = cv2.resize(mask, (w_frame, h_frame), interpolation=cv2.INTER_NEAREST)
        colored = np.zeros_like(drawn)
        colored[:] = color
        mask_bool = mask.astype(bool)
        drawn[mask_bool] = cv2.addWeighted(drawn, 0.6, colored, 0.4, 0)[mask_bool]


__all__ = ("Drawer", "draw_tracks", "track_color")
