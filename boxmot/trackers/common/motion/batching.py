"""Batch Kalman arithmetic while retaining each track's lifecycle bookkeeping."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _batch_geometry_hook(track: Any, method_name: str) -> bool:
    """Recognize an opted-in native method without bypassing custom hooks."""
    method = getattr(track, method_name)
    for cls in type(track).__mro__:
        if method_name in cls.__dict__:
            return bool(cls.__dict__.get("_batch_prediction_geometry", False)) and (
                getattr(method, "__func__", None) is cls.__dict__[method_name]
            )
    return False


def _batch_state_geometry(track: Any) -> bool:
    """Respect native state getters and DeepOCSORT's replaceable converter."""
    if not _batch_geometry_hook(track, "get_state"):
        return False
    converter = getattr(track, "x_to_bbox_func", None)
    return converter is None or (
        getattr(converter, "__self__", None) is track.motion_model
        and getattr(converter, "__func__", None) is type(track.motion_model).to_box
    )


def _batch_prediction_hook(track: Any) -> bool:
    """Use supplied boxes only when both lifecycle and geometry are native."""
    return _batch_geometry_hook(track, "_finish_prediction") and _batch_state_geometry(track)


def current_track_boxes(tracks: Sequence[Any], *, is_obb: bool) -> np.ndarray:
    """Convert current filter states in model groups, retaining track order."""
    boxes = np.empty((len(tracks), 5 if is_obb else 4), dtype=float)
    groups: dict[tuple, tuple[Any, list[int]]] = {}
    for index, track in enumerate(tracks):
        if not _batch_state_geometry(track):
            boxes[index] = track.get_state()[0]
            continue
        model = track.motion_model
        _, indices = groups.setdefault((model.kind, model.is_obb), (model, []))
        indices.append(index)
    for model, indices in groups.values():
        states = np.asarray([np.asarray(tracks[index].kf.x).reshape(-1) for index in indices])
        boxes[indices] = model.to_boxes(states)
    return boxes


def observation_output_boxes(tracks: Sequence[Any], *, is_obb: bool) -> np.ndarray:
    """Use current observations for output, batching states of new tracks."""
    boxes = np.empty((len(tracks), 5 if is_obb else 4), dtype=float)
    predicted = []
    for index, track in enumerate(tracks):
        if track.last_observation[-1] < 0:
            predicted.append(index)
        else:
            boxes[index] = track.last_observation[: boxes.shape[1]]
    if predicted:
        boxes[predicted] = current_track_boxes([tracks[index] for index in predicted], is_obb=is_obb)
    return boxes


def predict_tracks(tracks: Sequence[Any], *, dt: float | None = None) -> list[Any]:
    """Predict independent filters, then complete each track's frame counters."""
    groups: dict[tuple, tuple[Any, list[Any], list[int]]] = {}
    for index, track in enumerate(tracks):
        track._prepare_prediction(dt=dt)
        model = track.motion_model
        observation_centric = track.observation_centric
        _, filters, indices = groups.setdefault((model.kind, model.is_obb, observation_centric), (model, [], []))
        filters.append(track.kf)
        indices.append(index)
    boxes: dict[int, np.ndarray] = {}
    for (_, _, observation_centric), (model, filters, indices) in groups.items():
        model.predict_filters(filters, dt=dt, observation_centric=observation_centric)
        batch_indices = [index for index in indices if _batch_prediction_hook(tracks[index])]
        if batch_indices:
            states = np.asarray([np.asarray(tracks[index].kf.x).reshape(-1) for index in batch_indices])
            geometry = model.to_boxes(states)
            boxes.update((index, row.reshape(1, -1)) for index, row in zip(batch_indices, geometry, strict=True))
    return [
        track._finish_prediction(box=boxes[index]) if index in boxes else track._finish_prediction()
        for index, track in enumerate(tracks)
    ]


def update_tracks(tracks: Sequence[Any], *arguments: Sequence[Any], **keywords: Sequence[Any]) -> None:
    """Correct tracks from aligned columns of their scalar update arguments.

    Preparation preserves algorithm-specific observation and appearance state;
    completion records the corrected geometry after the grouped Kalman solve.
    """
    if any(len(column) != len(tracks) for column in (*arguments, *keywords.values())):
        raise ValueError("Every update argument must contain one value per track")
    groups: dict[tuple, tuple[Any, list[Any], list[Any], list[float]]] = {}
    measurements = []
    for index, track in enumerate(tracks):
        measurement = track._prepare_update(
            *(column[index] for column in arguments),
            **{name: column[index] for name, column in keywords.items()},
        )
        measurements.append(measurement)
        model = track.motion_model
        _, filters, observations, confidences = groups.setdefault(
            (model.kind, model.is_obb, track.observation_centric), (model, [], [], [])
        )
        filters.append(track.kf)
        observations.append(measurement)
        confidences.append(float(track.conf))
    for (_, _, observation_centric), (model, filters, observations, confidences) in groups.items():
        model.update_filters(filters, observations, observation_centric=observation_centric, confidence=confidences)
    for track, measurement in zip(tracks, measurements, strict=True):
        track._finish_update(measurement)
