"""Batch Kalman arithmetic while retaining each track's lifecycle bookkeeping."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def predict_tracks(tracks: Sequence[Any], *, dt: float | None = None) -> list[Any]:
    """Predict independent filters, then complete each track's frame counters."""
    groups: dict[type, list[Any]] = {}
    for track in tracks:
        track._prepare_prediction(dt=dt)
        groups.setdefault(type(track.kf), []).append(track.kf)
    for filter_type, filters in groups.items():
        filter_type.predict_many(filters, dt=dt)
    return [track._finish_prediction() for track in tracks]


def update_tracks(tracks: Sequence[Any], *arguments: Sequence[Any], **keywords: Sequence[Any]) -> None:
    """Correct tracks from aligned columns of their scalar update arguments.

    Preparation preserves algorithm-specific observation and appearance state;
    completion records the corrected geometry after the grouped Kalman solve.
    """
    if any(len(column) != len(tracks) for column in (*arguments, *keywords.values())):
        raise ValueError("Every update argument must contain one value per track")
    groups: dict[type, tuple[list[Any], list[Any]]] = {}
    measurements = []
    for index, track in enumerate(tracks):
        measurement = track._prepare_update(
            *(column[index] for column in arguments),
            **{name: column[index] for name, column in keywords.items()},
        )
        measurements.append(measurement)
        filters, observations = groups.setdefault(type(track.kf), ([], []))
        filters.append(track.kf)
        observations.append(measurement)
    for filter_type, (filters, observations) in groups.items():
        filter_type.update_many(filters, observations)
    for track, measurement in zip(tracks, measurements, strict=True):
        track._finish_update(measurement)
