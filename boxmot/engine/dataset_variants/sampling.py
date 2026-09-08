"""Seeded frame loss that preserves the source capture timeline."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

PROFILE = "bursty-v1"


@dataclass(frozen=True, slots=True)
class FrameSelection:
    """Selected zero-based source indices and the simulated outage bounds."""

    indices: tuple[int, ...]
    outage_start_index: int
    outage_end_index: int


def select_bursty_frames(timestamps: Sequence[float | None], *, seed: int = 0) -> FrameSelection:
    """Simulate throttling and loss without inventing image capture times.

    A nominal 30 FPS sequence alternates mostly full-rate delivery with roughly
    10 and 15 FPS phases. One approximately 300 ms outage occurs near 60% of
    the sequence. Every retained frame keeps its original timestamp.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if len(timestamps) < 30:
        raise ValueError("A time variant needs at least 30 source frames.")
    if any(value is None or not math.isfinite(value) for value in timestamps):
        raise ValueError("Every source frame needs a finite capture timestamp.")
    times = tuple(float(value) for value in timestamps)
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("Source capture timestamps must increase strictly.")

    rng = random.Random(seed)
    count = len(times)
    jitter = max(1, round(0.02 * count))
    outage_start = min(count - 2, max(1, round(0.60 * count) + rng.randint(-jitter, jitter)))
    outage_end = outage_start
    while outage_end < count - 1 and times[outage_end] - times[outage_start] < 0.3:
        outage_end += 1
    indices = [0]
    index = 0
    while index < count - 1:
        fraction = index / count
        if 0.18 <= fraction < 0.37:
            stride = rng.choices((2, 3, 4), weights=(0.2, 0.6, 0.2))[0]
        elif 0.69 <= fraction < 0.83:
            stride = rng.choices((1, 2, 3), weights=(0.1, 0.8, 0.1))[0]
        else:
            stride = rng.choices((1, 2), weights=(0.9, 0.1))[0]
        index = min(count - 1, index + stride)
        if not outage_start <= index < outage_end:
            indices.append(index)
    return FrameSelection(tuple(indices), outage_start, outage_end)


def timing_statistics(timestamps: Sequence[float], selection: FrameSelection) -> dict[str, Any]:
    """Summarize actual retained intervals, never processing throughput."""
    retained = [timestamps[index] for index in selection.indices]
    gaps = [right - left for left, right in zip(retained, retained[1:])]
    duration = retained[-1] - retained[0]
    histogram = Counter(round(gap * 1000, 6) for gap in gaps)
    return {
        "source_frames": len(timestamps),
        "retained_frames": len(retained),
        "dropped_frames": len(timestamps) - len(retained),
        "duration_s": duration,
        "effective_fps": (len(retained) - 1) / duration,
        "min_dt_s": min(gaps),
        "median_dt_s": median(gaps),
        "max_dt_s": max(gaps),
        "interval_histogram": [{"dt_ms": gap, "count": count} for gap, count in sorted(histogram.items())],
    }
