"""Timing collection and reporting primitives for engine-owned tracking workflows."""

from __future__ import annotations

import time
from collections.abc import Mapping
from math import isfinite
from typing import Any

from boxmot.utils import logger as LOGGER

DETECTOR_PHASES = ("preprocess", "process", "postprocess")
REID_PHASES = ("preprocess", "process", "postprocess")

SETUP_TIMING_COMPONENTS = (
    "detector_load",
    "tracker_reid_load",
    "reid_adapter",
    "output_prepare",
    "source_first_frame",
)

# Live tracking has a wider startup surface than the historical evaluation
# runner.  Keep its schema separate from ``SETUP_TIMING_COMPONENTS`` because
# the latter is also consumed by saved evaluation reports.
TRACK_STARTUP_TIMING_COMPONENTS = (
    "detector_load",
    "tracker_load",
    "reid_encoder_load",
    "segmentor_load",
    "pipeline_prepare",
    "output_prepare",
    "source_open",
    "source_first_frame",
)

TRACK_RUNTIME_TIMING_COMPONENTS = (
    "source_acquisition",
    "detector_preprocess",
    "detector_inference",
    "detector_postprocess",
    "detector_total",
    "segmentor_preprocess",
    "segmentor_inference",
    "segmentor_postprocess",
    "segmentor_total",
    "reid_preprocess",
    "reid_inference",
    "reid_postprocess",
    "reid_total",
    "enrichment",
    "validation",
    "tracker_association",
    "tracker_update",
    "tracker_total",
    "rendering",
    "sink_io",
    "other_overhead",
    "overall",
)

_TRACK_STARTUP_ALIASES = {
    "detector_load": ("detector_load",),
    "tracker_load": ("tracker_load", "tracker_reid_load"),
    "reid_encoder_load": ("reid_encoder_load", "reid_load", "reid_adapter"),
    "segmentor_load": ("segmentor_load",),
    "pipeline_prepare": ("pipeline_prepare",),
    "output_prepare": ("output_prepare", "outputs_prepare", "sink_prepare"),
    "source_open": ("source_open",),
    "source_first_frame": ("source_first_frame", "first_source_frame"),
}

_TRACK_RUNTIME_ALIASES = {
    "source_acquisition": ("source_acquisition", "source", "source_read", "source_io"),
    "detector_preprocess": ("detector_preprocess",),
    "detector_inference": ("detector_inference", "detector_process"),
    "detector_postprocess": ("detector_postprocess",),
    "detector_total": ("detector_total", "det"),
    "segmentor_preprocess": ("segmentor_preprocess",),
    "segmentor_inference": ("segmentor_inference", "segmentor_process"),
    "segmentor_postprocess": ("segmentor_postprocess",),
    "segmentor_total": ("segmentor_total", "segment"),
    "reid_preprocess": ("reid_preprocess",),
    "reid_inference": ("reid_inference", "reid_process"),
    "reid_postprocess": ("reid_postprocess",),
    "reid_total": ("reid_total", "reid"),
    "enrichment": ("enrichment", "pipeline_enrichment"),
    "validation": ("validation", "pipeline_validation"),
    "tracker_association": ("tracker_association",),
    "tracker_update": ("tracker_update", "tracker_association_update", "association_update", "track"),
    "tracker_total": ("tracker_total",),
    "rendering": ("rendering", "render", "plot"),
    "sink_io": ("sink_io", "sinks", "output_io"),
    "other_overhead": ("other_overhead", "overhead"),
    "overall": ("overall", "total"),
}

_DETECTOR_PHASE_KEYS = {
    "preprocess": "detector_preprocess",
    "process": "detector_process",
    "postprocess": "detector_postprocess",
}
_REID_PHASE_KEYS = {
    "preprocess": "reid_preprocess",
    "process": "reid_process",
    "postprocess": "reid_postprocess",
}


def normalize_setup_timings_ms(values: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Return startup timings with stable keys and a derived total."""

    source = values or {}
    normalized = {key: max(float(source.get(key, 0.0) or 0.0), 0.0) for key in SETUP_TIMING_COMPONENTS}
    normalized["total"] = sum(normalized.values())
    return normalized


def _nonnegative_ms(value: object) -> float:
    """Best-effort conversion for timing values collected during shutdown."""

    try:
        converted = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return converted if isfinite(converted) and converted > 0.0 else 0.0


def _aliased_timing_value(
    values: Mapping[str, Any],
    aliases: tuple[str, ...],
) -> float:
    for key in aliases:
        if key in values:
            return _nonnegative_ms(values[key])
    return 0.0


def normalize_track_startup_timings_ms(
    values: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Normalize the live-run startup snapshot and derive its total.

    This function is deliberately forgiving because it runs while presenting
    shutdown state, including a Ctrl-C received halfway through setup. Missing,
    negative, non-finite, or otherwise unusable fields render as zero rather
    than hiding the partial report.
    """

    source = values or {}
    normalized = {
        key: _aliased_timing_value(source, _TRACK_STARTUP_ALIASES[key]) for key in TRACK_STARTUP_TIMING_COMPONENTS
    }
    normalized["total"] = sum(normalized.values())
    return normalized


def normalize_track_runtime_timings_ms(
    values: Mapping[str, Any] | None = None,
    *,
    elapsed_ms: float | None = None,
) -> dict[str, float]:
    """Normalize a live run's exclusive runtime stages and derived totals.

    Component totals are never allowed to be smaller than the sum of their
    measured child phases. ``other_overhead`` is the unaccounted wall time, so
    the table remains honest when a partial run ends between timing events.
    """

    source = values or {}
    normalized: dict[str, float] = {}
    for key in TRACK_RUNTIME_TIMING_COMPONENTS:
        normalized[key] = _aliased_timing_value(source, _TRACK_RUNTIME_ALIASES[key])

    component_phases = {
        "detector_total": ("detector_preprocess", "detector_inference", "detector_postprocess"),
        "segmentor_total": ("segmentor_preprocess", "segmentor_inference", "segmentor_postprocess"),
        "reid_total": ("reid_preprocess", "reid_inference", "reid_postprocess"),
    }
    for total_key, phase_keys in component_phases.items():
        normalized[total_key] = max(
            normalized[total_key],
            sum(normalized[phase_key] for phase_key in phase_keys),
        )

    normalized["tracker_total"] = max(
        normalized["tracker_total"],
        normalized["tracker_association"] + normalized["tracker_update"],
    )

    supplied_elapsed = _nonnegative_ms(elapsed_ms) if elapsed_ms is not None else 0.0
    if supplied_elapsed > 0.0:
        normalized["overall"] = supplied_elapsed

    accounted_keys = (
        "source_acquisition",
        "detector_total",
        "segmentor_total",
        "reid_total",
        "enrichment",
        "validation",
        "tracker_total",
        "rendering",
        "sink_io",
    )
    accounted = sum(normalized[key] for key in accounted_keys)
    residual = max(0.0, normalized["overall"] - accounted)
    normalized["other_overhead"] = max(normalized["other_overhead"], residual)
    if normalized["overall"] == 0.0:
        normalized["overall"] = accounted + normalized["other_overhead"]

    return normalized


def build_track_runtime_display_rows(
    timings_ms: Mapping[str, Any] | None,
    frames: int,
    *,
    elapsed_ms: float | None = None,
    include_segmentor: bool | None = None,
    include_reid: bool | None = None,
) -> list[dict[str, object]]:
    """Build rows for the detailed live-tracking Rich timing table."""

    source = timings_ms or {}
    totals = normalize_track_runtime_timings_ms(source, elapsed_ms=elapsed_ms)
    frame_count = max(int(frames or 0), 0)
    if include_segmentor is None:
        include_segmentor = totals["segmentor_total"] > 0.0
    if include_reid is None:
        include_reid = totals["reid_total"] > 0.0

    def _group(label: str) -> dict[str, object]:
        return {"kind": "group", "label": label}

    def _row(
        label: str,
        key: str,
        *,
        strong: bool = False,
        additional_keys: tuple[str, ...] = (),
    ) -> dict[str, object]:
        total_ms = totals[key] + sum(totals[item] for item in additional_keys)
        avg_ms = total_ms / frame_count if frame_count else 0.0
        return {
            "kind": "row",
            "label": label,
            "total": total_ms,
            "avg": avg_ms,
            "fps": fps_from_avg_ms(avg_ms),
            "strong": strong,
        }

    rows = [
        _group("Source"),
        _row("  acquisition", "source_acquisition"),
        _group("Detector"),
        _row("  preprocess", "detector_preprocess"),
        _row("  inference", "detector_inference"),
        _row("  postprocess", "detector_postprocess"),
        _row("  Detector total", "detector_total", strong=True),
    ]
    if include_segmentor:
        rows.extend(
            [
                _group("Segmentor"),
                _row("  preprocess", "segmentor_preprocess"),
                _row("  inference", "segmentor_inference"),
                _row("  postprocess", "segmentor_postprocess"),
                _row("  Segmentor total", "segmentor_total", strong=True),
            ]
        )
    if include_reid:
        rows.extend(
            [
                _group("ReID encoder"),
                _row("  preprocess", "reid_preprocess"),
                _row("  inference", "reid_inference"),
                _row("  postprocess", "reid_postprocess"),
                _row("  ReID total", "reid_total", strong=True),
            ]
        )
    rows.extend(
        [
            _group("Pipeline"),
            _row("  enrichment", "enrichment"),
            _row("  validation", "validation"),
            _group("Tracker"),
            _row(
                "  association / update",
                "tracker_association",
                additional_keys=("tracker_update",),
            ),
            _row("  Tracker total", "tracker_total", strong=True),
            _group("Output"),
            _row("  rendering", "rendering"),
            _row("  sink I/O", "sink_io"),
            _row("Other overhead", "other_overhead"),
            _row("Overall", "overall", strong=True),
        ]
    )
    return rows


def fps_from_avg_ms(avg_ms: float) -> float:
    """Convert an average per-frame latency in milliseconds to FPS."""
    avg_ms = float(avg_ms or 0.0)
    return (1000.0 / avg_ms) if avg_ms > 0.0 else 0.0


def derive_timing_breakdown(
    totals: dict[str, float],
    frames: int,
    *,
    total_time_ms: float | None = None,
) -> dict[str, float | bool]:
    """Derive consistent timing buckets across tracking / eval renderers.

    ``track`` may either include ReID time (online tracking) or represent only
    non-ReID tracker work (cached benchmark replay). Reuse the existing batch
    heuristic so all timing reports label the same buckets consistently.
    """
    normalized = {
        key: float(totals.get(key, 0.0) or 0.0)
        for key in (
            "det",
            "reid",
            "track",
            "plot",
            "total",
            "detector_preprocess",
            "detector_process",
            "detector_postprocess",
            "reid_preprocess",
            "reid_process",
            "reid_postprocess",
        )
    }

    detector_has_split = any(normalized[_DETECTOR_PHASE_KEYS[phase]] > 0.0 for phase in DETECTOR_PHASES)
    if detector_has_split:
        detector_preprocess_total = normalized["detector_preprocess"]
        detector_process_total = normalized["detector_process"]
        detector_postprocess_total = normalized["detector_postprocess"]
    else:
        detector_preprocess_total = 0.0
        detector_process_total = normalized["det"]
        detector_postprocess_total = 0.0

    reid_has_split = any(normalized[_REID_PHASE_KEYS[phase]] > 0.0 for phase in REID_PHASES)
    if reid_has_split:
        reid_preprocess_total = normalized["reid_preprocess"]
        reid_process_total = normalized["reid_process"]
        reid_postprocess_total = normalized["reid_postprocess"]
    else:
        reid_preprocess_total = 0.0
        reid_process_total = normalized["reid"]
        reid_postprocess_total = 0.0

    det_total = detector_preprocess_total + detector_process_total + detector_postprocess_total
    reid_total = reid_preprocess_total + reid_process_total + reid_postprocess_total
    track_total = normalized["track"]
    plot_total = normalized["plot"]

    total_total = float(total_time_ms if total_time_ms not in {None, 0.0} else normalized["total"])
    if total_total == 0.0:
        total_total = det_total + reid_total + track_total + plot_total

    is_batch_mode = int(frames or 0) == 0 or (reid_total > 0.0 and det_total > 0.0)
    tracker_rest_total = track_total if is_batch_mode else max(0.0, track_total - reid_total)
    tracker_total = (reid_total + track_total) if is_batch_mode else track_total

    accounted_total = det_total + reid_total + track_total + plot_total
    overhead_total = max(0.0, total_total - accounted_total)

    return {
        "is_batch_mode": is_batch_mode,
        "detector_preprocess_total": detector_preprocess_total,
        "detector_process_total": detector_process_total,
        "detector_postprocess_total": detector_postprocess_total,
        "det_total": det_total,
        "reid_preprocess_total": reid_preprocess_total,
        "reid_process_total": reid_process_total,
        "reid_postprocess_total": reid_postprocess_total,
        "reid_total": reid_total,
        "track_total": track_total,
        "tracker_rest_total": tracker_rest_total,
        "tracker_total": tracker_total,
        "plot_total": plot_total,
        "total_total": total_total,
        "overhead_total": overhead_total,
    }


def build_timing_display_rows(
    breakdown: dict[str, float | bool],
    frames: int,
    *,
    metadata: dict[str, object] | None = None,
    overall_avg_ms: float | None = None,
    overall_fps: float | None = None,
) -> list[dict[str, object]]:
    """Build grouped detector/tracker timing rows for UI summaries."""
    frame_count = int(frames or 0)
    metadata = metadata or {}

    def _row(
        label: str,
        total_ms: float,
        *,
        strong: bool = False,
        avg_ms: float | None = None,
        fps: float | None = None,
    ) -> dict[str, object]:
        total_value = float(total_ms or 0.0)
        avg_value = float(avg_ms if avg_ms is not None else (total_value / frame_count if frame_count else 0.0))
        fps_value = float(fps if fps is not None else fps_from_avg_ms(avg_value))
        return {
            "kind": "row",
            "label": label,
            "total": total_value,
            "avg": avg_value,
            "fps": fps_value,
            "strong": strong,
        }

    def _note(label: str) -> dict[str, object]:
        return {
            "kind": "note",
            "label": label,
        }

    detector_cached = bool(metadata.get("detector_from_cache")) and float(breakdown["det_total"]) == 0.0
    reid_cached = bool(metadata.get("reid_from_cache")) and float(breakdown["reid_total"]) == 0.0

    detector_rows: list[dict[str, object]] = [{"kind": "group", "label": "Detector"}]
    if detector_cached:
        detector_rows.append(_note("  detections loaded from cache (0 ms)"))
        detector_rows.append(_row("  Detector total", float(breakdown["det_total"]), strong=True))
    else:
        detector_rows.extend(
            [
                _row("  preprocess", float(breakdown["detector_preprocess_total"])),
                _row("  process", float(breakdown["detector_process_total"])),
                _row("  postprocess", float(breakdown["detector_postprocess_total"])),
                _row("  Detector total", float(breakdown["det_total"]), strong=True),
            ]
        )

    tracker_rows: list[dict[str, object]] = [{"kind": "group", "label": "Tracker"}]
    if reid_cached:
        tracker_rows.append(_note("  embeddings loaded from cache (0 ms)"))
    else:
        tracker_rows.extend(
            [
                _row("  ReID preprocess", float(breakdown["reid_preprocess_total"])),
                _row("  ReID process", float(breakdown["reid_process_total"])),
                _row("  ReID postprocess", float(breakdown["reid_postprocess_total"])),
            ]
        )
    tracker_rows.extend(
        [
            _row("  association/update", float(breakdown["tracker_rest_total"])),
            _row("  Tracker total", float(breakdown["tracker_total"]), strong=True),
        ]
    )

    return [
        *detector_rows,
        *tracker_rows,
        _row(
            "Overall total",
            float(breakdown["total_total"]),
            strong=True,
            avg_ms=overall_avg_ms,
            fps=overall_fps,
        ),
    ]


class TimingStats:
    """Track timing statistics for detection, ReID, and tracking phases."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.totals: dict[str, float] = {
            "detector_preprocess": 0.0,
            "detector_process": 0.0,
            "detector_postprocess": 0.0,
            "reid": 0.0,
            "reid_preprocess": 0.0,
            "reid_process": 0.0,
            "reid_postprocess": 0.0,
            "track": 0.0,
            "plot": 0.0,
            "total": 0.0,
        }
        self.metadata = {}
        self.frames = 0
        self._frame_start = None
        self._track_start = None
        self._plot_start = None

    def start_frame(self):
        """Mark the start of frame processing."""
        self._frame_start = time.perf_counter()

    def start_tracking(self):
        """Mark the start of tracking phase."""
        self._track_start = time.perf_counter()

    def end_tracking(self):
        """Mark the end of tracking phase and record time."""
        if self._track_start is not None:
            elapsed = (time.perf_counter() - self._track_start) * 1000
            self.totals["track"] += elapsed
            self._last_track_time = elapsed
            self._track_start = None

    def get_last_track_time(self):
        """Get the last tracking time in ms."""
        return getattr(self, "_last_track_time", 0)

    def get_last_reid_time(self):
        """Get the last ReID time in ms (accumulated during last track update)."""
        return getattr(self, "_last_reid_time", 0)

    def reset_frame_reid(self):
        """Reset per-frame ReID accumulator (call before each track update)."""
        self._last_reid_time = 0

    def start_plot(self):
        """Mark the start of plotting phase."""
        self._plot_start = time.perf_counter()

    def end_plot(self):
        """Mark the end of plotting phase and record time."""
        if self._plot_start is not None:
            self.totals["plot"] += (time.perf_counter() - self._plot_start) * 1000
            self._plot_start = None

    def add_reid_time(self, time_ms):
        """Add ReID time in milliseconds."""
        self.totals["reid"] += time_ms
        # Also accumulate for per-frame tracking
        self._last_reid_time = getattr(self, "_last_reid_time", 0) + time_ms

    def add_detector_phase_time(self, phase: str, time_ms: float) -> None:
        """Record detector phase timing."""
        phase_key = str(phase).strip().lower()
        specific_key = _DETECTOR_PHASE_KEYS[phase_key]
        elapsed_ms = float(time_ms or 0.0)
        self.totals[specific_key] += elapsed_ms

    def add_reid_phase_time(self, phase: str, time_ms: float) -> None:
        """Record ReID phase timing and total per-frame ReID time."""
        phase_key = str(phase).strip().lower()
        specific_key = _REID_PHASE_KEYS[phase_key]
        elapsed_ms = float(time_ms or 0.0)
        self.totals[specific_key] += elapsed_ms
        self.add_reid_time(elapsed_ms)

    def record_ultralytics_times(self, predictor):
        """Record timing from Ultralytics results."""
        # Ultralytics stores speed info in results[].speed dict
        # speed contains: preprocess, inference, postprocess times in ms
        if hasattr(predictor, "results") and predictor.results:
            for result in predictor.results:
                if hasattr(result, "speed") and result.speed:
                    self.add_detector_phase_time("preprocess", result.speed.get("preprocess", 0) or 0)
                    self.add_detector_phase_time("process", result.speed.get("inference", 0) or 0)
                    self.add_detector_phase_time("postprocess", result.speed.get("postprocess", 0) or 0)

    def end_frame(self):
        """Mark the end of frame processing."""
        if self._frame_start is not None:
            self.totals["total"] += (time.perf_counter() - self._frame_start) * 1000
            self.frames += 1
            self._frame_start = None

    def format_summary(self) -> str:
        """Return a plain-text execution time summary table."""
        # Check if we have any data to display
        has_data = any(v > 0 for v in self.totals.values())
        if not has_data:
            return ""

        frames = self.frames if self.frames > 0 else 1  # Avoid division by zero

        breakdown = derive_timing_breakdown(self.totals, self.frames, total_time_ms=self.totals["total"])
        total_time = float(breakdown["total_total"])
        plot_time = float(breakdown["plot_total"])
        overhead = float(breakdown["overhead_total"])

        # Helper to calculate percentage
        def pct(value):
            return (value / total_time * 100) if total_time > 0 else 0

        lines = [
            "=" * 105,
            f"{'📊 TIMING SUMMARY':^105}",
            "=" * 105,
            f"{'Stage':<20} | {'Total (ms)':<12} | {'Avg (ms)':<12} | {'FPS':<10} | {'% of Total':<12}",
            "-" * 105,
        ]

        for entry in build_timing_display_rows(
            breakdown,
            frames,
            metadata=dict(getattr(self, "metadata", {})),
            overall_avg_ms=(total_time / frames if frames else 0.0),
            overall_fps=fps_from_avg_ms(total_time / frames if frames else 0.0),
        ):
            if entry["kind"] == "group":
                lines.append(str(entry["label"]))
                continue
            if entry["kind"] == "note":
                lines.append(str(entry["label"]))
                continue
            total = float(entry["total"])
            avg = float(entry["avg"])
            fps = float(entry["fps"])
            lines.append(
                f"{str(entry['label']):<20} | {total:<12.1f} | {avg:<12.2f} | {fps:<10.1f} | {pct(total):<12.1f}"
            )

        # Plotting and overhead
        if plot_time > 0:
            plot_avg = plot_time / frames
            plot_fps = fps_from_avg_ms(plot_avg)
            lines.append(
                f"{'Plotting':<20} | {plot_time:<12.1f} | {plot_avg:<12.2f} | {plot_fps:<10.1f} | "
                f"{pct(plot_time):<12.1f}"
            )

        if overhead > 0:
            overhead_avg = overhead / frames
            overhead_fps = fps_from_avg_ms(overhead_avg)
            lines.append(
                f"{'Other (I/O, etc)':<20} | {overhead:<12.1f} | {overhead_avg:<12.2f} | "
                f"{overhead_fps:<10.1f} | {pct(overhead):<12.1f}"
            )

        lines.append(f"{'Frames':<20} | {frames:<12}")
        lines.append("=" * 105)
        return "\n".join(lines)

    def print_summary(self):
        """Print execution time summary table with blue color palette."""
        summary = self.format_summary()
        if not summary:
            return
        LOGGER.info("\n%s", summary)

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize timing stats into a JSON-friendly summary dict."""
        totals = dict(self.totals)
        total_ms = float(totals.get("total", 0.0) or 0.0)
        if total_ms == 0.0:
            total_ms = float(sum(totals.values()))

        frames = int(self.frames)
        avg_ms = {key: (float(value) / frames if frames else 0.0) for key, value in totals.items()}
        avg_total_ms = total_ms / frames if frames else 0.0
        fps = (1000.0 * frames / total_ms) if total_ms else 0.0

        return {
            "frames": frames,
            "totals_ms": {**{key: float(value) for key, value in totals.items()}, "total": total_ms},
            "avg_ms": {**avg_ms, "total": avg_total_ms},
            "fps": fps,
            "metadata": dict(getattr(self, "metadata", {})),
        }


__all__ = (
    "DETECTOR_PHASES",
    "REID_PHASES",
    "SETUP_TIMING_COMPONENTS",
    "TRACK_RUNTIME_TIMING_COMPONENTS",
    "TRACK_STARTUP_TIMING_COMPONENTS",
    "TimingStats",
    "build_track_runtime_display_rows",
    "build_timing_display_rows",
    "derive_timing_breakdown",
    "fps_from_avg_ms",
    "normalize_setup_timings_ms",
    "normalize_track_runtime_timings_ms",
    "normalize_track_startup_timings_ms",
)
