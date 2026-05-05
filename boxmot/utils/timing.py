# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import time
from typing import Any

import numpy as np

from boxmot.utils import logger as LOGGER
from boxmot.utils.rich.ui import print_text


DETECTOR_PHASES = ("preprocess", "process", "postprocess")
REID_PHASES = ("preprocess", "process", "postprocess")

_DETECTOR_PHASE_KEYS = {
    "preprocess": "detector_preprocess",
    "process": "detector_process",
    "postprocess": "detector_postprocess",
}
_LEGACY_DETECTOR_KEYS = {
    "preprocess": "preprocess",
    "process": "inference",
    "postprocess": "postprocess",
}
_REID_PHASE_KEYS = {
    "preprocess": "reid_preprocess",
    "process": "reid_process",
    "postprocess": "reid_postprocess",
}


def fps_from_avg_ms(avg_ms: float) -> float:
    """Convert an average per-frame latency in milliseconds to FPS."""
    avg_ms = float(avg_ms or 0.0)
    return (1000.0 / avg_ms) if avg_ms > 0.0 else 0.0


def timed_reid_get_features(model, timing_stats: "TimingStats", *args, **kwargs):
    """Time ReID feature extraction at preprocess/process/postprocess granularity when possible."""
    if not all(
        hasattr(model, attr)
        for attr in ("get_crops", "inference_preprocess", "forward", "inference_postprocess")
    ):
        t0 = time.perf_counter()
        result = model.get_features(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        timing_stats.add_reid_phase_time("process", elapsed_ms)
        return result

    if len(args) < 2:
        t0 = time.perf_counter()
        result = model.get_features(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        timing_stats.add_reid_phase_time("process", elapsed_ms)
        return result

    xyxys, img = args[0], args[1]
    boxes = np.asarray(xyxys)
    if boxes.size == 0:
        return np.array([])

    preprocess_started = time.perf_counter()
    crops = model.get_crops(boxes, img)
    crops = model.inference_preprocess(crops)
    timing_stats.add_reid_phase_time("preprocess", (time.perf_counter() - preprocess_started) * 1000)

    process_started = time.perf_counter()
    features = model.forward(crops)
    timing_stats.add_reid_phase_time("process", (time.perf_counter() - process_started) * 1000)

    postprocess_started = time.perf_counter()
    features = model.inference_postprocess(features)
    features = np.asarray(features, dtype=np.float32)
    if features.size != 0:
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        features = features / norms
    timing_stats.add_reid_phase_time("postprocess", (time.perf_counter() - postprocess_started) * 1000)
    return features


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
            "preprocess",
            "inference",
            "postprocess",
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
    detector_has_legacy = any(normalized[key] > 0.0 for key in ("preprocess", "inference", "postprocess"))
    if detector_has_split:
        detector_preprocess_total = normalized["detector_preprocess"]
        detector_process_total = normalized["detector_process"]
        detector_postprocess_total = normalized["detector_postprocess"]
    elif detector_has_legacy:
        detector_preprocess_total = normalized["preprocess"]
        detector_process_total = normalized["inference"]
        detector_postprocess_total = normalized["postprocess"]
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
        detector_rows.append(_note("  detections loaded from cache"))
    else:
        detector_rows.extend([
            _row("  preprocess", float(breakdown["detector_preprocess_total"])),
            _row("  process", float(breakdown["detector_process_total"])),
            _row("  postprocess", float(breakdown["detector_postprocess_total"])),
        ])
    detector_rows.append(_row("  Detector total", float(breakdown["det_total"]), strong=True))

    tracker_rows: list[dict[str, object]] = [{"kind": "group", "label": "Tracker"}]
    if reid_cached:
        tracker_rows.append(_note("  embeddings loaded from cache"))
    else:
        tracker_rows.extend([
            _row("  ReID preprocess", float(breakdown["reid_preprocess_total"])),
            _row("  ReID process", float(breakdown["reid_process_total"])),
            _row("  ReID postprocess", float(breakdown["reid_postprocess_total"])),
        ])
    tracker_rows.extend([
        _row("  association/update", float(breakdown["tracker_rest_total"])),
        _row("  Tracker total", float(breakdown["tracker_total"]), strong=True),
    ])

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
    
    def reset(self):
        self.totals = {
            'preprocess': 0.0,
            'inference': 0.0,
            'postprocess': 0.0,
            'detector_preprocess': 0.0,
            'detector_process': 0.0,
            'detector_postprocess': 0.0,
            'reid': 0.0,
            'reid_preprocess': 0.0,
            'reid_process': 0.0,
            'reid_postprocess': 0.0,
            'track': 0.0,
            'plot': 0.0,
            'total': 0.0,
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
            self.totals['track'] += elapsed
            self._last_track_time = elapsed
            self._track_start = None
    
    def get_last_track_time(self):
        """Get the last tracking time in ms."""
        return getattr(self, '_last_track_time', 0)
    
    def get_last_reid_time(self):
        """Get the last ReID time in ms (accumulated during last track update)."""
        return getattr(self, '_last_reid_time', 0)
    
    def reset_frame_reid(self):
        """Reset per-frame ReID accumulator (call before each track update)."""
        self._last_reid_time = 0
    
    def start_plot(self):
        """Mark the start of plotting phase."""
        self._plot_start = time.perf_counter()
    
    def end_plot(self):
        """Mark the end of plotting phase and record time."""
        if self._plot_start is not None:
            self.totals['plot'] += (time.perf_counter() - self._plot_start) * 1000
            self._plot_start = None
    
    def add_reid_time(self, time_ms):
        """Add ReID time in milliseconds."""
        self.totals['reid'] += time_ms
        # Also accumulate for per-frame tracking
        self._last_reid_time = getattr(self, '_last_reid_time', 0) + time_ms

    def add_detector_phase_time(self, phase: str, time_ms: float) -> None:
        """Record detector phase timing while keeping legacy aggregate buckets updated."""
        phase_key = str(phase).strip().lower()
        specific_key = _DETECTOR_PHASE_KEYS[phase_key]
        legacy_key = _LEGACY_DETECTOR_KEYS[phase_key]
        elapsed_ms = float(time_ms or 0.0)
        self.totals[specific_key] += elapsed_ms
        self.totals[legacy_key] += elapsed_ms

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
        if hasattr(predictor, 'results') and predictor.results:
            for result in predictor.results:
                if hasattr(result, 'speed') and result.speed:
                    self.add_detector_phase_time('preprocess', result.speed.get('preprocess', 0) or 0)
                    self.add_detector_phase_time('process', result.speed.get('inference', 0) or 0)
                    self.add_detector_phase_time('postprocess', result.speed.get('postprocess', 0) or 0)
    
    def end_frame(self):
        """Mark the end of frame processing."""
        if self._frame_start is not None:
            self.totals['total'] += (time.perf_counter() - self._frame_start) * 1000
            self.frames += 1
            self._frame_start = None
    
    def format_summary(self) -> str:
        """Return a plain-text execution time summary table."""
        # Check if we have any data to display
        has_data = any(v > 0 for v in self.totals.values())
        if not has_data:
            return ""

        frames = self.frames if self.frames > 0 else 1  # Avoid division by zero

        breakdown = derive_timing_breakdown(self.totals, self.frames, total_time_ms=self.totals['total'])
        total_time = float(breakdown['total_total'])
        plot_time = float(breakdown['plot_total'])
        overhead = float(breakdown['overhead_total'])

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
                f"{'Plotting':<20} | {plot_time:<12.1f} | {plot_avg:<12.2f} | {plot_fps:<10.1f} | {pct(plot_time):<12.1f}"
            )

        if overhead > 0:
            overhead_avg = overhead / frames
            overhead_fps = fps_from_avg_ms(overhead_avg)
            lines.append(
                f"{'Other (I/O, etc)':<20} | {overhead:<12.1f} | {overhead_avg:<12.2f} | {overhead_fps:<10.1f} | {pct(overhead):<12.1f}"
            )

        lines.append(f"{'Frames':<20} | {frames:<12}")
        lines.append("=" * 105)
        return "\n".join(lines)

    def print_summary(self):
        """Print execution time summary table with blue color palette."""
        summary = self.format_summary()
        if not summary:
            return
        print_text(summary)

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize timing stats into a JSON-friendly summary dict."""
        totals = dict(self.totals)
        total_ms = float(totals.get("total", 0.0) or 0.0)
        if total_ms == 0.0:
            total_ms = float(sum(totals.values()))

        frames = int(self.frames)
        avg_ms = {
            key: (float(value) / frames if frames else 0.0)
            for key, value in totals.items()
        }
        avg_total_ms = total_ms / frames if frames else 0.0
        fps = (1000.0 * frames / total_ms) if total_ms else 0.0

        return {
            "frames": frames,
            "totals_ms": {**{key: float(value) for key, value in totals.items()}, "total": total_ms},
            "avg_ms": {**avg_ms, "total": avg_total_ms},
            "fps": fps,
            "metadata": dict(getattr(self, "metadata", {})),
        }


class TimedReIDWrapper:
    """Wrapper around ReID model to track timing."""
    
    def __init__(self, model, timing_stats):
        self._model = model
        self._timing_stats = timing_stats
    
    def get_features(self, *args, **kwargs):
        """Wrap get_features to measure timing."""
        return timed_reid_get_features(self._model, self._timing_stats, *args, **kwargs)
    
    def __getattr__(self, name):
        """Forward all other attributes to the wrapped model."""
        return getattr(self._model, name)


def wrap_tracker_reid(tracker, timing_stats):
    """
    Wrap a tracker's ReID model with timing instrumentation.
    
    Args:
        tracker: The tracker instance.
        timing_stats: TimingStats instance to record ReID timing.
    """
    # Different trackers store ReID model in different attributes
    reid_model = None
    reid_attr = None
    
    if hasattr(tracker, 'model') and tracker.model is not None:
        reid_model = tracker.model
        reid_attr = 'model'
    elif hasattr(tracker, 'reid_model') and tracker.reid_model is not None:
        reid_model = tracker.reid_model
        reid_attr = 'reid_model'
    
    if reid_model is not None and hasattr(reid_model, 'get_features'):
        wrapped = TimedReIDWrapper(reid_model, timing_stats)
        setattr(tracker, reid_attr, wrapped)
