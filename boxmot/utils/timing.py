# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import time

from boxmot.utils import logger as LOGGER
from boxmot.utils.rich.ui import print_text


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
        for key in ("preprocess", "inference", "postprocess", "reid", "track", "plot", "total")
    }

    det_total = normalized["preprocess"] + normalized["inference"] + normalized["postprocess"]
    reid_total = normalized["reid"]
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
        "det_total": det_total,
        "reid_total": reid_total,
        "track_total": track_total,
        "tracker_rest_total": tracker_rest_total,
        "tracker_total": tracker_total,
        "plot_total": plot_total,
        "total_total": total_total,
        "overhead_total": overhead_total,
    }


class TimingStats:
    """Track timing statistics for detection, ReID, and tracking phases."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.totals = {
            'preprocess': 0.0,
            'inference': 0.0,
            'postprocess': 0.0,
            'reid': 0.0,
            'track': 0.0,
            'plot': 0.0,
            'total': 0.0,
        }
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
    
    def record_ultralytics_times(self, predictor):
        """Record timing from Ultralytics results."""
        # Ultralytics stores speed info in results[].speed dict
        # speed contains: preprocess, inference, postprocess times in ms
        if hasattr(predictor, 'results') and predictor.results:
            for result in predictor.results:
                if hasattr(result, 'speed') and result.speed:
                    self.totals['preprocess'] += result.speed.get('preprocess', 0) or 0
                    self.totals['inference'] += result.speed.get('inference', 0) or 0
                    self.totals['postprocess'] += result.speed.get('postprocess', 0) or 0
    
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
        det_total = float(breakdown['det_total'])
        total_time = float(breakdown['total_total'])
        plot_time = float(breakdown['plot_total'])
        reid_total = float(breakdown['reid_total'])
        tracker_rest_total = float(breakdown['tracker_rest_total'])
        tracker_total = float(breakdown['tracker_total'])
        overhead = float(breakdown['overhead_total'])

        # Helper to calculate percentage
        def pct(value):
            return (value / total_time * 100) if total_time > 0 else 0

        lines = [
            "=" * 105,
            f"{'📊 TIMING SUMMARY':^105}",
            "=" * 105,
            f"{'Component':<20} | {'Total (ms)':<12} | {'Avg (ms)':<12} | {'FPS':<10} | {'% of Total':<12}",
            "-" * 105,
        ]

        # Detection pipeline
        for key in ['preprocess', 'inference', 'postprocess']:
            total = self.totals[key]
            avg = total / frames
            fps = fps_from_avg_ms(avg)
            lines.append(
                f"{key.capitalize():<20} | {total:<12.1f} | {avg:<12.2f} | {fps:<10.1f} | {pct(total):<12.1f}"
            )

        det_avg = det_total / frames
        det_fps = fps_from_avg_ms(det_avg)
        lines.append(
            f"{'Detection (total)':<20} | {det_total:<12.1f} | {det_avg:<12.2f} | {det_fps:<10.1f} | {pct(det_total):<12.1f}"
        )

        lines.append("-" * 105)

        # ReID / Tracking section - display depends on workflow mode
        reid_avg = reid_total / frames if frames > 0 else 0
        reid_fps = fps_from_avg_ms(reid_avg)
        lines.append(
            f"{'ReID':<20} | {reid_total:<12.1f} | {reid_avg:<12.2f} | {reid_fps:<10.1f} | {pct(reid_total):<12.1f}"
        )

        if tracker_rest_total > 0 or tracker_total > 0:
            tracker_rest_avg = tracker_rest_total / frames if frames > 0 else 0
            tracker_rest_fps = fps_from_avg_ms(tracker_rest_avg)
            lines.append(
                f"{'Tracker rest':<20} | {tracker_rest_total:<12.1f} | {tracker_rest_avg:<12.2f} | {tracker_rest_fps:<10.1f} | {pct(tracker_rest_total):<12.1f}"
            )

            tracker_total_avg = tracker_total / frames if frames > 0 else 0
            tracker_total_fps = fps_from_avg_ms(tracker_total_avg)
            lines.append(
                f"{'Tracker total':<20} | {tracker_total:<12.1f} | "
                f"{tracker_total_avg:<12.2f} | {tracker_total_fps:<10.1f} | {pct(tracker_total):<12.1f}"
            )

        lines.append("-" * 105)

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

        lines.append("-" * 105)
        avg_total = total_time / frames
        total_fps = fps_from_avg_ms(avg_total)
        lines.append(f"{'Total':<20} | {total_time:<12.1f} | {avg_total:<12.2f} | {total_fps:<10.1f} | {100.0:<12.1f}")
        lines.append(f"{'Frames':<20} | {frames:<12}")
        lines.append("=" * 105)
        return "\n".join(lines)

    def print_summary(self):
        """Print execution time summary table with blue color palette."""
        summary = self.format_summary()
        if not summary:
            return
        print_text(summary)


class TimedReIDWrapper:
    """Wrapper around ReID model to track timing."""
    
    def __init__(self, model, timing_stats):
        self._model = model
        self._timing_stats = timing_stats
    
    def get_features(self, *args, **kwargs):
        """Wrap get_features to measure timing."""
        t0 = time.perf_counter()
        result = self._model.get_features(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._timing_stats.add_reid_time(elapsed_ms)
        return result
    
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
