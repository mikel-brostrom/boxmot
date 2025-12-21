# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import time

from boxmot.utils import logger as LOGGER


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
    
    def print_summary(self):
        """Print execution time summary table with blue color palette."""
        if self.frames == 0:
            return
        
        frames = self.frames
        
        # Calculate detection total and association time
        det_total = self.totals['preprocess'] + self.totals['inference'] + self.totals['postprocess']
        assoc_time = self.totals['track'] - self.totals['reid']
        total_time = self.totals['total']
        plot_time = self.totals['plot']
        
        # Calculate overhead (unaccounted time)
        accounted = det_total + self.totals['track'] + plot_time
        overhead = max(0, total_time - accounted)
        
        # Helper to calculate percentage
        def pct(value):
            return (value / total_time * 100) if total_time > 0 else 0
        
        # Helper for colored logging
        def log(msg):
            LOGGER.opt(colors=True).info(msg)
        
        log("")
        log("<blue>" + "=" * 90 + "</blue>")
        log(f"<bold><cyan>{'📊 TIMING SUMMARY':^90}</cyan></bold>")
        log("<blue>" + "=" * 90 + "</blue>")
        log(f"<bold>{'Component':<20}</bold> | {'Total (ms)':<12} | {'Avg (ms)':<12} | {'% of Total':<12}")
        log("<blue>" + "-" * 90 + "</blue>")
        
        # Detection pipeline
        for key in ['preprocess', 'inference', 'postprocess']:
            total = self.totals[key]
            avg = total / frames
            log(f"{key.capitalize():<20} | <blue>{total:<12.1f}</blue> | <blue>{avg:<12.2f}</blue> | {pct(total):<12.1f}")
        
        det_avg = det_total / frames
        log(f"<bold>{'Detection (total)':<20}</bold> | <cyan>{det_total:<12.1f}</cyan> | <cyan>{det_avg:<12.2f}</cyan> | {pct(det_total):<12.1f}")
        
        log("<blue>" + "-" * 90 + "</blue>")
        
        # Tracking pipeline (split into ReID + Association)
        reid_total = self.totals['reid']
        reid_avg = reid_total / frames
        log(f"{'ReID':<20} | <blue>{reid_total:<12.1f}</blue> | <blue>{reid_avg:<12.2f}</blue> | {pct(reid_total):<12.1f}")
        
        assoc_avg = assoc_time / frames
        log(f"{'Association':<20} | <blue>{assoc_time:<12.1f}</blue> | <blue>{assoc_avg:<12.2f}</blue> | {pct(assoc_time):<12.1f}")
        
        track_total = self.totals['track']
        track_avg = track_total / frames
        log(f"<bold>{'Track (total)':<20}</bold> | <cyan>{track_total:<12.1f}</cyan> | <cyan>{track_avg:<12.2f}</cyan> | {pct(track_total):<12.1f}")
        
        log("<blue>" + "-" * 90 + "</blue>")
        
        # Plotting and overhead
        plot_avg = plot_time / frames
        log(f"{'Plotting':<20} | <blue>{plot_time:<12.1f}</blue> | <blue>{plot_avg:<12.2f}</blue> | {pct(plot_time):<12.1f}")
        
        overhead_avg = overhead / frames
        log(f"{'Other (I/O, etc)':<20} | <blue>{overhead:<12.1f}</blue> | <blue>{overhead_avg:<12.2f}</blue> | {pct(overhead):<12.1f}")
        
        # Sanity check: verify components sum to total
        components_sum = det_total + self.totals['track'] + plot_time + overhead
        sum_pct = pct(det_total) + pct(self.totals['track']) + pct(plot_time) + pct(overhead)
        
        log("<blue>" + "-" * 90 + "</blue>")
        avg_total = total_time / frames
        fps = 1000 / avg_total if avg_total > 0 else 0
        log(f"<bold>{'Total':<20}</bold> | <cyan>{total_time:<12.1f}</cyan> | <cyan>{avg_total:<12.2f}</cyan> | {sum_pct:<12.1f}")
        log(f"<bold>{'Frames':<20}</bold> | <cyan>{frames:<12}</cyan>")
        log(f"<bold>{'Average FPS':<20}</bold> | <cyan>{fps:<12.1f}</cyan>")
        
        # Warn if there's a significant discrepancy
        if abs(components_sum - total_time) > 1.0:  # More than 1ms difference
            LOGGER.warning(f"Components sum ({components_sum:.1f}ms) != Total ({total_time:.1f}ms)")
        
        log("<blue>" + "=" * 90 + "</blue>")
        log("")


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
