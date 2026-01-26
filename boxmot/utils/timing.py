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
        # Check if we have any data to display
        has_data = any(v > 0 for v in self.totals.values())
        if not has_data:
            return
        
        frames = self.frames if self.frames > 0 else 1  # Avoid division by zero
        
        # Calculate detection total
        det_total = self.totals['preprocess'] + self.totals['inference'] + self.totals['postprocess']
        total_time = self.totals['total']
        plot_time = self.totals['plot']
        reid_total = self.totals['reid']
        track_total = self.totals['track']
        
        # Determine workflow mode based on what was recorded
        # - Real-time tracking: tracking done frame-by-frame with ReID embedded (assoc = track - reid)
        # - Batch evaluation: ReID + tracking both recorded separately (assoc = track only since ReID is standalone)
        # 
        # In batch mode, ReID is done *before* tracking with pre-computed embeddings,
        # so track_total is pure association time. In real-time, ReID is inside track.
        # We can detect batch mode if frames==0 (timing was aggregated from subprocess)
        # or by looking for a flag. For now, use heuristic: if reid_total > 0 but frames==0, batch mode.
        
        is_batch_mode = self.frames == 0 or (reid_total > 0 and det_total > 0)
        
        # In batch mode: track_total is pure association (ReID was separate)
        # In real-time mode: association = track - reid
        if is_batch_mode:
            assoc_time = track_total  # Track is association-only when ReID is pre-computed
        else:
            assoc_time = max(0, track_total - reid_total)
        
        # Calculate overhead (unaccounted time) - only meaningful if total was recorded
        accounted = det_total + reid_total + track_total + plot_time
        
        # If no total time recorded, estimate from components
        if total_time == 0:
            total_time = accounted
        
        # For batch mode, track time doesn't overlap with det+reid
        overhead = max(0, total_time - accounted)
        
        # Helper to calculate percentage
        def pct(value):
            return (value / total_time * 100) if total_time > 0 else 0
        
        # Helper to calculate FPS from avg ms
        def fps_from_avg(avg_ms):
            return 1000 / avg_ms if avg_ms > 0 else 0
        
        # Helper for colored logging
        def log(msg):
            LOGGER.opt(colors=True).info(msg)
        
        log("")
        log("<blue>" + "=" * 105 + "</blue>")
        log(f"<bold><cyan>{'📊 TIMING SUMMARY':^105}</cyan></bold>")
        log("<blue>" + "=" * 105 + "</blue>")
        log(f"<bold>{'Component':<20}</bold> | {'Total (ms)':<12} | {'Avg (ms)':<12} | {'FPS':<10} | {'% of Total':<12}")
        log("<blue>" + "-" * 105 + "</blue>")
        
        # Detection pipeline
        for key in ['preprocess', 'inference', 'postprocess']:
            total = self.totals[key]
            avg = total / frames
            fps = fps_from_avg(avg)
            log(f"{key.capitalize():<20} | <blue>{total:<12.1f}</blue> | <blue>{avg:<12.2f}</blue> | <blue>{fps:<10.1f}</blue> | {pct(total):<12.1f}")
        
        det_avg = det_total / frames
        det_fps = fps_from_avg(det_avg)
        log(f"<bold>{'Detection (total)':<20}</bold> | <cyan>{det_total:<12.1f}</cyan> | <cyan>{det_avg:<12.2f}</cyan> | <cyan>{det_fps:<10.1f}</cyan> | {pct(det_total):<12.1f}")
        
        log("<blue>" + "-" * 105 + "</blue>")
        
        # ReID / Tracking section - display depends on workflow mode
        reid_avg = reid_total / frames if frames > 0 else 0
        reid_fps = fps_from_avg(reid_avg)
        log(f"{'ReID':<20} | <blue>{reid_total:<12.1f}</blue> | <blue>{reid_avg:<12.2f}</blue> | <blue>{reid_fps:<10.1f}</blue> | {pct(reid_total):<12.1f}")
        
        # Show association/track in both modes (since we now track association in batch mode too)
        if track_total > 0:
            assoc_avg = assoc_time / frames if frames > 0 else 0
            assoc_fps = fps_from_avg(assoc_avg)
            log(f"{'Association':<20} | <blue>{assoc_time:<12.1f}</blue> | <blue>{assoc_avg:<12.2f}</blue> | <blue>{assoc_fps:<10.1f}</blue> | {pct(assoc_time):<12.1f}")
            
            if not is_batch_mode:
                # In real-time mode, also show track total (which includes reid + assoc)
                track_avg = track_total / frames if frames > 0 else 0
                track_fps = fps_from_avg(track_avg)
                log(f"<bold>{'Track (total)':<20}</bold> | <cyan>{track_total:<12.1f}</cyan> | <cyan>{track_avg:<12.2f}</cyan> | <cyan>{track_fps:<10.1f}</cyan> | {pct(track_total):<12.1f}")
        
        log("<blue>" + "-" * 105 + "</blue>")
        
        # Plotting and overhead
        if plot_time > 0:
            plot_avg = plot_time / frames
            plot_fps = fps_from_avg(plot_avg)
            log(f"{'Plotting':<20} | <blue>{plot_time:<12.1f}</blue> | <blue>{plot_avg:<12.2f}</blue> | <blue>{plot_fps:<10.1f}</blue> | {pct(plot_time):<12.1f}")
        
        if overhead > 0:
            overhead_avg = overhead / frames
            overhead_fps = fps_from_avg(overhead_avg)
            log(f"{'Other (I/O, etc)':<20} | <blue>{overhead:<12.1f}</blue> | <blue>{overhead_avg:<12.2f}</blue> | <blue>{overhead_fps:<10.1f}</blue> | {pct(overhead):<12.1f}")
        
        log("<blue>" + "-" * 105 + "</blue>")
        avg_total = total_time / frames
        total_fps = fps_from_avg(avg_total)
        log(f"<bold>{'Total':<20}</bold> | <cyan>{total_time:<12.1f}</cyan> | <cyan>{avg_total:<12.2f}</cyan> | <cyan>{total_fps:<10.1f}</cyan> | {100.0:<12.1f}")
        log(f"<bold>{'Frames':<20}</bold> | <cyan>{frames:<12}</cyan>")
        
        log("<blue>" + "=" * 105 + "</blue>")
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
