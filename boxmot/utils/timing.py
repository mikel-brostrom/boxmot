# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import time


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
            'total': 0.0,
        }
        self.frames = 0
        self._frame_start = None
        self._track_start = None
    
    def start_frame(self):
        """Mark the start of frame processing."""
        self._frame_start = time.perf_counter()
    
    def start_tracking(self):
        """Mark the start of tracking phase."""
        self._track_start = time.perf_counter()
    
    def end_tracking(self):
        """Mark the end of tracking phase and record time."""
        if self._track_start is not None:
            self.totals['track'] += (time.perf_counter() - self._track_start) * 1000
            self._track_start = None
    
    def add_reid_time(self, time_ms):
        """Add ReID time in milliseconds."""
        self.totals['reid'] += time_ms
    
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
        """Print execution time summary table."""
        if self.frames == 0:
            return
        
        frames = self.frames
        
        # Calculate detection total and association time
        det_total = self.totals['preprocess'] + self.totals['inference'] + self.totals['postprocess']
        assoc_time = self.totals['track'] - self.totals['reid']
        
        print("\n" + "=" * 75)
        print(f"{'TIMING SUMMARY':^75}")
        print("=" * 75)
        print(f"{'Component':<20} | {'Total Time (ms)':<20} | {'Avg per Frame (ms)':<20}")
        print("-" * 75)
        
        # Detection pipeline
        for key in ['preprocess', 'inference', 'postprocess']:
            total = self.totals[key]
            avg = total / frames
            print(f"{key.capitalize():<20} | {total:<20.1f} | {avg:<20.2f}")
        
        det_avg = det_total / frames
        print(f"{'Detection (total)':<20} | {det_total:<20.1f} | {det_avg:<20.2f}")
        
        print("-" * 75)
        
        # Tracking pipeline (split into ReID + Association)
        reid_total = self.totals['reid']
        reid_avg = reid_total / frames
        print(f"{'ReID':<20} | {reid_total:<20.1f} | {reid_avg:<20.2f}")
        
        assoc_avg = assoc_time / frames
        print(f"{'Association':<20} | {assoc_time:<20.1f} | {assoc_avg:<20.2f}")
        
        track_total = self.totals['track']
        track_avg = track_total / frames
        print(f"{'Track (total)':<20} | {track_total:<20.1f} | {track_avg:<20.2f}")
        
        print("-" * 75)
        total_time = self.totals['total']
        avg_total = total_time / frames
        fps = 1000 / avg_total if avg_total > 0 else 0
        print(f"{'Total':<20} | {total_time:<20.1f} | {avg_total:<20.2f}")
        print(f"{'Frames':<20} | {frames:<20}")
        print(f"{'Average FPS':<20} | {fps:<20.1f}")
        print("=" * 75 + "\n")


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
