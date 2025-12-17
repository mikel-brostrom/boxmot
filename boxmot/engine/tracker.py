# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from functools import partial
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from boxmot import TRACKERS
from boxmot.trackers.tracker_zoo import create_tracker
from boxmot.utils import ROOT, TRACKER_CONFIGS, WEIGHTS
from boxmot.utils.checks import RequirementsChecker
from boxmot.detectors import default_imgsz, get_yolo_inferer, is_ultralytics_model

checker = RequirementsChecker()
checker.check_packages(("ultralytics", ))  # install

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


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


def on_predict_start(predictor, args, timing_stats=None):
    """
    Initialize trackers for object tracking during prediction.
    
    Args:
        predictor (object): The predictor object to initialize trackers for.
        args: CLI arguments containing tracking configuration.
        timing_stats: Optional TimingStats for ReID timing instrumentation.
    """
    assert args.tracking_method in TRACKERS, \
        f"'{args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            args.tracking_method,
            tracking_config,
            args.reid_model,
            predictor.device,
            args.half,
            args.per_class,
        )
        # set target_id if user passed it
        if args.target_id is not None:
            tracker.target_id = args.target_id
        
        # Wrap ReID model for timing instrumentation
        if timing_stats is not None:
            wrap_tracker_reid(tracker, timing_stats)
        
        trackers.append(tracker)

    predictor.trackers = trackers
    predictor.custom_args = args  # Store for later use


def plot_trajectories(predictor, timing_stats=None):
    """
    Callback to run tracking update and plot trajectories on each frame.
    
    Args:
        predictor (object): The predictor object containing results and trackers.
        timing_stats (TimingStats, optional): Timing statistics tracker.
    """
    # predictor.results is a list of Results, one per frame in the batch
    for i, result in enumerate(predictor.results):
        tracker = predictor.trackers[i]
        
        # Get detections from result
        dets = result.boxes.data.cpu().numpy() if result.boxes is not None else np.empty((0, 6))
        img = result.orig_img
        
        # Run tracking update (includes ReID)
        if timing_stats:
            timing_stats.start_tracking()
        
        tracks = tracker.update(dets, img)
        
        if timing_stats:
            timing_stats.end_tracking()
        
        # Plot results
        result.orig_img = tracker.plot_results(img, predictor.custom_args.show_trajectories)
        cv2.waitKey(1)
    
    # End frame timing
    if timing_stats:
        timing_stats.end_frame()


def setup_yolox_model(predictor, args, yolo_model_instance):
    """
    Setup YOLOX model by replacing the predictor's model with our custom inferer.
    Called via on_predict_start callback.
    
    Args:
        predictor: The Ultralytics predictor object.
        args: CLI arguments.
        yolo_model_instance: The YoloXStrategy instance to use.
    """
    # Replace the YOLO model with our custom inferer
    predictor.model = yolo_model_instance

    # Override preprocess and postprocess for non-ultralytics models
    predictor.preprocess = lambda imgs: yolo_model_instance.preprocess(im=imgs)
    predictor.postprocess = lambda preds, im, im0s: yolo_model_instance.postprocess(
        preds=preds, im=im, im0s=im0s
    )


@torch.no_grad()
def main(args):
    """
    Run tracking using the integrated Ultralytics workflow.
    
    Args:
        args: Arguments from CLI (SimpleNamespace from cli.py)
    """
    # Set default image size based on model type
    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)
    
    # Initialize timing stats
    timing_stats = TimingStats()
    
    # Initialize YOLO model (use placeholder if non-ultralytics model)
    yolo = YOLO(
        args.yolo_model if is_ultralytics_model(args.yolo_model) else "yolov8n.pt",
    )

    # Add callbacks for tracker initialization and trajectory plotting
    # Pass args and timing_stats through partial to make them available in callbacks
    yolo.add_callback("on_predict_start", partial(on_predict_start, args=args, timing_stats=timing_stats))
    yolo.add_callback("on_predict_postprocess_end", partial(plot_trajectories, timing_stats=timing_stats))
    
    # Add callback to start frame timing
    yolo.add_callback("on_predict_batch_start", lambda p: timing_stats.start_frame())

    # Handle non-ultralytics models (e.g., YOLOX)
    # We need to setup the model replacement via callback since predictor
    # doesn't exist until predict() is called
    yolox_model = None
    if not is_ultralytics_model(args.yolo_model):
        # Create the YOLOX model inferer - will be setup in callback
        m = get_yolo_inferer(args.yolo_model)
        
        # Define a callback that will setup YOLOX when predictor is ready
        def setup_yolox_callback(predictor):
            nonlocal yolox_model
            yolox_model = m(
                model=args.yolo_model,
                device=predictor.device,
                args=predictor.args,
            )
            setup_yolox_model(predictor, args, yolox_model)
        
        # Add the setup callback - it will run on_predict_start
        yolo.add_callback("on_predict_start", setup_yolox_callback)
        
        # Add callback to save image paths for further processing
        def update_paths_callback(predictor):
            if yolox_model is not None:
                yolox_model.update_im_paths(predictor)
        yolo.add_callback("on_predict_batch_start", update_paths_callback)

    # Use predict() instead of track() to avoid Ultralytics' default tracking callbacks
    results = yolo.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=args.show,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width,
        save_crop=args.save_crop,
    )

    # Iterate through results to run the tracking pipeline
    try:
        for result in results:
            # Record Ultralytics timing from result.speed (populated after yield)
            if hasattr(result, 'speed') and result.speed:
                timing_stats.totals['preprocess'] += result.speed.get('preprocess', 0) or 0
                timing_stats.totals['inference'] += result.speed.get('inference', 0) or 0
                timing_stats.totals['postprocess'] += result.speed.get('postprocess', 0) or 0
    finally:
        # Print timing summary when done
        if args.verbose:
            timing_stats.print_summary()


if __name__ == "__main__":
    main()
