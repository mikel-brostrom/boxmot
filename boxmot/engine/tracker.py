# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch

from boxmot import TRACKERS
from boxmot.detectors import (default_imgsz, get_yolo_inferer,
                              is_ultralytics_model)
from boxmot.trackers.tracker_zoo import create_tracker
from boxmot.utils import TRACKER_CONFIGS
from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.timing import TimingStats, wrap_tracker_reid

checker = RequirementsChecker()
checker.check_packages(("ultralytics", ))  # install

from ultralytics import YOLO


class VideoWriter:
    """Handles video writing for tracking results."""
    
    def __init__(self, output_path, fps=30):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.writer = None
        self.frame_size = None
    
    def write(self, frame):
        """Write a frame to the video."""
        if self.writer is None:
            h, w = frame.shape[:2]
            self.frame_size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                str(self.output_path), fourcc, self.fps, self.frame_size
            )
            LOGGER.opt(colors=True).info(f"<bold>Saving video to:</bold> <cyan>{self.output_path}</cyan>")
        
        self.writer.write(frame)
    
    def release(self):
        """Release the video writer."""
        if self.writer is not None:
            self.writer.release()
            LOGGER.opt(colors=True).info(f"<bold>Video saved:</bold> <cyan>{self.output_path}</cyan>")


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
    # Ensure at least 1 tracker is created (bs might be 0 for some sources)
    batch_size = max(1, predictor.dataset.bs)
    for i in range(batch_size):
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


def plot_trajectories(predictor, timing_stats=None, video_writer=None):
    """
    Callback to run tracking update and plot trajectories on each frame.
    
    Args:
        predictor (object): The predictor object containing results and trackers.
        timing_stats (TimingStats, optional): Timing statistics tracker.
        video_writer (VideoWriter, optional): Video writer for saving output.
    """
    # Ensure trackers are initialized
    if not hasattr(predictor, 'trackers') or not predictor.trackers:
        LOGGER.warning("Trackers not initialized, skipping frame")
        return
        
    # predictor.results is a list of Results, one per frame in the batch
    for i, result in enumerate(predictor.results):
        if i >= len(predictor.trackers):
            LOGGER.warning(f"No tracker for batch index {i}, skipping")
            continue
            
        tracker = predictor.trackers[i]
        
        # Get detections from result
        dets = result.boxes.data.cpu().numpy() if result.boxes is not None else np.empty((0, 6))
        img = result.orig_img
        
        # Reset per-frame ReID accumulator
        if timing_stats:
            timing_stats.reset_frame_reid()
        
        # Run tracking update (includes ReID)
        if timing_stats:
            timing_stats.start_tracking()
        
        tracks = tracker.update(dets, img)
        
        track_time = reid_time = assoc_time = 0
        if timing_stats:
            timing_stats.end_tracking()
            track_time = timing_stats.get_last_track_time()
            reid_time = timing_stats.get_last_reid_time()
            assoc_time = track_time - reid_time
        
        # Store tracks in result for downstream use
        result.tracks = tracks
        n_tracks = len(tracks) if tracks is not None and len(tracks) > 0 else 0
        
        # Log per-frame tracking info (detection time shown by ultralytics above)
        LOGGER.opt(colors=True).info(
            f"<bold>Track:</bold> <cyan>{n_tracks}</cyan> IDs, "
            f"reid: <blue>{reid_time:.1f}ms</blue>, "
            f"assoc: <blue>{assoc_time:.1f}ms</blue>, "
            f"total: <cyan>{track_time:.1f}ms</cyan>"
        )
        
        # Plot results
        if timing_stats:
            timing_stats.start_plot()
        
        result.orig_img = tracker.plot_results(
            img,
            predictor.custom_args.show_trajectories,
            show_lost=predictor.custom_args.show_lost
        )
        
        if timing_stats:
            timing_stats.end_plot()
        
        # Save frame to video
        if video_writer is not None:
            video_writer.write(result.orig_img)
        
        # Show the frame if requested
        if predictor.custom_args.show:
            cv2.imshow("BoxMOT", result.orig_img)
            key = cv2.waitKey(1) & 0xFF
            # Exit on 'q' key press - set flag for clean shutdown
            if key == ord('q'):
                predictor.custom_args._user_quit = True
    
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
    # Print tracking pipeline header (blue palette)
    LOGGER.info("")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    LOGGER.opt(colors=True).info("<bold><cyan>🎯 BoxMOT Tracking Pipeline</cyan></bold>")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    LOGGER.opt(colors=True).info(f"<bold>Detector:</bold>  <cyan>{args.yolo_model}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>ReID:</bold>      <cyan>{args.reid_model}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Tracker:</bold>   <cyan>{args.tracking_method}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Source:</bold>    <cyan>{args.source}</cyan>")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    
    # Set default image size based on model type
    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)
    
    # Initialize timing stats
    timing_stats = TimingStats()
    
    # Initialize video writer if saving is enabled
    video_writer = None
    if args.save:
        # Determine output path
        project = Path(args.project) if args.project else Path("runs/track")
        name = args.name if args.name else "exp"
        save_dir = project / name
        
        # Handle exist_ok
        if not args.exist_ok:
            i = 1
            while save_dir.exists():
                save_dir = project / f"{name}{i}"
                i += 1
        
        # Determine video filename from source
        source_path = Path(args.source)
        if source_path.is_file():
            video_name = source_path.stem + "_tracked.mp4"
        elif source_path.is_dir():
            video_name = source_path.name + "_tracked.mp4"
        else:
            video_name = "tracking_output.mp4"
        
        video_writer = VideoWriter(save_dir / video_name, fps=30)
    
    # Initialize YOLO model (use placeholder if non-ultralytics model)
    yolo = YOLO(
        args.yolo_model if is_ultralytics_model(args.yolo_model) else "yolov8n.pt",
    )

    # Add callbacks for tracker initialization and trajectory plotting
    # Pass args, timing_stats and video_writer through partial to make them available in callbacks
    yolo.add_callback("on_predict_start", partial(on_predict_start, args=args, timing_stats=timing_stats))
    yolo.add_callback("on_predict_postprocess_end", partial(plot_trajectories, timing_stats=timing_stats, video_writer=video_writer))
    
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
        show=False,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=False,  # We handle video saving ourselves with tracking overlays
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

    # Initialize quit flag
    args._user_quit = False
    
    # Iterate through results to run the tracking pipeline
    try:
        for result in results:
            # Check if user requested quit
            if args._user_quit:
                break
                
            # Record Ultralytics timing from result.speed (populated after yield)
            if hasattr(result, 'speed') and result.speed:
                timing_stats.totals['preprocess'] += result.speed.get('preprocess', 0) or 0
                timing_stats.totals['inference'] += result.speed.get('inference', 0) or 0
                timing_stats.totals['postprocess'] += result.speed.get('postprocess', 0) or 0
    except KeyboardInterrupt:
        pass  # Handle Ctrl+C gracefully
    finally:
        # Release video writer
        if video_writer is not None:
            video_writer.release()
        # Always print timing summary when done
        timing_stats.print_summary()
        # Clean up windows
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit("Run via CLI: boxmot track [options]")
