
from pathlib import Path

from boxmot import TRACKERS
from boxmot.trackers.tracker_zoo import create_tracker
from boxmot.utils import TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker

# High-level API imports
from boxmot import track, ReID
from boxmot.engine.detectors import YOLOX, UltralyticsYolo
from boxmot.engine.detectors import is_yolox_model

checker = RequirementsChecker()
checker.check_packages(("ultralytics", ))  # install


def main(args):
    """
    Run tracking using the high-level boxmot.track API.
    
    Args:
        args: Arguments from CLI (SimpleNamespace from cli.py)
    """
    # 1. Setup Detector
    if is_yolox_model(args.yolo_model):
        detector = YOLOX(
            args.yolo_model, 
            device=args.device, 
            conf=args.conf, 
            iou=args.iou, 
            imgsz=args.imgsz if isinstance(args.imgsz, (int, float)) else args.imgsz[0]
        )
    else:
        # Assume Ultralytics
        detector = UltralyticsYolo(
            args.yolo_model,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz if isinstance(args.imgsz, (int, float)) else args.imgsz[0]
        )

    # 2. Setup ReID
    reid = ReID(args.reid_model, device=args.device, half=args.half)

    # 3. Setup Tracker
    tracking_config = TRACKER_CONFIGS / (args.tracking_method + '.yaml')
    tracker = create_tracker(
        args.tracking_method,
        tracking_config,
        args.reid_model,
        args.device,
        args.half,
        args.per_class
    )
    
    # 4. Run Track Loop
    results = track(args.source, detector, reid, tracker)
    
    # 5. Output/Show
    for frame_idx, tracks in enumerate(results):
        if args.show:
            tracks.show()