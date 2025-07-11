# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
from functools import partial
from pathlib import Path

import cv2
import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, TRACKER_CONFIGS, WEIGHTS
from boxmot.utils.checks import RequirementsChecker
from boxmot.engine.detectors import default_imgsz, get_yolo_inferer, is_ultralytics_model
from ultralytics.utils.plotting import save_one_box


checker = RequirementsChecker()
checker.check_packages(("ultralytics", ))  # install

from boxmot.utils.patches import apply_patches
apply_patches()
from ultralytics import YOLO


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.
    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """
    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class,
        )
        # motion only models do not have
        if hasattr(tracker, "model"):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers

# callback to plot trajectories on each frame
def plot_trajectories(predictor):
    # predictor.results is a list of Results, one per frame in the batch
    for i, result in enumerate(predictor.results):
        tracker = predictor.trackers[i]
        crop_img = result.orig_img.copy()
        result.orig_img = tracker.plot_results(result.orig_img.copy(), predictor.custom_args.show_trajectories)

        # wait 1ms for a key, then check if it was 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # clean up and exit
            cv2.destroyAllWindows()
            sys.exit(0)
        
        if tracker.per_class_active_tracks is None:  # dict
            active_tracks = tracker.active_tracks
        else:
            active_tracks = []
            for k in tracker.per_class_active_tracks.keys():
                active_tracks += tracker.per_class_active_tracks[k]

        for a in active_tracks:
            if not a.history_observations: continue
            box = a.history_observations[-1][0:4]
            box = torch.as_tensor(box).unsqueeze(0)
            save_one_box(
                box,
                crop_img,
                file=Path(predictor.save_dir) / str(a.id) / Path(predictor.txt_path.stem).with_suffix(".jpg"),
                BGR=True,
                gain=1,
                pad=0
            )


@torch.no_grad()
def main(args):
    
    
    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)
    yolo = YOLO(
        args.yolo_model if is_ultralytics_model(args.yolo_model) else "yolov8n.pt",
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=True,
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

    yolo.add_callback("on_predict_start", partial(on_predict_start, persist=True))
    yolo.add_callback("on_predict_postprocess_end", plot_trajectories)

    if not is_ultralytics_model(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        yolo_model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args,
        )
        yolo.predictor.model = yolo_model

        # If current model is YOLOX, change the preprocess and postprocess
        if not is_ultralytics_model(args.yolo_model):
            # add callback to save image paths for further processing
            yolo.add_callback(
                "on_predict_batch_start", lambda p: yolo_model.update_im_paths(p)
            )
            yolo.predictor.preprocess = lambda imgs: yolo_model.preprocess(im=imgs)
            yolo.predictor.postprocess = lambda preds, im, im0s: yolo_model.postprocess(
                preds=preds, im=im, im0s=im0s
            )

    # store custom args in predictor
    yolo.predictor.custom_args = args

    for _ in results:
        pass


if __name__ == "__main__":
    main()