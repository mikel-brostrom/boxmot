# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
from pathlib import Path
import numpy as np

import torch

from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
from examples.detectors import get_yolo_inferer
from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend

__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.data.utils import VID_FORMATS


@torch.no_grad()
def run(args):

    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    results = yolo(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=args.show,
        stream=True,
        device=args.device,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
    )

    reid = ReIDDetectMultiBackend(weights=args.reid_model, device=yolo.predictor.device, fp16=args.half)


    if 'yolov8' not in str(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    # store custom args in predictor
    yolo.predictor.custom_args = args
    dets_n_embs = []

    p = yolo.predictor.save_dir / 'det_n_embs' / (Path(args.source).parent.name + '.txt')
    yolo.predictor.det_n_embs_txt_path = p

    # create parent folder
    yolo.predictor.det_n_embs_txt_path.parent.mkdir(parents=True, exist_ok=True)
    # create mot txt file
    yolo.predictor.det_n_embs_txt_path.touch(exist_ok=True)

    with open(str(yolo.predictor.det_n_embs_txt_path), 'ab+') as f:  # append binary mode
        np.savetxt(f, [], fmt='%f', header=str(args.source))  # save as ints instead of scientific notation

    for frame_idx, r in enumerate(results):

        nr_dets = len(r.boxes)
        frame_idx = torch.full((1, 1), frame_idx + 1)
        frame_idx = frame_idx.repeat(nr_dets, 1)

        dets = r.boxes.data[:, 0:4].numpy()
        img = r.orig_img
        embs = reid.get_features(dets, img)

        dets_n_embs = np.concatenate(
            [
                frame_idx,
                r.boxes.xyxy.to('cpu'),
                r.boxes.conf.unsqueeze(1).to('cpu'),
                r.boxes.cls.unsqueeze(1).to('cpu'),
                embs
            ], axis=1
        )

        with open(str(yolo.predictor.det_n_embs_txt_path), 'ab+') as f:  # append binary mode
            np.savetxt(f, dets_n_embs, fmt='%f')  # save as ints instead of scientific notation


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--mot-seq-folder', type=Path, default='/home/mikel.brostrom/yolo_tracking/assets/MOT17-mini/train',
                        help='fodler to MOT dataset')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, default=0,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', default=True,
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    mot_folder_paths = [item for item in opt.mot_seq_folder.iterdir()]
    
    for mot_folder_path in mot_folder_paths:
        opt.source = mot_folder_path / 'img1'
        run(opt)