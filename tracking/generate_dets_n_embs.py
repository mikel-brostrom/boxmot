# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch

from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from tracking.detectors import get_yolo_inferer
from ultralytics.utils.files import increment_path

__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from tracking.detectors import create_detector

@torch.no_grad()
def run(args):
    save_dir = args.save_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    detector = create_detector(args)

    dets_path = save_dir / 'dets' / (Path(args.source).parent.name + '.txt')
    dets_path.parent.mkdir(parents=True, exist_ok=True)
    dets_path.touch(exist_ok=True)

    reid_model = None
    embs_path = None
    if str(args.reid_model) != '.':
        rab = ReidAutoBackend(
            weights=args.reid_model, device=device, half=args.half
        )
        reid_model = rab.get_backend()
        embs_path = save_dir / 'embs' / str(args.reid_model) / (Path(args.source).parent.name + '.txt')
        embs_path.parent.mkdir(parents=True, exist_ok=True)
        embs_path.touch(exist_ok=True)
    
    dets_results = []
    for frame_idx, img_name in enumerate(sorted(os.listdir(args.source))):
        img_path = os.path.join(args.source, img_name)
        img = cv2.imread(img_path)
        dets = detector.inference(img_path)

        for det in dets:
            x, y, w, h, conf, cls = det
            dets_results.append([frame_idx, -1, x, y, w, h, conf, cls, -1])

        dets = np.array(dets)[2:6]
        dets[:, 2] = dets[:, 0] + dets[:, 2]
        dets[:, 3] = dets[:, 1] + dets[:, 3]
        if reid_model:
            embs = reid_model.get_features(dets[:, 0:4], img)
            with open(str(embs_path), 'ab+') as f:  # append binary mode
                np.savetxt(f, embs, fmt='%f')  # save as ints instead of scientific notation
                
    with open(dets_path, 'a') as f:
        for det in dets_results:
            # x,y,w,h  => (top,left),(width,height)
            frame_idx, track_id, x, y, w, h, conf, cls, _ = det
            f.write(f'{frame_idx+1},{track_id},{x},{y},{w},{h},{conf},{cls},-1\n')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default='yolox_m',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default='',
                        help='reid model path')
    parser.add_argument('--source', type=str, default='/home/legkovas/Projects/tracking/yolo_tracking_save_det/yolo_tracking/assets/MOT17-mini/train/',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, default=0,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'dets_n_embs',
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
    args = parse_opt()
    mot_folder_paths = [item for item in Path(args.source).iterdir()]
    
    save_dir = Path(args.project) / args.yolo_model.stem
    if os.path.exists(save_dir):
        save_dir = increment_path(Path(args.project) / args.yolo_model.stem, exist_ok=False)
    args.save_dir = save_dir
    
    for mot_folder_path in mot_folder_paths:
        print(mot_folder_path)
        args.source = mot_folder_path / 'img1'
        run(args)
