# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
from pathlib import Path
import numpy as np
from functools import partial
import json
import torch

from tqdm import tqdm

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker

from ultralytics.utils.files import increment_path 
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import TestRequirements
from boxmot.utils import logger as LOGGER

from ultralytics.data.loaders import LoadImages
from ultralytics import YOLO
from ultralytics.data.utils import VID_FORMATS

from tracking.utils import convert_to_mot_format, write_mot_results


__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install


def generate_mot_results(args):

    tracker = create_tracker(
        args.tracking_method,
        TRACKER_CONFIGS / (args.tracking_method + '.yaml'),
        args.reid_model.with_suffix('.pt'),
        'cpu',
        False,
        False
    )

    with open(args.dets_file_path, 'r') as file:
        args.source = file.readline().strip().replace("# ", "")  # .strip() removes leading/trailing whitespace and newline characters

    LOGGER.info(f"\nStarting tracking on:\n\t{args.source}\nwith preloaded dets\n\t({args.dets_file_path.relative_to(ROOT)})\nand embs\n\t({args.embs_file_path.relative_to(ROOT)})\nusing\n\t{args.tracking_method}")

    dets = np.loadtxt(args.dets_file_path, skiprows=1)  # skiprows=1 skips the header row
    embs = np.loadtxt(args.embs_file_path)  # skiprows=1 skips the header row

    dets_n_embs = np.concatenate(
        [
            dets,
            embs
        ], axis=1
    )

    dataset = LoadImages(args.source)
    
    txt_path = args.exp_folder_path / (Path(args.source).parent.name + '.txt')
    for frame_idx, d in enumerate(tqdm(dataset, desc="Frames")):

        # don't generate dets_n_emb for the last frame
        if (frame_idx + 1) == len(dataset):
            break

        im = d[1][0]

        # get dets and embedding associated to this frame
        frame_dets_n_embs = dets_n_embs[dets_n_embs[:, 0] == frame_idx + 1]

        # frame id, x1, y1, x2, y2, conf, cls
        dets = frame_dets_n_embs[:, 1:7]
        embs = frame_dets_n_embs[:, 7:]
        tracks = tracker.update(dets, im, embs)

        mot_results = convert_to_mot_format(tracks, frame_idx + 1)
        write_mot_results(txt_path, mot_results)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default='yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default='osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--project', default=ROOT / 'runs' / 'mot',
                        help='save results to project/name')
    parser.add_argument('--name', default='mot',
                        help='save results to project/name')
    parser.add_argument('--dets', type=str, default='yolov8n',
                        help='the folder name under project to load the detections from')
    parser.add_argument('--embs', type=str, default='osnet_x0_25_msmt17',
                        help='the folder name under project/dets to load the embeddings from')
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
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--benchmark', type=str, default='MOT17',
                        help='MOT16, MOT17, MOT20')

    opt = parser.parse_args()
    return opt

def run_generate_mot_results(opt):
    if opt is None:
        opt = parse_opt()  
    else:
        opt = opt

    exp_folder_path = opt.project / (str(opt.dets) + "_" + str(opt.embs) + "_" + str(opt.tracking_method))
    exp_folder_path = increment_path(path=exp_folder_path, sep="_", exist_ok=False)
    opt.exp_folder_path = exp_folder_path
    dets_file_paths = [item for item in (opt.project.parent / "dets_n_embs" / opt.dets / 'dets').glob('*.txt')]
    embs_file_paths = [item for item in (opt.project.parent / "dets_n_embs" / opt.dets / 'embs' /  opt.embs).glob('*.txt')]
    print(dets_file_paths)
    print(embs_file_paths)
    for d, e in zip(dets_file_paths, embs_file_paths):
        opt.dets_file_path = d
        opt.embs_file_path = e
        generate_mot_results(opt)


if __name__ == "__main__":
    run_generate_mot_results(None)
