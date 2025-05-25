#!/usr/bin/env python3

import argparse
from pathlib import Path
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS, logger as LOGGER, EXAMPLES


def main():
    eval_parent = argparse.ArgumentParser(add_help=False)
    eval_parent.add_argument('--yolo-model', nargs='+', type=Path,
                            default=[WEIGHTS/'yolov8n.pt'], help='…')
    eval_parent.add_argument('--reid-model', nargs='+', type=Path,
                            default=[WEIGHTS/'osnet_x0_25_msmt17.pt'], help='…')

    parser = argparse.ArgumentParser(prog='boxmot_cli', conflict_handler='resolve',)
    # common args
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='Path to YOLO weights')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='Path to re-identification model')
    parser.add_argument('--source', type=str, default="./tracking/val_utils/data/MOT17-50/train", help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img-size', nargs='+', type=int, default=None, help='inference size h,w')
    parser.add_argument('--fps', type=int, default=None, help='video frame-rate')
    parser.add_argument('--conf', type=float, default=0.01, help='min confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, default=0, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs', type=Path, help='save results to project/name')
    parser.add_argument('--name', default='', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', default=True, help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--ci', action='store_true', help='Automatically reuse existing due to no UI in CI')
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack, boosttrack')
    parser.add_argument('--dets-file-path', type=Path, help='path to detections file')
    parser.add_argument('--embs-file-path', type=Path, help='path to embeddings file')
    parser.add_argument('--exp-folder-path', type=Path, help='path to experiment folder')
    parser.add_argument('--verbose', action='store_true', help='print results')
    parser.add_argument('--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--gsi', action='store_true', help='apply Gaussian smooth interpolation postprocessing')
    parser.add_argument('--n-trials', type=int, default=4, help='nr of trials for evolution')
    parser.add_argument('--objectives', type=str, nargs='+', default=["HOTA", "MOTA", "IDF1"], help='set of objective metrics: HOTA,MOTA,IDF1')
    parser.add_argument('--val-tools-path', type=Path, default=EXAMPLES / 'val_utils', help='path to store trackeval repo in')
    parser.add_argument('--split-dataset', action='store_true', help='Use the second half of the dataset')
    parser.add_argument('--show', action='store_true', help='Display tracking in a window')
    parser.add_argument('--show-labels', action='store_false',
                        help='Hide detection labels')
    parser.add_argument('--show-conf', action='store_false',
                        help='Hide detection confidences')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='Overlay object trajectories')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save tracking output to text')
    parser.add_argument('--save-crop', action='store_true',
                        help='Save cropped detections')
    parser.add_argument('--save', action='store_true',
                        help='Save tracking results')
    parser.add_argument('--line-width', type=int, default=None,
                        help='Bounding box line width')
    parser.add_argument('--per-class', action='store_true',
                        help='Track each class separately')

        # sub-commands
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('track')
    sub.add_parser('generate-dets-embs')
    sub.add_parser('generate-mot-results')
    sub.add_parser('eval', parents=[eval_parent], conflict_handler='resolve')
    sub.add_parser('tune', parents=[eval_parent], conflict_handler='resolve')
    sub.add_parser('all')

    # parse and dispatch
    args = parser.parse_args()
    source_path = Path(args.source)
    args.benchmark, args.split = source_path.parent.name, source_path.name

    if args.command == 'track':
        from tracking.track import main as run_track
        run_track(args)
    elif args.command == 'generate-dets-embs':
        from tracking.val import run_generate_dets_embs
        run_generate_dets_embs(args)
    elif args.command == 'generate-mot-results':
        from tracking.val import run_generate_mot_results
        run_generate_mot_results(args)
    elif args.command in ('eval', 'all'):
        from tracking.val import main as run_eval
        run_eval(args)
    elif args.command == 'tune':
        from tracking.evolve import main as run_tuning
        run_tuning(args)


if __name__ == "__main__":
    main()