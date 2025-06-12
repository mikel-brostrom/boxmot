#!/usr/bin/env python3

import argparse
from pathlib import Path
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS, logger as LOGGER, EXAMPLES


def main():
    # Parent parser for evaluation commands: allow multiple weights
    eval_parent = argparse.ArgumentParser(add_help=False)
    eval_parent.add_argument(
        '--yolo-model', nargs='+', type=Path,
        default=[WEIGHTS / 'yolov8n.pt'],
        help='one or more YOLO weights for detection (only for generate/eval/tune)'
    )
    eval_parent.add_argument(
        '--reid-model', nargs='+', type=Path,
        default=[WEIGHTS / 'osnet_x0_25_msmt17.pt'],
        help='one or more ReID model weights (only for generate/eval/tune)'
    )
    eval_parent.add_argument('--classes', nargs='+', type=int,
        default=[0], help='filter by class indices')

    # Common arguments for all commands (flags only, no positionals)
    common_parser = argparse.ArgumentParser(add_help=False, conflict_handler='resolve')
    common_parser.add_argument(
        '--yolo-model', type=Path, default=WEIGHTS / 'yolov8n.pt',
        help='path to YOLO weights for detection'
    )
    common_parser.add_argument(
        '--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
        help='path to ReID model weights'
    )
    common_parser.add_argument(
        '--source', type=str, default='0',
        help='file/dir/URL/glob, 0 for webcam'
    )
    common_parser.add_argument('--imgsz', '--img-size', nargs='+', type=int,
                               default=None, help='inference size h,w')
    common_parser.add_argument('--fps', type=int, default=30,
                               help='video frame-rate')
    common_parser.add_argument('--conf', type=float, default=0.01,
                               help='min confidence threshold')
    common_parser.add_argument('--iou', type=float, default=0.7,
                               help='IoU threshold for NMS')
    common_parser.add_argument('--device', default='', help='cuda device(s), e.g. 0 or 0,1,2,3 or cpu')
    common_parser.add_argument('--classes', nargs='+', type=int,
                               help='filter by class indices')
    common_parser.add_argument('--project', type=Path, default=ROOT / 'runs',
                               help='save results to project/name')
    common_parser.add_argument('--name', default='', help='save results to project/name')
    common_parser.add_argument('--exist-ok', action='store_true', default=True,
                               help='existing project/name ok, do not increment')
    common_parser.add_argument('--half', action='store_true',
                               help='use FP16 half-precision inference')
    common_parser.add_argument('--vid-stride', type=int, default=1,
                               help='video frame-rate stride')
    common_parser.add_argument('--ci', action='store_true',
                               help='reuse existing runs in CI (no UI)')
    common_parser.add_argument('--tracking-method', type=str,
                               default='deepocsort',
                               help='deepocsort, botsort, strongsort, ...')
    common_parser.add_argument('--dets-file-path', type=Path,
                               help='path to precomputed detections file')
    common_parser.add_argument('--embs-file-path', type=Path,
                               help='path to precomputed embeddings file')
    common_parser.add_argument('--exp-folder-path', type=Path,
                               help='path to experiment folder')
    common_parser.add_argument('--verbose', action='store_false',
                               help='print detailed logs')
    common_parser.add_argument('--agnostic-nms', action='store_true',
                               help='class-agnostic NMS')
    common_parser.add_argument('--gsi', action='store_true',
                               help='apply Gaussian smoothing interpolation')
    common_parser.add_argument('--n-trials', type=int, default=4,
                               help='number of trials for evolutionary tuning')
    common_parser.add_argument('--objectives', nargs='+', type=str,
                               default=["HOTA", "MOTA", "IDF1"],
                               help='objectives for tuning: HOTA, MOTA, IDF1')
    common_parser.add_argument('--val-tools-path', type=Path,
                               default=EXAMPLES / 'val_utils',
                               help='where to clone trackeval')
    common_parser.add_argument('--split-dataset', action='store_true',
                               help='use second half of dataset')
    common_parser.add_argument('--show', action='store_true',
                               help='display tracking in a window')
    common_parser.add_argument('--show-labels', action='store_false',
                               help='hide detection labels')
    common_parser.add_argument('--show-conf', action='store_false',
                               help='hide detection confidences')
    common_parser.add_argument('--show-trajectories', action='store_true',
                               help='overlay past trajectories')
    common_parser.add_argument('--save-txt', action='store_true',
                               help='save results to a .txt file')
    common_parser.add_argument('--save-crop', action='store_true',
                               help='save cropped detections')
    common_parser.add_argument('--save', action='store_true',
                               help='save annotated video')
    common_parser.add_argument('--line-width', type=int, default=None,
                               help='bounding box line width')
    common_parser.add_argument('--per-class', action='store_true',
                               help='track each class separately')

    # Top‐level parser: only for sub‐command selection
    parser = argparse.ArgumentParser(prog='boxmot_cli')
    sub = parser.add_subparsers(dest='command', required=True)

    # Sub-commands inherit their respective flags
    sub.add_parser('track', parents=[common_parser], help='Run tracking only')
    sub.add_parser(
        'generate-dets-embs',
        parents=[common_parser, eval_parent],
        conflict_handler='resolve',
        help='Generate detections and embeddings'
    )
    sub.add_parser(
        'generate-mot-results',
        parents=[common_parser, eval_parent],
        conflict_handler='resolve',
        help='Generate MOT evaluation results'
    )
    sub.add_parser(
        'eval',
        parents=[common_parser, eval_parent],
        conflict_handler='resolve',
        help='Evaluate tracking performance'
    )
    sub.add_parser(
        'tune',
        parents=[common_parser, eval_parent],
        conflict_handler='resolve',
        help='Tune models via evolutionary algorithms'
    )
    sub.add_parser(
        'all',
        parents=[common_parser, eval_parent],
        conflict_handler='resolve',
        help='Run all steps: generate, evaluate, tune'
    )

    # Parse & dispatch
    args = parser.parse_args()
    source_path = Path(args.source)
    args.benchmark, args.split = source_path.parent.name, source_path.name

    if args.command == 'track':
        from boxmot.engine.track import main as run_track
        run_track(args)
    elif args.command == 'generate-dets-embs':
        from boxmot.engine.val import run_generate_dets_embs
        run_generate_dets_embs(args)
    elif args.command == 'generate-mot-results':
        from boxmot.engine.val import run_generate_mot_results
        run_generate_mot_results(args)
    # trackeval only support single class evaluation in its current setup
    elif args.command in ('eval', 'all'):
        from boxmot.engine.val import main as run_eval
        args.classes = [0]
        run_eval(args)
    elif args.command == 'tune':
        from boxmot.engine.evolve import main as run_tuning
        args.classes = [0]
        run_tuning(args)


if __name__ == "__main__":
    main()
