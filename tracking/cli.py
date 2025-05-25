#!/usr/bin/env python3

import click
from pathlib import Path
from types import SimpleNamespace
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS, logger as LOGGER, EXAMPLES


def _normalize_args(args: SimpleNamespace):
    """Convert tuple-based Click multi-options into lists for compatibility and handle single-value imgsz."""
    for attr in ('yolo_model', 'reid_model', 'classes', 'objectives'):
        if hasattr(args, attr):
            val = getattr(args, attr)
            if isinstance(val, tuple):
                setattr(args, attr, list(val))
    # Handle imgsz separately: Click multiple always gives tuple
    if hasattr(args, 'imgsz'):
        val = getattr(args, 'imgsz')
        if isinstance(val, tuple):
            lst = list(val)
            if len(lst) == 1:
                # single value -> square
                lst = [lst[0], lst[0]]
            if len(lst) == 0:
                # no size provided -> None
                args.imgsz = None
            else:
                args.imgsz = lst
    return args


def common_options(func):
    """Decorator for common CLI options (flags only, no positionals)."""
    options = [
        click.option('--source', type=str, default='0',
                     help='file/dir/URL/glob, 0 for webcam'),
        click.option('--imgsz', '--img-size', nargs=2, type=int, default=None, help='inference size h,w')),
        click.option('--fps', type=int, default=None,
                     help='video frame-rate'),
        click.option('--conf', type=float, default=0.01,
                     help='min confidence threshold'),
        click.option('--iou', type=float, default=0.7,
                     help='IoU threshold for NMS'),
        click.option('--device', default='',
                     help='cuda device(s), e.g. 0 or 0,1,2,3 or cpu'),
        click.option('--classes', multiple=True, type=int,
                     default=(0,), help='filter by class indices'),
        click.option('--project', type=click.Path(path_type=Path),
                     default=ROOT / 'runs', help='save results to project/name'),
        click.option('--name', default='',
                     help='save results to project/name'),
        click.option('--exist-ok', is_flag=True, default=True,
                     help='existing project/name ok, do not increment'),
        click.option('--half', is_flag=True,
                     help='use FP16 half-precision inference'),
        click.option('--vid-stride', type=int, default=1,
                     help='video frame-rate stride'),
        click.option('--ci', is_flag=True,
                     help='reuse existing runs in CI (no UI)'),
        click.option('--tracking-method', default='deepocsort',
                     help='deepocsort, botsort, strongsort, ...'),
        click.option('--dets-file-path', type=click.Path(path_type=Path),
                     default=None, help='path to precomputed detections file'),
        click.option('--embs-file-path', type=click.Path(path_type=Path),
                     default=None, help='path to precomputed embeddings file'),
        click.option('--exp-folder-path', type=click.Path(path_type=Path),
                     default=None, help='path to experiment folder'),
        click.option('--verbose', is_flag=True,
                     help='print detailed logs'),
        click.option('--agnostic-nms', is_flag=True,
                     help='class-agnostic NMS'),
        click.option('--gsi', is_flag=True,
                     help='apply Gaussian smoothing interpolation'),
        click.option('--n-trials', type=int, default=4,
                     help='number of trials for evolutionary tuning'),
        click.option('--objectives', multiple=True, type=str,
                     default=('HOTA', 'MOTA', 'IDF1'),
                     help='objectives for tuning: HOTA, MOTA, IDF1'),
        click.option('--val-tools-path', type=click.Path(path_type=Path),
                     default=EXAMPLES / 'val_utils',
                     help='where to clone trackeval'),
        click.option('--split-dataset', is_flag=True,
                     help='use second half of dataset'),
        click.option('--show', is_flag=True,
                     help='display tracking in a window'),
        click.option('--show-labels', default=True, flag_value=False,
                     help='hide detection labels'),
        click.option('--show-conf', default=True, flag_value=False,
                     help='hide detection confidences'),
        click.option('--show-trajectories', is_flag=True,
                     help='overlay past trajectories'),
        click.option('--save-txt', is_flag=True,
                     help='save results to a .txt file'),
        click.option('--save-crop', is_flag=True,
                     help='save cropped detections'),
        click.option('--save', is_flag=True,
                     help='save annotated video'),
        click.option('--line-width', type=int, default=None,
                     help='bounding box line width'),
        click.option('--per-class', is_flag=True,
                     help='track each class separately'),
    ]
    for option in reversed(options):
        func = option(func)
    return func


def single_model_options(func):
    """Decorator for single-model options used by 'track'."""
    options = [
        click.option('--yolo-model', type=click.Path(path_type=Path),
                     default=WEIGHTS / 'yolov8n.pt',
                     help='path to YOLO weights for detection'),
        click.option('--reid-model', type=click.Path(path_type=Path),
                     default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                     help='path to ReID model weights'),
    ]
    for option in reversed(options):
        func = option(func)
    return func


def multi_model_options(func):
    """Decorator for multi-model options used by generate/eval/tune commands."""
    options = [
        click.option('--yolo-model', multiple=True, type=click.Path(path_type=Path),
                     default=(WEIGHTS / 'yolov8n.pt',),
                     help='one or more YOLO weights for detection (only for generate/eval/tune)'),
        click.option('--reid-model', multiple=True, type=click.Path(path_type=Path),
                     default=(WEIGHTS / 'osnet_x0_25_msmt17.pt',),
                     help='one or more ReID model weights (only for generate/eval/tune)'),
    ]
    for option in reversed(options):
        func = option(func)
    return func

@click.group()
def cli():
    """boxmot_cli implemented with Click."""
    pass

@cli.command()
@common_options
@single_model_options
@click.pass_context
def track(ctx, **kwargs):  # Run tracking only
    args = SimpleNamespace(**kwargs)
    args = _normalize_args(args)
    source_path = Path(args.source)
    args.benchmark, args.split = source_path.parent.name, source_path.name
    from tracking.track import main as run_track
    run_track(args)

@cli.command('generate-dets-embs')
@common_options
@multi_model_options
@click.pass_context
def generate_dets_embs(ctx, **kwargs):  # Generate detections and embeddings
    args = SimpleNamespace(**kwargs)
    args = _normalize_args(args)
    source_path = Path(args.source)
    args.benchmark, args.split = source_path.parent.name, source_path.name
    from tracking.val import run_generate_dets_embs
    run_generate_dets_embs(args)

@cli.command('generate-mot-results')
@common_options
@multi_model_options
@click.pass_context
def generate_mot_results(ctx, **kwargs):  # Generate MOT evaluation results
    args = SimpleNamespace(**kwargs)
    args = _normalize_args(args)
    source_path = Path(args.source)
    args.benchmark, args.split = source_path.parent.name, source_path.name
    from tracking.val import run_generate_mot_results
    run_generate_mot_results(args)

@cli.command()
@common_options
@multi_model_options
@click.pass_context
def eval(ctx, **kwargs):  # Evaluate tracking performance
    args = SimpleNamespace(**kwargs)
    args = _normalize_args(args)
    source_path = Path(args.source)
    args.benchmark, args.split = source_path.parent.name, source_path.name
    from tracking.val import main as run_eval
    run_eval(args)

@cli.command()
@common_options
@multi_model_options
@click.pass_context
def tune(ctx, **kwargs):  # Tune models via evolutionary algorithms
    args = SimpleNamespace(**kwargs)
    args = _normalize_args(args)
    source_path = Path(args.source)
    args.benchmark, args.split = source_path.parent.name, source_path.name
    from tracking.evolve import main as run_tuning
    run_tuning(args)

@cli.command('all')
@common_options
@multi_model_options
@click.pass_context
def all_steps(ctx, **kwargs):  # Run all steps: generate, evaluate, tune
    args = SimpleNamespace(**kwargs)
    args = _normalize_args(args)
    source_path = Path(args.source)
    args.benchmark, args.split = source_path.parent.name, source_path.name
    from tracking.val import main as run_eval
    run_eval(args)


if __name__ == '__main__':
    cli()
