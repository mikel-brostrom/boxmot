#!/usr/bin/env python3
import click
from pathlib import Path
from types import SimpleNamespace
from boxmot.utils import ROOT, WEIGHTS, EXAMPLES

def common_options(fn):
    """Shared options for all commands, excluding model weights."""
    opts = [
        click.option('--source', type=str, default='0',
                     help='file/dir/URL/glob, 0 for webcam'),
        click.option('--imgsz', '--img-size', type=int, multiple=True, default=None,  
                     help='inference size h,w'),
        click.option('--fps', type=int, default=None,
                     help='video frame-rate'),
        click.option('--conf', type=float, default=0.01,
                     help='min confidence threshold'),
        click.option('--iou', type=float, default=0.7,
                     help='IoU threshold for NMS'),
        click.option('--device', default='',
                     help='cuda device(s), e.g. 0 or 0,1,2,3 or cpu'),
        click.option('--classes', type=int, multiple=True, default=[0],
                     help='filter by class indices'),
        click.option('--project', type=click.Path(path_type=Path), 
                     default=ROOT / 'runs',
                     help='save results to project/name'),
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
        click.option('--tracking-method', type=str, default='deepocsort',
                     help='deepocsort, botsort, strongsort, ...'),
        click.option('--dets-file-path', type=click.Path(path_type=Path),
                     help='path to precomputed detections file'),
        click.option('--embs-file-path', type=click.Path(path_type=Path),
                     help='path to precomputed embeddings file'),
        click.option('--exp-folder-path', type=click.Path(path_type=Path),
                     help='path to experiment folder'),
        click.option('--verbose', is_flag=True,
                     help='print detailed logs'),
        click.option('--agnostic-nms', is_flag=True,
                     help='class-agnostic NMS'),
        click.option('--gsi', is_flag=True,
                     help='apply Gaussian smoothing interpolation'),
        click.option('--n-trials', type=int, default=4,
                     help='number of trials for evolutionary tuning'),
        click.option('--objectives', type=str, multiple=True,
                     default=['HOTA', 'MOTA', 'IDF1'],
                     help='objectives for tuning: HOTA, MOTA, IDF1'),
        click.option('--val-tools-path', type=click.Path(path_type=Path),
                     default=EXAMPLES / 'val_utils',
                     help='where to clone trackeval'),
        click.option('--split-dataset', is_flag=True,
                     help='use second half of dataset'),
        click.option('--show', is_flag=True,
                     help='display tracking in a window'),
        click.option('--show-labels', is_flag=True,
                     help='hide detection labels'),
        click.option('--show-conf', is_flag=True,
                     help='hide detection confidences'),
        click.option('--show-trajectories', is_flag=True,
                     help='overlay past trajectories'),
        click.option('--save-txt', is_flag=True,
                     help='save results to a .txt file'),
        click.option('--save-crop', is_flag=True,
                     help='save cropped detections'),
        click.option('--save', is_flag=True,
                     help='save annotated video'),
        click.option('--line-width', type=int,
                     help='bounding box line width'),
        click.option('--per-class', is_flag=True,
                     help='track each class separately'),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn

def singular_model_options(fn):
    """Single-weight options for tracking-only."""
    opts = [
        click.option('--yolo-model', type=click.Path(path_type=Path), 
                     default=WEIGHTS / 'yolov8n.pt',
                     help='path to YOLO weights for detection'),
        click.option('--reid-model', type=click.Path(path_type=Path), 
                     default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                     help='path to ReID model weights'),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn

def multi_model_options(fn):
    """Multi-weight options for generate/eval/tune."""
    opts = [
        click.option('--yolo-model', type=click.Path(path_type=Path),
                     multiple=True,
                     default=(WEIGHTS / 'yolov8n.pt',),
                     help='one or more YOLO weights for detection'),
        click.option('--reid-model', type=click.Path(path_type=Path),
                     multiple=True,
                     default=(WEIGHTS / 'osnet_x0_25_msmt17.pt',),
                     help='one or more ReID model weights'),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn

def _build_args(namespace_kwargs):
    """Helper: Turn kwargs into an args namespace, set benchmark & split."""
    args = SimpleNamespace(**namespace_kwargs)
    source_path = Path(args.source)
    args.benchmark = source_path.parent.name
    args.split     = source_path.name
    return args

@click.group()
@click.version_option()
def cli():
    """boxmot_cli: tracking, generation, evaluation, tuning."""
    pass

@cli.command('track', help='Run tracking only')
@common_options
@singular_model_options
def track(yolo_model, reid_model, **kwargs):
    from tracking.track import main as run_track
    ns = {
        'yolo_model': Path(yolo_model),
        'reid_model': Path(reid_model),
        **kwargs
    }
    args = _build_args(ns)
    run_track(args)

@cli.command('generate-dets-embs', help='Generate detections and embeddings')
@common_options
@multi_model_options
def generate_dets_embs(yolo_model, reid_model, **kwargs):
    from tracking.val import run_generate_dets_embs
    ns = {
        'yolo_model': [Path(p) for p in yolo_model],
        'reid_model': [Path(p) for p in reid_model],
        **kwargs
    }
    args = _build_args(ns)
    run_generate_dets_embs(args)

@cli.command('generate-mot-results', help='Generate MOT evaluation results')
@common_options
@multi_model_options
def generate_mot_results(yolo_model, reid_model, **kwargs):
    from tracking.val import run_generate_mot_results
    ns = {
        'yolo_model': [Path(p) for p in yolo_model],
        'reid_model': [Path(p) for p in reid_model],
        **kwargs
    }
    args = _build_args(ns)
    run_generate_mot_results(args)

@cli.command('eval', help='Evaluate tracking performance')
@common_options
@multi_model_options
def evaluate(yolo_model, reid_model, **kwargs):
    from tracking.val import run_all as run_eval
    ns = {
        'yolo_model': [Path(p) for p in yolo_model],
        'reid_model': [Path(p) for p in reid_model],
        **kwargs
    }
    args = _build_args(ns)
    run_eval(args)

@cli.command('tune', help='Tune models via evolutionary algorithms')
@common_options
@multi_model_options
def tune(yolo_model, reid_model, **kwargs):
    from tracking.evolve import main as run_tuning
    ns = {
        'yolo_model': [Path(p) for p in yolo_model],
        'reid_model': [Path(p) for p in reid_model],
        **kwargs
    }
    args = _build_args(ns)
    run_tuning(args)

@cli.command('all', help='Run all steps: generate, evaluate, tune')
@common_options
@multi_model_options
def all_cmd(yolo_model, reid_model, **kwargs):
    # "all" behaves like your original "all" branch: calls run_eval
    from tracking.val import main as run_eval
    ns = {
        'yolo_model': [Path(p) for p in yolo_model],
        'reid_model': [Path(p) for p in reid_model],
        **kwargs
    }
    args = _build_args(ns)
    run_eval(args)

if __name__ == '__main__':
    cli()
