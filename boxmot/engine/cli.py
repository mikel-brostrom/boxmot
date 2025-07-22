#!/usr/bin/env python3
import click
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS, logger as LOGGER, TRACKEVAL


def make_args(**kwargs):
    """
    Helper to build an argparse-style Namespace for engine entrypoints.
    """
    return SimpleNamespace(**kwargs)


def parse_tuple(value: str) -> Tuple[int, int]:
    """
    Parse a comma-separated string into a tuple of two integers.
    """
    try:
        parts = value.split(',')
        if len(parts) != 2:
            raise ValueError("Expected two integers separated by a comma.")
        return int(parts[0]), int(parts[1])
    except Exception as e:
        raise click.BadParameter(f"Invalid format for --imgsz: {value}. {e}")

# Core options (excluding model & classes)
def core_options(func):
    options = [
        click.option('--source', type=str, default='0',
                     help='file/dir/URL/glob, 0 for webcam'),
        click.option('--imgsz', '--img-size', type=Tuple[int, int], default=None,
                     help='inference size h w'),
        click.option('--fps', type=int, default=30,
                     help='video frame-rate'),
        click.option('--conf', type=float, default=0.01,
                     help='min confidence threshold'),
        click.option('--iou', type=float, default=0.7,
                     help='IoU threshold for NMS'),
        click.option('--device', default='',
                     help='cuda device(s), e.g. 0 or 0,1,2,3 or cpu'),
        click.option('--project', type=Path, default=ROOT / 'runs',
                     help='save results to project/name'),
        click.option('--name', default='', help='save results to project/name'),
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
        click.option('--dets-file-path', type=Path,
                     help='path to precomputed detections file'),
        click.option('--embs-file-path', type=Path,
                     help='path to precomputed embeddings file'),
        click.option('--exp-folder-path', type=Path,
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
                     default=["HOTA", "MOTA", "IDF1"],
                     help='objectives for tuning: HOTA, MOTA, IDF1'),
        click.option('--val-tools-path', type=Path, default=TRACKEVAL,
                     help='where to clone trackeval'),
        click.option('--split-dataset', is_flag=True,
                     help='use second half of dataset'),
        click.option('--show', is_flag=True,
                     help='display tracking in a window'),
        click.option('--show-labels/--hide-labels', default=True,
                     help='show or hide detection labels'),
        click.option('--show-conf/--hide-conf', default=True,
                     help='show or hide detection confidences'),
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
                     help='track each class separately')
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


def singular_model_options(func):
    options = [
        click.option('--yolo-model', type=Path,
                     default=WEIGHTS / 'yolov8n.pt',
                     help='path to YOLO weights for detection'),
        click.option('--reid-model', type=Path,
                     default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                     help='path to ReID model weights'),
        click.option('--classes', type=int, multiple=True,
                     help='filter by class indices')
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


def plural_model_options(func):
    options = [
        click.option('--yolo-model', type=Path, multiple=True,
                     default=[WEIGHTS / 'yolov8n.pt'],
                     help='one or more YOLO weights for detection'),
        click.option('--reid-model', type=Path, multiple=True,
                     default=[WEIGHTS / 'osnet_x0_25_msmt17.pt'],
                     help='one or more ReID model weights'),
        click.option('--classes', type=int, multiple=True,
                     default=[0], help='filter by class indices')
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


@click.group(context_settings=dict(help_option_names=['-h', '--help']),
             invoke_without_command=False)
def cli():
    """boxmot_cli: multi-step MOT pipeline"""
    pass


@cli.command(help='Run tracking only')
@core_options
@singular_model_options
@click.pass_context
def track(ctx, yolo_model, reid_model, classes, **kwargs):
    src = kwargs.pop('source')
    source_path = Path(src)
    bench, split = source_path.parent.name, source_path.name
    params = {**kwargs,
              'yolo_model': yolo_model,
              'reid_model': reid_model,
              'classes': list(classes) if classes else None,
              'source': src,
              'benchmark': bench,
              'split': split}
    args = make_args(**params)
    from boxmot.engine.track import main as run_track
    run_track(args)


@cli.command(help='Generate detections and embeddings')
@core_options
@plural_model_options
@click.pass_context
def generate(ctx, yolo_model, reid_model, classes, **kwargs):
    src = kwargs.pop('source')
    source_path = Path(src)
    bench, split = source_path.parent.name, source_path.name
    params = {**kwargs,
              'yolo_model': list(yolo_model),
              'reid_model': list(reid_model),
              'classes': list(classes),
              'source': src,
              'benchmark': bench,
              'split': split}
    args = make_args(**params)
    from boxmot.engine.val import run_generate_dets_embs
    run_generate_dets_embs(args)


@cli.command(help='Evaluate tracking performance')
@core_options
@plural_model_options
@click.pass_context
def eval(ctx, yolo_model, reid_model, classes, **kwargs):
    src = kwargs.pop('source')
    source_path = Path(src)
    bench, split = source_path.parent.name, source_path.name
    params = {**kwargs,
              'yolo_model': list(yolo_model),
              'reid_model': list(reid_model),
              'classes': [0],
              'source': src,
              'benchmark': bench,
              'split': split}
    args = make_args(**params)
    from boxmot.engine.val import main as run_eval
    run_eval(args)


@cli.command(help='Tune models via evolutionary algorithms')
@core_options
@plural_model_options
@click.pass_context
def tune(ctx, yolo_model, reid_model, classes, **kwargs):
    src = kwargs.pop('source')
    source_path = Path(src)
    bench, split = source_path.parent.name, source_path.name
    params = {**kwargs,
              'yolo_model': list(yolo_model),
              'reid_model': list(reid_model),
              'classes': list(classes),
              'source': src,
              'benchmark': bench,
              'split': split}
    args = make_args(**params)
    from boxmot.engine.evolve import main as run_tuning
    run_tuning(args)


main = cli

if __name__ == "__main__":
    cli()
