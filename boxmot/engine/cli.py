#!/usr/bin/env python3
"""
CLI for BoxMOT: multi-step multiple object tracking pipeline.
Provides commands to track, generate detections and embeddings, evaluate performance, tune models, or run all steps.
"""
import click
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS, logger as LOGGER, TRACKEVAL

    
def make_args(**kwargs):
    """
    Build an argparse-style namespace for engine entrypoints.

    Converts any 'imgsz' tuple in kwargs to a list and returns a SimpleNamespace.

    Args:
        **kwargs: Keyword arguments corresponding to CLI options.

    Returns:
        SimpleNamespace: Namespace object with attributes set from kwargs.
    """
    # Convert imgsz tuple to list if provided
    if 'imgsz' in kwargs and kwargs['imgsz'] is not None:
        size = kwargs['imgsz']
        kwargs['imgsz'] = [size[0], size[1]]
    return SimpleNamespace(**kwargs)


def parse_tuple(value: str) -> Tuple[int, int]:
    """
    Parse a string into a (width, height) tuple of integers.

    Accepts separators 'x', ',' or space, and a single value for square size.

    Args:
        value (str): Input string, e.g. "640x480", "640,480", "320".

    Returns:
        Tuple[int, int]: Parsed (width, height) tuple.

    Raises:
        click.BadParameter: If the input cannot be parsed as one or two integers.
    """
    s = value.replace('x', ' ').replace(',', ' ')
    parts = s.split()
    if len(parts) == 1:
        try:
            n = int(parts[0])
            return (n, n)
        except ValueError:
            raise click.BadParameter(f"Invalid --imgsz: {value}")
    elif len(parts) == 2:
        try:
            w, h = int(parts[0]), int(parts[1])
            return (w, h)
        except ValueError:
            raise click.BadParameter(f"Invalid --imgsz: {value}")
    else:
        raise click.BadParameter(
            f"--imgsz expects 1 or 2 integers separated by ',' 'x' or space, got '{value}'"
        )


# Core options (excluding model & classes)
def core_options(func):
    """
    Decorator adding core CLI options to a command.

    Options include data source, image size, frame rate, thresholds, devices,
    output parameters, visualization and tracking flags.

    Args:
        func (callable): Click command function to wrap.

    Returns:
        callable: Wrapped function with core options applied.
    """
    options = [
        click.option('--source', type=str, default='0',
                     help='file/dir/URL/glob, 0 for webcam'),
        click.option('--imgsz', '--img-size', type=parse_tuple, default=None,
                     help='inference size h,w (e.g. 640,480 or 640x480)'),
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
                     help='track each class separately'),
        click.option('--target-id', type=int, default=None,
                     help='ID to highlight in green')
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


def singular_model_options(func):
    """
    Decorator adding single-model options to a command.

    Options include paths for YOLO detection weights, ReID model weights,
    and class filters.

    Args:
        func (callable): Click command function to wrap.

    Returns:
        callable: Wrapped function with singular model options applied.
    """
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
    """
    Decorator adding multi-model options to a command.

    Options allow providing multiple YOLO weights, multiple ReID weights,
    and class filters.

    Args:
        func (callable): Click command function to wrap.

    Returns:
        callable: Wrapped function with plural model options applied.
    """
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


class CommandFirstGroup(click.Group):
    """Show  COMMAND [OPTIONS]...  instead of  [OPTIONS] COMMAND …"""
    def format_usage(self, ctx, formatter):
        # ctx.command_path == "boxmot"
        formatter.write_usage(ctx.command_path, "COMMAND [ARGS]...")
        
@click.group(cls=CommandFirstGroup)   # ← NEW API
def boxmot():
    """
    BoxMOT: Pluggable SOTA multi-object tracking modules modules for segmentation, object detection and pose estimation models
    """
    pass


@boxmot.command(help='Run tracking only')
@core_options
@singular_model_options
@click.pass_context
def track(ctx, yolo_model, reid_model, classes, **kwargs):
    """
    Command 'track': run object tracking only.

    Loads a single YOLO detector and ReID model to track objects in the given source.
    """
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


@boxmot.command(help='Generate detections and embeddings')
@core_options
@plural_model_options
@click.pass_context
def generate(ctx, yolo_model, reid_model, classes, **kwargs):
    """
    Command 'generate': produce detection boxes and embedding vectors.

    Uses one or more YOLO models for detection and one or more ReID models
    to compute embeddings for each detection in the source dataset.
    """
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


@boxmot.command(help='Evaluate tracking performance')
@core_options
@plural_model_options
@click.pass_context
def eval(ctx, yolo_model, reid_model, classes, **kwargs):
    """
    Command 'eval': evaluate tracking results with TrackEval.

    Runs performance metrics (HOTA, MOTA, IDF1, etc.) on precomputed detections and
    embeddings for the source.
    """
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


@boxmot.command(help='Tune models via evolutionary algorithms')
@core_options
@plural_model_options
@click.pass_context
def tune(ctx, yolo_model, reid_model, classes, **kwargs):
    """
    Command 'tune': optimize model hyperparameters using evolutionary search.

    Performs multiple trials to tune tracking parameters based on specified objectives.
    """
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


@boxmot.command(help='Run all steps: generate, evaluate, tune')
@core_options
@plural_model_options
@click.pass_context
def all(ctx, yolo_model, reid_model, classes, **kwargs):
    """
    Command 'all': execute full pipeline.

    Runs generate, eval, and tune sequentially for the given source and models.
    """
    src = kwargs.pop('source')
    source_path = Path(src)
    bench, split = source_path.parent.name, source_path.name
    params = {**kwargs,
              'yolo_model': list(yolo-model),
              'reid_model': list(reid-model),
              'classes': [0],
              'source': src,
              'benchmark': bench,
              'split': split}
    args = make_args(**params)
    # generate
    from boxmot.engine.val import run_generate_dets_embs
    run_generate_dets_embs(args)
    # eval
    from boxmot.engine.val import main as run_eval
    run_eval(args)
    # tune
    from boxmot.engine.evolve import main as run_tuning
    run_tuning(args)


main = boxmot

if __name__ == "__main__":
    boxmot()