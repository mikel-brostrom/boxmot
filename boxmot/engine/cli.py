#!/usr/bin/env python3
import click
from pathlib import Path
from types import SimpleNamespace
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS, logger as LOGGER, TRACKEVAL


def parse_size(value: str):
    """
    Parse a size string into a list of two ints. Supports:
      - single int (e.g. "320" → [320, 320])
      - comma or x separated (e.g. "640,480" or "640x480")
      - space separated ("640 480")
    """
    if value is None:
        return None
    s = value.replace('x', ' ').replace(',', ' ')
    parts = s.split()
    if len(parts) == 1:
        try:
            n = int(parts[0])
            return [n, n]
        except ValueError:
            raise click.BadParameter(f"Invalid --imgsz '{value}', must be int")
    elif len(parts) == 2:
        try:
            w, h = int(parts[0]), int(parts[1])
            return [w, h]
        except ValueError:
            raise click.BadParameter(f"Invalid --imgsz '{value}', values must be int")
    else:
        raise click.BadParameter(
            f"--imgsz expects 1 or 2 ints, got {len(parts)}")


def make_args(**kwargs):
    """
    Build argparse-style Namespace for engine.
    """
    # Parse and normalize imgsz
    if 'imgsz' in kwargs:
        raw = kwargs.get('imgsz')
        if isinstance(raw, str):
            kwargs['imgsz'] = parse_size(raw)
    return SimpleNamespace(**kwargs)


# Core options (excluding model & classes)
def core_options(func):
    opts = [
        click.option('--source', type=str, default='0',
                     help='file/dir/URL/glob, 0 for webcam'),
        click.option('--imgsz', '--img-size', type=str, default=None,
                     help='inference size h,w (e.g. 640,480 or 640x480 or "320")'),
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
                     help='existing project/name ok'),
        click.option('--half', is_flag=True,
                     help='use FP16 half-precision inference'),
        click.option('--vid-stride', type=int, default=1,
                     help='video frame-rate stride'),
        click.option('--ci', is_flag=True,
                     help='reuse existing runs in CI'),
        click.option('--tracking-method', type=str, default='deepocsort',
                     help='tracking algorithm'),
        click.option('--dets-file-path', type=Path,
                     help='precomputed detections file'),
        click.option('--embs-file-path', type=Path,
                     help='precomputed embeddings file'),
        click.option('--exp-folder-path', type=Path,
                     help='experiment folder'),
        click.option('--verbose', is_flag=True,
                     help='detailed logs'),
        click.option('--agnostic-nms', is_flag=True,
                     help='class-agnostic NMS'),
        click.option('--gsi', is_flag=True,
                     help='Gaussian smoothing interpolation'),
        click.option('--n-trials', type=int, default=4,
                     help='evolutionary tuning trials'),
        click.option('--objectives', type=str, multiple=True,
                     default=["HOTA","MOTA","IDF1"],
                     help='tuning objectives'),
        click.option('--val-tools-path', type=Path, default=TRACKEVAL,
                     help='trackeval path'),
        click.option('--split-dataset', is_flag=True,
                     help='use second half of dataset'),
        click.option('--show', is_flag=True,
                     help='display window'),
        click.option('--show-labels/--hide-labels', default=True,
                     help='show detection labels'),
        click.option('--show-conf/--hide-conf', default=True,
                     help='show detection confidences'),
        click.option('--show-trajectories', is_flag=True,
                     help='overlay trajectories'),
        click.option('--save-txt', is_flag=True,
                     help='save .txt'),
        click.option('--save-crop', is_flag=True,
                     help='save crops'),
        click.option('--save', is_flag=True,
                     help='save video'),
        click.option('--line-width', type=int,
                     help='bbox line width'),
        click.option('--per-class', is_flag=True,
                     help='track per class')
    ]
    for o in reversed(opts):
        func = o(func)
    return func


def singular_model_options(func):
    opts = [
        click.option('--yolo-model', type=Path,
                     default=WEIGHTS / 'yolov8n.pt',
                     help='YOLO weights'),
        click.option('--reid-model', type=Path,
                     default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                     help='ReID weights'),
        click.option('--classes', type=int, multiple=True,
                     help='class indices')
    ]
    for o in reversed(opts): func = o(func)
    return func


def plural_model_options(func):
    opts = [
        click.option('--yolo-model', type=Path, multiple=True,
                     default=[WEIGHTS / 'yolov8n.pt'],
                     help='YOLO weights list'),
        click.option('--reid-model', type=Path, multiple=True,
                     default=[WEIGHTS / 'osnet_x0_25_msmt17.pt'],
                     help='ReID weights list'),
        click.option('--classes', type=int, multiple=True,
                     default=[0], help='class indices')
    ]
    for o in reversed(opts): func = o(func)
    return func

@click.group(context_settings=dict(help_option_names=['-h','--help']))
def cli():
    """boxmot_cli: multi-step MOT pipeline"""
    pass

# subcommands... (track, generate, eval, tune, all) identical to before

main = cli

if __name__ == '__main__':
    cli()
