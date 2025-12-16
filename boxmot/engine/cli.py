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
from boxmot.utils.misc import parse_imgsz


def ensure_model_extension(model_path):
    """
    Ensure model path has .pt extension.
    
    Args:
        model_path: Path to model file (str or Path)
        
    Returns:
        Path with .pt extension
    """
    if model_path is None:
        return None
    
    model_path = Path(model_path)
    # If no extension, add .pt
    if not model_path.suffix and "openvino" not in model_path.name:
        model_path = model_path.with_suffix('.pt')
    
    return model_path


# Core options (excluding model & classes)
def core_options(func):
    options = [
        click.option('--source', type=str, default='0',
                     help='file/dir/URL/glob, 0 for webcam'),
        click.option('--imgsz', callback=parse_imgsz, default=640, type=str,
                     help='desired image size for the model input. Can be an integer for square images or a tuple (height, width) for specific dimensions.'),
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


def export_options(func):
    """
    Decorator adding ReID export options (ported from argparse export script).
    """
    options = [
        click.option('--batch-size', type=int, default=1,
                     help='Batch size for export'),
        click.option('--imgsz', '--img', '--img-size', callback=parse_imgsz, type=str,
                     default=640, help='Image size as H,W (e.g. 256,128)'),
        click.option('--device', default='cpu',
                     help="CUDA device (e.g., '0', '0,1,2,3', or 'cpu')"),
        click.option('--optimize', is_flag=True,
                     help='Optimize TorchScript for mobile (CPU export only)'),
        click.option('--dynamic', is_flag=True,
                     help='Enable dynamic axes for ONNX/TF/TensorRT export'),
        click.option('--simplify', is_flag=True,
                     help='Simplify ONNX model'),
        click.option('--opset', type=int, default=18,
                     help='ONNX opset version'),
        click.option('--workspace', type=int, default=4,
                     help='TensorRT workspace size (GB)'),
        click.option('--verbose', is_flag=True,
                     help='Enable verbose logging for TensorRT'),
        click.option('--weights', type=Path,
                     default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                     help='Path to the model weights (.pt file)'),
        click.option('--half', is_flag=True,
                     help='Enable FP16 half-precision export (GPU only)'),
        click.option('--include', multiple=True, default=('torchscript',),
                     help='Export formats to include. Options: torchscript, onnx, openvino, engine, tflite'),
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


def tune_options(func):
    """
    Decorator adding ReID export options (ported from argparse export script).
    """
    options = [
        click.option('--n-trials', type=int, default=4,
                     help='number of trials for evolutionary tuning'),
        click.option('--objectives', type=str, multiple=True,
                     default=["HOTA", "MOTA", "IDF1"],
                     help='objectives for tuning: HOTA, MOTA, IDF1'),
    ]
    for opt in reversed(options):
        func = opt(func)
    return func



class CommandFirstGroup(click.Group):
    """Custom Click Group with improved help formatting."""
    
    def format_usage(self, ctx, formatter):
        formatter.write_text("Usage: boxmot TRACKER DETECTOR REID [OPTIONS]")
        formatter.write_text("   or: boxmot COMMAND [OPTIONS]")
    
    def format_help(self, ctx, formatter):
        """Override to show custom help with examples."""
        self.format_usage(ctx, formatter)
        formatter.write_paragraph()
        
        # Description
        formatter.write_text(
            "BoxMOT: Pluggable SOTA multi-object tracking modules for segmentation, "
            "object detection and pose estimation models"
        )
        formatter.write_paragraph()
        
        formatter.write_text("Shorthand:")
        with formatter.indentation():
            formatter.write_text("boxmot MODE DETECTOR REID TRACKER [OPTIONS]")
        formatter.write_paragraph()
        
        formatter.write_text("Arguments:")
        with formatter.indentation():
            formatter.write_text("MODE      Subcommand (track, eval, tune, generate)")
            formatter.write_text("DETECTOR  YOLO model name (yolov8n, yolov9c, etc.)")
            formatter.write_text("REID      ReID model name (osnet_x0_25_msmt17, etc.)")
            formatter.write_text("TRACKER   Tracking method (deepocsort, botsort, bytetrack, etc.)")
        formatter.write_paragraph()
        
        # Commands section
        formatter.write_text("Commands (MODE):")
        with formatter.indentation():
            formatter.write_text("track      Run tracking on video/webcam")
            formatter.write_text("generate   Generate detections and embeddings")
            formatter.write_text("eval       Evaluate tracking performance")
            formatter.write_text("tune       Tune tracker hyperparameters")
            formatter.write_text("export     Export ReID models")
            formatter.write_text("version    Show BoxMOT version")
            formatter.write_text("help       Show this help message")
        formatter.write_paragraph()
        
        # Examples section
        formatter.write_text("Examples:")
        with formatter.indentation():
            formatter.write_text("1. Track with shorthand syntax (webcam):")
            with formatter.indentation():
                formatter.write_text("boxmot track yolov8n osnet_x0_25_msmt17 deepocsort --source 0 --show")
            formatter.write_paragraph()
            
            formatter.write_text("2. Track with explicit command (video file):")
            with formatter.indentation():
                formatter.write_text("boxmot track --source video.mp4 --yolo-model yolov8n --reid-model osnet_x0_25_msmt17 --tracking-method botsort")
            formatter.write_paragraph()
            
            formatter.write_text("3. Evaluate tracking performance on MOT dataset:")
            with formatter.indentation():
                formatter.write_text("boxmot eval yolov8n osnet_x0_25_msmt17 deepocsort --source MOT17-ablation")
            formatter.write_paragraph()
            
            formatter.write_text("4. Tune tracker hyperparameters:")
            with formatter.indentation():
                formatter.write_text("boxmot tune --source MOT17-ablation --tracking-method deepocsort --n-trials 10")
            formatter.write_paragraph()
            
            formatter.write_text("5. Show version or get help:")
            with formatter.indentation():
                formatter.write_text("boxmot version")
                formatter.write_text("boxmot track --help")
        formatter.write_paragraph()
        
        # Links
        formatter.write_text("Resources:")
        with formatter.indentation():
            formatter.write_text("Docs:      https://github.com/mikel-brostrom/yolo_tracking")
            formatter.write_text("GitHub:    https://github.com/mikel-brostrom/yolo_tracking")
            formatter.write_text("Community: https://github.com/mikel-brostrom/yolo_tracking/discussions")


@click.group(cls=CommandFirstGroup)
@click.pass_context
def boxmot(ctx):
    """
    BoxMOT: Pluggable SOTA multi-object tracking modules for segmentation, object detection and pose estimation models
    """
    pass


@boxmot.command(help='Run tracking only')
@click.argument('detector', required=False)
@click.argument('reid', required=False)
@click.argument('tracker', required=False)
@core_options
@singular_model_options
@click.pass_context
def track(ctx, detector, reid, tracker, yolo_model, reid_model, classes, **kwargs):
    # Override options with positional args if provided
    if detector:
        yolo_model = ensure_model_extension(detector)
    if reid:
        reid_model = ensure_model_extension(reid)
    if tracker:
        kwargs['tracking_method'] = tracker
    src = kwargs.pop('source')
    source_path = Path(src)
    bench, split = source_path.parent.name, source_path.name
    
    # Auto-append .pt extension if missing
    yolo_model = ensure_model_extension(yolo_model)
    reid_model = ensure_model_extension(reid_model)
    
    params = {**kwargs,
              'yolo_model': yolo_model,
              'reid_model': reid_model,
              'classes': list(classes) if classes else None,
              'source': src,
              'benchmark': bench,
              'split': split}
    args = SimpleNamespace(**params)
    from boxmot.engine.tracker import main as run_track
    run_track(args)


@boxmot.command(help='Generate detections and embeddings')
@click.argument('detector', required=False)
@click.argument('reid', required=False)
@core_options
@plural_model_options
@click.pass_context
def generate(ctx, detector, reid, yolo_model, reid_model, classes, **kwargs):
    # Override options with positional args if provided
    # Note: Plural options are tuples, so handle single arg input as list
    if detector:
        yolo_model = [ensure_model_extension(detector)]
    if reid:
        reid_model = [ensure_model_extension(reid)]
    src = kwargs.pop('source')
    source_path = Path(src)
    bench, split = source_path.parent.name, source_path.name
    
    # Auto-append .pt extension if missing
    yolo_model = [ensure_model_extension(m) for m in yolo_model]
    reid_model = [ensure_model_extension(m) for m in reid_model]
    
    params = {**kwargs,
              'yolo_model': list(yolo_model),
              'reid_model': list(reid_model),
              'classes': list(classes),
              'source': src,
              'benchmark': bench,
              'split': split}
    args = SimpleNamespace(**params)
    from boxmot.engine.evaluator import run_generate_dets_embs
    run_generate_dets_embs(args)


@boxmot.command(help='Evaluate tracking performance')
@click.argument('detector', required=False)
@click.argument('reid', required=False)
@click.argument('tracker', required=False)
@core_options
@plural_model_options
@click.pass_context
def eval(ctx, detector, reid, tracker, yolo_model, reid_model, classes, **kwargs):
    # Override options with positional args if provided
    # Note: Plural options are tuples, so handle single arg input as list
    if detector:
        yolo_model = [ensure_model_extension(detector)]
    if reid:
        reid_model = [ensure_model_extension(reid)]
    if tracker:
        kwargs['tracking_method'] = tracker
    src = kwargs.pop('source')
    source_path = Path(src)
    bench, split = source_path.parent.name, source_path.name
    
    # Auto-append .pt extension if missing
    yolo_model = [ensure_model_extension(m) for m in yolo_model]
    reid_model = [ensure_model_extension(m) for m in reid_model]
    
    params = {**kwargs,
              'yolo_model': list(yolo_model),
              'reid_model': list(reid_model),
              'classes': [0],
              'source': src,
              'benchmark': bench,
              'split': split}
    args = SimpleNamespace(**params)
    from boxmot.engine.evaluator import main as run_eval
    run_eval(args)


@boxmot.command(help='Tune models via evolutionary algorithms')
@click.argument('detector', required=False)
@click.argument('reid', required=False)
@click.argument('tracker', required=False)
@core_options
@tune_options
@plural_model_options
@click.pass_context
def tune(ctx, detector, reid, tracker, yolo_model, reid_model, classes, **kwargs):
    # Override options with positional args if provided
    # Note: Plural options are tuples, so handle single arg input as list
    if detector:
        yolo_model = [ensure_model_extension(detector)]
    if reid:
        reid_model = [ensure_model_extension(reid)]
    if tracker:
        kwargs['tracking_method'] = tracker
    src = kwargs.pop('source')
    source_path = Path(src)
    bench, split = source_path.parent.name, source_path.name
    
    # Auto-append .pt extension if missing
    yolo_model = [ensure_model_extension(m) for m in yolo_model]
    reid_model = [ensure_model_extension(m) for m in reid_model]
    
    params = {**kwargs,
              'yolo_model': list(yolo_model),
              'reid_model': list(reid_model),
              'classes': list(classes),
              'source': src,
              'benchmark': bench,
              'split': split}
    args = SimpleNamespace(**params)
    from boxmot.engine.tuner import main as run_tuning
    run_tuning(args)


@boxmot.command(help='Export ReID models')
@export_options
@click.pass_context
def export(ctx, **kwargs):
    """
    Command 'export': export ReID model weights and configurations for deployment.
    Mirrors the standalone argparse-based export script.
    """
    # kwargs already contains all export args; convert imgsz tuple -> list
    args = SimpleNamespace(**kwargs)
    from boxmot.appearance.reid.export import main as run_export
    run_export(args)


@boxmot.command(help='Show BoxMOT version')
def version():
    """Display the current BoxMOT version."""
    from boxmot import __version__
    click.echo(f"BoxMOT {__version__}")


@boxmot.command(help='Show help information')
@click.pass_context
def help(ctx):
    """Display help information."""
    # Get the parent context (main boxmot group)
    parent_ctx = ctx.parent
    click.echo(parent_ctx.get_help())


main = boxmot

if __name__ == "__main__":
    boxmot()
