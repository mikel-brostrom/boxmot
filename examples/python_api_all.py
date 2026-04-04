#!/usr/bin/env python3
"""Example script covering the main BoxMOT Python APIs.

This script demonstrates:

- the high-level workflow wrapper via ``boxmot(...)``
- ``val()``
- ``tune()``
- ``track()``
- ``export()``
- the lower-level streaming helper ``track(...)``
- ``create_tracker(...)`` / ``get_tracker_config(...)``
- ``ReID``

Run from the repository root, for example:

```bash
uv run python examples/python_api_all.py \
  --source assets/MOT17-mini/train/MOT17-04-FRCNN/img1 \
  --benchmark mot17-mini \
  --detector yolov8n.pt \
  --reid osnet_x0_25_msmt17.pt \
  --tracker botsort \
  --modes val tune track export stream
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from ultralytics import YOLO

from boxmot import ReID, boxmot, create_tracker, get_tracker_config, track as stream_track

REID_TRACKERS = {"strongsort", "botsort", "deepocsort", "hybridsort", "boosttrack"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all major BoxMOT Python API workflows.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["val", "tune", "track", "export", "stream"],
        choices=["val", "tune", "track", "export", "stream"],
        help="Which Python API workflows to run.",
    )
    parser.add_argument("--benchmark", default="mot17-mini", help="Benchmark name or benchmark YAML path.")
    parser.add_argument("--source", type=str, help="Tracking source for high-level track() and stream demo.")
    parser.add_argument("--detector", default="yolov8n.pt", help="Detector name or weights path.")
    parser.add_argument("--reid", default="osnet_x0_25_msmt17.pt", help="ReID name or weights path.")
    parser.add_argument("--tracker", default="botsort", help="Tracker name.")
    parser.add_argument("--device", default="cpu", help="Device string, for example 'cpu' or '0'.")
    parser.add_argument("--half", action="store_true", help="Enable half precision where supported.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for tracking/eval demos.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for tracking demos.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for tracking demos.")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="Optional class filter.")
    parser.add_argument("--n-trials", type=int, default=3, help="Number of tuning trials.")
    parser.add_argument(
        "--include",
        nargs="+",
        default=["onnx"],
        help="Export targets, for example: onnx engine",
    )
    parser.add_argument("--save", action="store_true", help="Save annotated output in high-level track().")
    parser.add_argument("--save-txt", action="store_true", help="Save MOT text output in high-level track().")
    parser.add_argument(
        "--stream-output",
        type=Path,
        default=Path("runs/python_api_stream.txt"),
        help="Output file for the lower-level streaming helper.",
    )
    parser.add_argument("--stream-limit", type=int, default=5, help="Max frames to print in stream demo.")
    return parser.parse_args()


def require_source(args: argparse.Namespace, modes: Iterable[str]) -> None:
    if any(mode in {"track", "stream"} for mode in modes) and not args.source:
        raise SystemExit("--source is required when running 'track' or 'stream'.")


def make_detector(weights: str):
    """Wrap an Ultralytics model as a BoxMOT-compatible detector callable."""
    model = YOLO(weights)

    def detector(frame: np.ndarray) -> np.ndarray:
        result = model(frame, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().reshape(-1, 1)
        return np.concatenate((xyxy, conf, cls), axis=1).astype(np.float32)

    return detector


def run_high_level_api(args: argparse.Namespace) -> None:
    model = boxmot(
        detector=args.detector,
        reid=args.reid,
        tracker=args.tracker,
        classes=args.classes,
    )

    if "val" in args.modes:
        metrics = model.val(
            benchmark=args.benchmark,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
        )
        print("\n[val]")
        print("summary:", metrics.summary)

    if "tune" in args.modes:
        tune_results = model.tune(
            benchmark=args.benchmark,
            n_trials=args.n_trials,
            device=args.device,
        )
        print("\n[tune]")
        print("best config:", tune_results.best_config)
        print("best yaml:", tune_results.best_yaml)
        print("best metrics:", tune_results.best.metrics.summary)

    if "track" in args.modes:
        track_results = model.track(
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save=args.save,
            save_txt=args.save_txt,
        )
        print("\n[track]")
        print("source:", track_results.source)
        print("video:", track_results.video_path)
        print("text:", track_results.text_path)
        print("timings:", track_results.timings)

    if "export" in args.modes:
        export_results = model.export(
            include=tuple(args.include),
            device=args.device,
            half=args.half,
        )
        print("\n[export]")
        print("weights:", export_results.weights)
        print("files:", export_results.files)


def run_streaming_api(args: argparse.Namespace) -> None:
    detector = make_detector(args.detector)
    reid_callable = ReID(args.reid, device=args.device, half=args.half) if args.tracker in REID_TRACKERS else None
    tracker = create_tracker(
        tracker_type=args.tracker,
        tracker_config=get_tracker_config(args.tracker),
        reid_weights=Path(args.reid),
        device=args.device,
        half=args.half,
        per_class=False,
    )

    stream = stream_track(args.source, detector, reid_callable, tracker, verbose=False)

    print("\n[stream]")
    for idx, frame_tracks in enumerate(stream, start=1):
        print(
            f"frame={frame_tracks.frame_id} "
            f"tracks={len(frame_tracks)} "
            f"ids={frame_tracks.id.tolist()}"
        )
        print("  xyxy:", frame_tracks.xyxy)
        print("  xywha:", frame_tracks.xywha)
        print("  conf:", frame_tracks.conf)
        print("  cls:", frame_tracks.cls)
        print("  det_ind:", frame_tracks.det_ind)
        if idx >= args.stream_limit:
            break

    saved_to = stream.save(args.stream_output)
    print("saved stream output to:", saved_to)


def main() -> None:
    args = parse_args()
    require_source(args, args.modes)

    run_high_level_api(args)

    if "stream" in args.modes:
        run_streaming_api(args)


if __name__ == "__main__":
    main()
