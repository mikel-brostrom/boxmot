"""Benchmark ``tracker.update(...)`` throughput with synthetic detections.

The default sweep benchmarks every registered native-capable tracker with the
Python and C++ backends, 10/50/100 detections, 5 warmup frames, and 100 measured
frames. Appearance trackers receive deterministic precomputed embeddings by
default, so the timing isolates tracker work from detector and ReID inference.

Run the default sweep::

    uv run --no-sync python -m tests.performance.trackers.benchmark_fps

Restrict the sweep or include live ReID inference::

    uv run --no-sync python -m tests.performance.trackers.benchmark_fps \
        --trackers botsort,ocsort --backends python --counts 50,100
    uv run --no-sync python -m tests.performance.trackers.benchmark_fps \
        --trackers botsort --reid-mode live --reid osnet_x0_25_msmt17.pt

Save the same result rows as JSON or CSV::

    uv run --no-sync python -m tests.performance.trackers.benchmark_fps \
        --json results/tracker_fps.json --csv results/tracker_fps.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from boxmot.native.registry import supported_native_live_trackers
from boxmot.trackers.registry import REID_TRACKERS, TRACKER_MAPPING, create_tracker
from boxmot.utils import logger as LOGGER

FloatArray = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]
ReIDMode = Literal["precomputed", "live"]
ResultRow = dict[str, Any]

NATIVE_TRACKERS = supported_native_live_trackers()
DEFAULT_TRACKERS = NATIVE_TRACKERS
DEFAULT_BACKENDS = ("python", "cpp")
DEFAULT_COUNTS = (10, 50, 100)
DEFAULT_WARMUP = 5
DEFAULT_FRAMES = 100
DEFAULT_IMG_HW = (1080, 1920)
DEFAULT_REID_WEIGHTS = "osnet_x0_25_msmt17.pt"
DEFAULT_EMBEDDING_DIM = 512
REID_TRACKER_NAMES = frozenset(REID_TRACKERS)

CSV_FIELDS = (
    "tracker",
    "backend",
    "reid_mode",
    "reid_backend",
    "reid_weights",
    "n_dets",
    "frames",
    "elapsed_s",
    "fps",
    "ms_per_frame",
    "error",
)

_PROVIDER_TO_DEVICE = {
    "CUDAExecutionProvider": "cuda",
    "TensorrtExecutionProvider": "cuda",
    "CoreMLExecutionProvider": "coreml",
    "CPUExecutionProvider": "cpu",
}


@dataclass(frozen=True, slots=True)
class FrameInput:
    """Inputs supplied to one tracker update."""

    detections: FloatArray
    embeddings: FloatArray | None


def _make_random_detections(
    count: int,
    image_hw: tuple[int, int],
    rng: np.random.Generator,
) -> FloatArray:
    """Create valid synthetic AABB detections for a positive image size."""

    height, width = image_hw
    x1 = rng.uniform(0.02 * width, 0.82 * width, size=count)
    y1 = rng.uniform(0.02 * height, 0.72 * height, size=count)
    box_width = rng.uniform(0.05 * width, 0.15 * width, size=count)
    box_height = rng.uniform(0.08 * height, 0.25 * height, size=count)
    x2 = np.minimum(x1 + box_width, width)
    y2 = np.minimum(y1 + box_height, height)
    confidence = rng.uniform(0.55, 0.95, size=count)
    classes = np.zeros(count, dtype=np.float32)
    return np.stack([x1, y1, x2, y2, confidence, classes], axis=1).astype(np.float32)


def _jitter_detections(
    detections: FloatArray,
    rng: np.random.Generator,
    image_hw: tuple[int, int],
) -> FloatArray:
    """Apply deterministic small motion to a frame of detections."""

    height, width = image_hw
    output = detections.copy()
    dx = rng.normal(0.0, max(width * 0.002, 0.01), size=len(detections)).astype(np.float32)
    dy = rng.normal(0.0, max(height * 0.002, 0.01), size=len(detections)).astype(np.float32)
    output[:, 0] = np.clip(output[:, 0] + dx, 0, width)
    output[:, 2] = np.clip(output[:, 2] + dx, 0, width)
    output[:, 1] = np.clip(output[:, 1] + dy, 0, height)
    output[:, 3] = np.clip(output[:, 3] + dy, 0, height)
    return output


def _normalize_embeddings(embeddings: FloatArray) -> FloatArray:
    """L2-normalize appearance embeddings row-wise."""

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, np.finfo(np.float32).eps)
    return normalized.astype(np.float32, copy=False)


def _make_random_embeddings(
    count: int,
    rng: np.random.Generator,
    dimension: int = DEFAULT_EMBEDDING_DIM,
) -> FloatArray:
    """Create deterministic normalized base embeddings."""

    embeddings = rng.standard_normal((count, dimension), dtype=np.float32)
    return _normalize_embeddings(embeddings)


def _jitter_embeddings(base: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Create a normalized embedding frame while preserving synthetic identities."""

    noise = rng.normal(0.0, 0.01, size=base.shape).astype(np.float32)
    return _normalize_embeddings(base + noise)


def _build_workload(
    *,
    detection_count: int,
    frame_count: int,
    image_hw: tuple[int, int],
    seed: int,
    with_embeddings: bool,
) -> tuple[UInt8Array, tuple[FrameInput, ...]]:
    """Precompute every image, detection, and embedding input before timing."""

    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(*image_hw, 3), dtype=np.uint8)
    base_detections = _make_random_detections(detection_count, image_hw, rng)
    detection_frames = tuple(_jitter_detections(base_detections, rng, image_hw) for _ in range(frame_count))

    if with_embeddings:
        base_embeddings = _make_random_embeddings(detection_count, rng)
        embedding_frames: tuple[FloatArray | None, ...] = tuple(
            _jitter_embeddings(base_embeddings, rng) for _ in range(frame_count)
        )
    else:
        embedding_frames = (None,) * frame_count

    frames = tuple(
        FrameInput(detections=detections, embeddings=embeddings)
        for detections, embeddings in zip(detection_frames, embedding_frames)
    )
    return image, frames


def _build_tracker(
    tracker_name: str,
    backend: str,
    *,
    reid_mode: ReIDMode,
    reid_weights: str | Path,
    device: str,
    half: bool,
) -> Any:
    """Construct a tracker for precomputed or live appearance inference."""

    uses_reid = tracker_name in REID_TRACKER_NAMES
    use_live_reid = uses_reid and reid_mode == "live"
    return create_tracker(
        tracker_type=tracker_name,
        reid_weights=reid_weights if use_live_reid else None,
        device=device,
        half=half,
        per_class=False,
        tracker_backend=backend,
        precomputed_reid=uses_reid and reid_mode == "precomputed",
    )


def _resolve_live_reid_label(
    tracker: Any,
    tracker_name: str,
    backend: str,
    reid_weights: str | Path,
    device: str,
) -> str:
    """Return a best-effort label for the live ReID runtime."""

    if tracker_name not in REID_TRACKER_NAMES:
        return "—"

    if backend == "cpp":
        environment_backend = os.environ.get("BOXMOT_REID_BACKEND", "").lower()
        runtime_format = "opencv" if environment_backend in {"opencv", "dnn"} else "onnx"
        environment_device = os.environ.get("BOXMOT_REID_DEVICE", "").lower()
        if environment_device in {"cpu", "cuda", "gpu", "coreml", "mps", "metal"}:
            runtime_device = {
                "gpu": "cuda",
                "mps": "coreml",
                "metal": "coreml",
            }.get(environment_device, environment_device)
        else:
            runtime_device = "auto"
        return f"{runtime_format}-{runtime_device}"

    model = getattr(tracker, "model", None)
    if model is None:
        suffix = Path(str(reid_weights)).suffix.lower().lstrip(".") or "pt"
        return f"{suffix}-{device}"

    class_name = model.__class__.__name__
    format_by_class = {
        "PyTorchBackend": "pt",
        "ONNXBackend": "onnx",
        "TorchscriptBackend": "torchscript",
        "TensorRTBackend": "tensorrt",
        "OpenVinoBackend": "openvino",
        "TFLiteBackend": "tflite",
    }
    runtime_format = format_by_class.get(class_name, class_name.removesuffix("Backend").lower())
    if class_name == "ONNXBackend":
        session = getattr(model, "session", None)
        providers = list(session.get_providers()) if session is not None else []
        runtime_device = next(
            (_PROVIDER_TO_DEVICE[provider] for provider in providers if provider in _PROVIDER_TO_DEVICE),
            device,
        )
    else:
        device_object = getattr(model, "device", device)
        runtime_device = str(getattr(device_object, "type", device_object))
    return f"{runtime_format}-{runtime_device}"


def _measure(
    tracker_name: str,
    backend: str,
    detection_count: int,
    *,
    warmup_count: int,
    measured_count: int,
    image_hw: tuple[int, int],
    seed: int,
    reid_mode: ReIDMode,
    reid_weights: str | Path,
    device: str,
    half: bool,
) -> ResultRow:
    """Measure one tracker/backend/detection-count combination."""

    uses_reid = tracker_name in REID_TRACKER_NAMES
    image, frames = _build_workload(
        detection_count=detection_count,
        frame_count=warmup_count + measured_count,
        image_hw=image_hw,
        seed=seed,
        with_embeddings=uses_reid and reid_mode == "precomputed",
    )
    tracker = _build_tracker(
        tracker_name,
        backend,
        reid_mode=reid_mode,
        reid_weights=reid_weights,
        device=device,
        half=half,
    )
    warmup_frames = frames[:warmup_count]
    measured_frames = frames[warmup_count:]

    for frame in warmup_frames:
        tracker.update(frame.detections, image, embs=frame.embeddings)

    started = time.perf_counter()
    for frame in measured_frames:
        tracker.update(frame.detections, image, embs=frame.embeddings)
    elapsed = time.perf_counter() - started

    fps = measured_count / elapsed if elapsed > 0 else float("inf")
    if not uses_reid:
        reid_backend = "—"
        effective_reid_mode = "none"
    elif reid_mode == "precomputed":
        reid_backend = f"precomputed-{DEFAULT_EMBEDDING_DIM}d"
        effective_reid_mode = reid_mode
    else:
        reid_backend = _resolve_live_reid_label(
            tracker,
            tracker_name,
            backend,
            reid_weights,
            device,
        )
        effective_reid_mode = reid_mode

    return {
        "tracker": tracker_name,
        "backend": backend,
        "reid_mode": effective_reid_mode,
        "reid_backend": reid_backend,
        "reid_weights": str(reid_weights) if uses_reid and reid_mode == "live" else "",
        "n_dets": detection_count,
        "frames": measured_count,
        "elapsed_s": round(elapsed, 4),
        "fps": round(fps, 2),
        "ms_per_frame": round(1000.0 * elapsed / measured_count, 3),
    }


def _parse_csv_values(value: str) -> tuple[str, ...]:
    """Parse a non-empty, normalized comma-separated list."""

    values = tuple(dict.fromkeys(part.strip().lower() for part in value.split(",") if part.strip()))
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def _parse_positive_counts(value: str) -> tuple[int, ...]:
    """Parse a non-empty comma-separated list of positive integers."""

    try:
        counts = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("counts must be comma-separated integers") from exc
    if not counts:
        raise argparse.ArgumentTypeError("at least one detection count is required")
    if any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("detection counts must be positive")
    return counts


def _positive_int(value: str) -> int:
    """Parse a positive integer argument."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    """Parse a non-negative integer argument."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parse_image_size(value: str) -> tuple[int, int]:
    """Parse a positive ``height,width`` image size."""

    try:
        dimensions = tuple(int(part.strip()) for part in value.lower().replace("x", ",").split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("image size must contain integers") from exc
    if len(dimensions) != 2:
        raise argparse.ArgumentTypeError("image size must be height,width")
    if any(dimension <= 0 for dimension in dimensions):
        raise argparse.ArgumentTypeError("image dimensions must be positive")
    return dimensions


def _build_parser() -> argparse.ArgumentParser:
    """Build the standalone benchmark argument parser."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trackers",
        type=_parse_csv_values,
        default=DEFAULT_TRACKERS,
        metavar="LIST",
        help=f"Comma-separated tracker names. Default: {','.join(DEFAULT_TRACKERS)}.",
    )
    parser.add_argument(
        "--backends",
        type=_parse_csv_values,
        default=DEFAULT_BACKENDS,
        metavar="LIST",
        help="Comma-separated backends: python,cpp. Default: python,cpp.",
    )
    parser.add_argument(
        "--counts",
        type=_parse_positive_counts,
        default=DEFAULT_COUNTS,
        metavar="LIST",
        help=f"Positive detection counts. Default: {','.join(str(count) for count in DEFAULT_COUNTS)}.",
    )
    parser.add_argument(
        "--frames",
        type=_positive_int,
        default=DEFAULT_FRAMES,
        help=f"Measured frames per setting. Default: {DEFAULT_FRAMES}.",
    )
    parser.add_argument(
        "--warmup",
        type=_non_negative_int,
        default=DEFAULT_WARMUP,
        help=f"Warmup frames per setting. Default: {DEFAULT_WARMUP}.",
    )
    parser.add_argument(
        "--img-size",
        type=_parse_image_size,
        default=DEFAULT_IMG_HW,
        metavar="H,W",
        help=f"Synthetic image height,width. Default: {DEFAULT_IMG_HW[0]},{DEFAULT_IMG_HW[1]}.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reid-mode",
        choices=("precomputed", "live"),
        default="precomputed",
        help="Use precomputed embeddings or include live ReID inference. Default: precomputed.",
    )
    parser.add_argument(
        "--reid",
        type=Path,
        default=Path(DEFAULT_REID_WEIGHTS),
        help=f"ReID weights used in live mode. Default: {DEFAULT_REID_WEIGHTS}.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used for live ReID inference: cpu, cuda, or mps. Default: cpu.",
    )
    parser.add_argument("--half", action="store_true", help="Use FP16 for live ReID where supported.")
    parser.add_argument("--json", dest="json_path", type=Path, help="Optional JSON result path.")
    parser.add_argument("--csv", dest="csv_path", type=Path, help="Optional CSV result path.")
    parser.add_argument(
        "--skip-missing-cpp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip trackers without a native backend; use --no-skip-missing-cpp to record errors.",
    )
    return parser


def _validate_selection(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate tracker and backend selections after parsing."""

    unknown_trackers = sorted(set(args.trackers) - set(TRACKER_MAPPING))
    if unknown_trackers:
        parser.error(f"unknown tracker(s): {', '.join(unknown_trackers)}")
    invalid_backends = sorted(set(args.backends) - set(DEFAULT_BACKENDS))
    if invalid_backends:
        parser.error(f"unknown backend(s): {', '.join(invalid_backends)}; choose from python, cpp")


def _error_row(
    *,
    tracker_name: str,
    backend: str,
    detection_count: int,
    reid_mode: ReIDMode,
    reid_weights: str | Path,
    error: Exception,
) -> ResultRow:
    """Build a serializable result row for one failed benchmark case."""

    uses_reid = tracker_name in REID_TRACKER_NAMES
    return {
        "tracker": tracker_name,
        "backend": backend,
        "reid_mode": reid_mode if uses_reid else "none",
        "reid_backend": "—",
        "reid_weights": str(reid_weights) if uses_reid and reid_mode == "live" else "",
        "n_dets": detection_count,
        "frames": 0,
        "elapsed_s": 0.0,
        "fps": 0.0,
        "ms_per_frame": 0.0,
        "error": f"{error.__class__.__name__}: {error}",
    }


def _run_sweep(args: argparse.Namespace) -> tuple[list[ResultRow], bool]:
    """Run every selected case while retaining failures in the result set."""

    rows: list[ResultRow] = []
    had_errors = False
    native_trackers = set(NATIVE_TRACKERS)
    for tracker_name in args.trackers:
        for backend in args.backends:
            if backend == "cpp" and tracker_name not in native_trackers and args.skip_missing_cpp:
                LOGGER.info(f"Skipping {tracker_name}/cpp (no native backend).")
                continue
            for detection_count in args.counts:
                LOGGER.info(
                    f"Benchmarking tracker={tracker_name} backend={backend} "
                    f"n_dets={detection_count} reid_mode={args.reid_mode} ..."
                )
                try:
                    result = _measure(
                        tracker_name=tracker_name,
                        backend=backend,
                        detection_count=detection_count,
                        warmup_count=args.warmup,
                        measured_count=args.frames,
                        image_hw=args.img_size,
                        seed=args.seed + detection_count,
                        reid_mode=args.reid_mode,
                        reid_weights=args.reid,
                        device=args.device,
                        half=args.half,
                    )
                except Exception as exc:
                    had_errors = True
                    LOGGER.error(f"  -> failed: {exc.__class__.__name__}: {exc}")
                    rows.append(
                        _error_row(
                            tracker_name=tracker_name,
                            backend=backend,
                            detection_count=detection_count,
                            reid_mode=args.reid_mode,
                            reid_weights=args.reid,
                            error=exc,
                        )
                    )
                    continue
                LOGGER.info(f"  -> {result['fps']:.2f} FPS ({result['ms_per_frame']:.3f} ms/frame)")
                rows.append(result)
    return rows, had_errors


def _print_table(rows: list[ResultRow], reid_mode: ReIDMode) -> None:
    """Render benchmark results with Rich, imported only for presentation."""

    from rich.console import Console
    from rich.table import Table

    table = Table(title=f"BoxMOT tracker FPS (synthetic detections; ReID mode: {reid_mode})")
    table.add_column("tracker", style="bold")
    table.add_column("backend")
    table.add_column("ReID backend")
    table.add_column("detections", justify="right")
    table.add_column("frames", justify="right")
    table.add_column("elapsed (s)", justify="right")
    table.add_column("FPS", justify="right", style="green")
    table.add_column("ms/frame", justify="right")
    table.add_column("status")
    for row in rows:
        error = row.get("error", "")
        table.add_row(
            str(row["tracker"]),
            str(row["backend"]),
            str(row["reid_backend"]),
            str(row["n_dets"]),
            str(row["frames"]),
            f"{float(row['elapsed_s']):.3f}",
            f"{float(row['fps']):.2f}",
            f"{float(row['ms_per_frame']):.3f}",
            "error" if error else "ok",
        )
    Console().print(table)


def _write_json(path: Path, rows: list[ResultRow]) -> None:
    """Write benchmark rows as UTF-8 JSON, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(f"Wrote JSON results to {path}")


def _write_csv(path: Path, rows: list[ResultRow]) -> None:
    """Write benchmark rows as UTF-8 CSV, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    LOGGER.info(f"Wrote CSV results to {path}")


def main(argv: list[str] | None = None) -> int:
    """Run the tracker benchmark CLI and return a process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_selection(parser, args)

    rows, had_errors = _run_sweep(args)
    _print_table(rows, args.reid_mode)
    if args.json_path is not None:
        _write_json(args.json_path, rows)
    if args.csv_path is not None:
        _write_csv(args.csv_path, rows)

    if not rows:
        LOGGER.error("No benchmark cases were run.")
        return 1
    return 1 if had_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
