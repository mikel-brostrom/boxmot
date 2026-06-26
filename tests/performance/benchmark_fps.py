"""FPS benchmark across trackers, backends, and detection counts.

Measures pure ``tracker.update(...)`` throughput (no detector, no ReID inference)
by feeding synthetic detections per frame. ReID-using trackers (botsort,
occluboost) are fed pre-computed random embeddings so the benchmark isolates
the tracker's CPU cost from any detector / ReID model cost.

Usage
-----

Run with default sweep (Python + C++ backends, all native-supported trackers,
detection counts 10/50/100/200/500, 200 warmup + 200 measured frames each)::

    uv run python -m tests.performance.benchmark_fps

Restrict to a subset::

    uv run python -m tests.performance.benchmark_fps \
        --trackers botsort,ocsort --backends cpp --counts 50,200,500

Save results as JSON / CSV::

    uv run python -m tests.performance.benchmark_fps --json results.json --csv results.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from boxmot.trackers.registry import REID_TRACKERS, create_tracker
from boxmot.utils import logger as LOGGER

try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover - rich is a hard dep elsewhere
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]


# Trackers that have a registered native (C++) live backend.
NATIVE_TRACKERS = ("botsort", "bytetrack", "occluboost", "ocsort", "sfsort")
DEFAULT_TRACKERS = NATIVE_TRACKERS
DEFAULT_BACKENDS = ("python", "cpp")
DEFAULT_COUNTS = (10, 50, 100)
DEFAULT_WARMUP = 5
DEFAULT_FRAMES = 100
DEFAULT_IMG_HW = (1080, 1920)
DEFAULT_REID_WEIGHTS = "osnet_x0_25_msmt17.pt"


def _make_random_dets(n: int, img_hw: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Synthesize ``n`` AABB detections in MOT format ``(x1, y1, x2, y2, conf, cls)``.

    Boxes have small per-frame jitter so association is non-trivial but
    deterministic across frames within a single benchmark.
    """
    h, w = img_hw
    if n <= 0:
        return np.zeros((0, 6), dtype=np.float32)
    cx = rng.uniform(80, w - 80, size=n)
    cy = rng.uniform(80, h - 80, size=n)
    bw = rng.uniform(40, 100, size=n)
    bh = rng.uniform(80, 200, size=n)
    x1 = np.clip(cx - bw / 2, 0, w - 1)
    y1 = np.clip(cy - bh / 2, 0, h - 1)
    x2 = np.clip(cx + bw / 2, 1, w)
    y2 = np.clip(cy + bh / 2, 1, h)
    conf = rng.uniform(0.55, 0.95, size=n)
    cls = np.zeros(n, dtype=np.float32)
    return np.stack([x1, y1, x2, y2, conf, cls], axis=1).astype(np.float32)


def _jitter_dets(dets: np.ndarray, rng: np.random.Generator, img_hw: tuple[int, int]) -> np.ndarray:
    """Apply small motion to existing dets so trackers actually do work."""
    if dets.shape[0] == 0:
        return dets
    h, w = img_hw
    out = dets.copy()
    dx = rng.normal(0.0, 4.0, size=dets.shape[0]).astype(np.float32)
    dy = rng.normal(0.0, 4.0, size=dets.shape[0]).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0] + dx, 0, w - 1)
    out[:, 2] = np.clip(out[:, 2] + dx, 1, w)
    out[:, 1] = np.clip(out[:, 1] + dy, 0, h - 1)
    out[:, 3] = np.clip(out[:, 3] + dy, 1, h)
    return out


def _build_tracker(tracker_name: str, backend: str, *, reid_weights, device: str, half: bool):
    """Construct a tracker. For ReID-using trackers, real ReID weights are
    passed so the tracker computes embeddings internally on every frame and
    that cost is part of the measured ``update`` time.

    Non-ReID trackers ignore ``reid_weights`` (the wrapper kwarg is dropped
    upstream by ``create_tracker``).
    """
    return create_tracker(
        tracker_type=tracker_name,
        reid_weights=reid_weights if tracker_name in REID_TRACKERS else None,
        device=device,
        half=half,
        per_class=False,
        tracker_backend=backend,
    )


_PROVIDER_TO_DEVICE = {
    "CUDAExecutionProvider": "cuda",
    "TensorrtExecutionProvider": "cuda",
    "CoreMLExecutionProvider": "coreml",
    "CPUExecutionProvider": "cpu",
}


def _resolve_reid_label(tracker, tracker_name: str, backend: str, reid_weights, device: str) -> str:
    """Best-effort label for the ReID runtime actually used by the tracker.

    Examples: ``pt-cuda``, ``onnx-coreml``, ``onnx-cpu``, ``onnx-cuda``.
    Returns ``"—"`` for non-ReID trackers.
    """
    if tracker_name not in REID_TRACKERS:
        return "—"

    if backend == "cpp":
        env_be = os.environ.get("BOXMOT_REID_BACKEND", "").lower()
        fmt = "opencv" if env_be in {"opencv", "dnn"} else "onnx"
        env_dev = os.environ.get("BOXMOT_REID_DEVICE", "").lower()
        if env_dev in {"cpu", "cuda", "gpu", "coreml", "mps", "metal"}:
            dev = {"gpu": "cuda", "mps": "coreml", "metal": "coreml"}.get(env_dev, env_dev)
        elif sys.platform == "darwin":
            dev = "coreml"
        elif sys.platform.startswith("linux") or sys.platform == "win32":
            dev = "cuda"  # ORT falls back to cpu if EP unavailable
        else:
            dev = "cpu"
        return f"{fmt}-{dev}"

    # Python path: introspect the live ReID model.
    model = getattr(tracker, "model", None)
    if model is None:
        suffix = Path(str(reid_weights)).suffix.lower().lstrip(".") or "pt"
        return f"{suffix}-{device}"
    cls = model.__class__.__name__
    fmt_map = {
        "PyTorchBackend": "pt",
        "ONNXBackend": "onnx",
        "TorchscriptBackend": "torchscript",
        "TensorRTBackend": "tensorrt",
        "OpenVinoBackend": "openvino",
        "TFLiteBackend": "tflite",
    }
    fmt = fmt_map.get(cls, cls.replace("Backend", "").lower())
    if cls == "ONNXBackend":
        session = getattr(model, "session", None)
        providers = list(session.get_providers()) if session is not None else []
        dev = next((_PROVIDER_TO_DEVICE[p] for p in providers if p in _PROVIDER_TO_DEVICE), device)
    else:
        dev_obj = getattr(model, "device", device)
        dev = str(getattr(dev_obj, "type", dev_obj))
    return f"{fmt}-{dev}"


def _measure(
    tracker_name: str,
    backend: str,
    n_dets: int,
    *,
    n_warmup: int,
    n_frames: int,
    img_hw: tuple[int, int],
    seed: int,
    reid_weights,
    device: str,
    half: bool,
) -> dict:
    rng = np.random.default_rng(seed)
    # Use a non-trivial random image so ReID crops contain real content.
    img = rng.integers(0, 255, size=(img_hw[0], img_hw[1], 3), dtype=np.uint8)

    tracker = _build_tracker(
        tracker_name, backend, reid_weights=reid_weights, device=device, half=half
    )

    base_dets = _make_random_dets(n_dets, img_hw, rng)

    def step():
        dets = _jitter_dets(base_dets, rng, img_hw)
        tracker.update(dets, img)

    # Warmup
    for _ in range(n_warmup):
        step()

    # Measured run
    t0 = time.perf_counter()
    for _ in range(n_frames):
        step()
    elapsed = time.perf_counter() - t0
    fps = n_frames / elapsed if elapsed > 0 else float("inf")
    uses_reid = tracker_name in REID_TRACKERS
    reid_label = _resolve_reid_label(tracker, tracker_name, backend, reid_weights, device)
    return {
        "tracker": tracker_name,
        "backend": backend,
        "n_dets": n_dets,
        "frames": n_frames,
        "elapsed_s": round(elapsed, 4),
        "fps": round(fps, 2),
        "ms_per_frame": round(1000.0 * elapsed / max(1, n_frames), 3),
        "reid_weights": str(reid_weights) if uses_reid else "",
        "reid_backend": reid_label,
    }


def _parse_csv(text: str | None, default: Iterable):
    if text is None or text == "":
        return list(default)
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_int_csv(text: str | None, default: Iterable[int]) -> list[int]:
    if text is None or text == "":
        return list(default)
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _print_table(rows: list[dict]) -> None:
    if Console is None or Table is None:
        # Fallback plain text
        header = f"{'tracker':<12} {'backend':<8} {'reid':<14} {'n_dets':>7} {'fps':>10} {'ms/frame':>10}"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row['tracker']:<12} {row['backend']:<8} {row.get('reid_backend', '—'):<14} {row['n_dets']:>7} "
                f"{row['fps']:>10.2f} {row['ms_per_frame']:>10.3f}"
            )
        return

    console = Console()
    table = Table(title="BoxMOT tracker FPS (synthetic detections; ReID included for ReID trackers)")
    table.add_column("tracker", style="bold")
    table.add_column("backend")
    table.add_column("reid backend")
    table.add_column("n_dets", justify="right")
    table.add_column("frames", justify="right")
    table.add_column("elapsed (s)", justify="right")
    table.add_column("FPS", justify="right", style="green")
    table.add_column("ms/frame", justify="right")
    for row in rows:
        table.add_row(
            row["tracker"],
            row["backend"],
            row.get("reid_backend", "—"),
            str(row["n_dets"]),
            str(row["frames"]),
            f"{row['elapsed_s']:.3f}",
            f"{row['fps']:.2f}",
            f"{row['ms_per_frame']:.3f}",
        )
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trackers",
        type=str,
        default=",".join(DEFAULT_TRACKERS),
        help=f"Comma-separated tracker names. Default: {','.join(DEFAULT_TRACKERS)}",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default=",".join(DEFAULT_BACKENDS),
        help="Comma-separated backends to benchmark. Choices: python, cpp.",
    )
    parser.add_argument(
        "--counts",
        type=str,
        default=",".join(str(c) for c in DEFAULT_COUNTS),
        help=f"Comma-separated detection counts per frame. Default: {','.join(str(c) for c in DEFAULT_COUNTS)}",
    )
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="Measured frames per setting.")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup frames per setting.")
    parser.add_argument(
        "--img-size",
        type=str,
        default=f"{DEFAULT_IMG_HW[0]},{DEFAULT_IMG_HW[1]}",
        help="Image (height,width) used to seed detection coordinates.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reid",
        type=str,
        default=DEFAULT_REID_WEIGHTS,
        help=(
            "ReID weights for ReID-using trackers (timing includes the ReID forward pass). "
            f"Default: {DEFAULT_REID_WEIGHTS} (auto-downloaded)."
        ),
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for ReID inference (cpu/cuda/mps).")
    parser.add_argument("--half", action="store_true", help="Use FP16 for ReID inference where supported.")
    parser.add_argument("--json", type=str, default=None, help="Optional path to save JSON results.")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to save CSV results.")
    parser.add_argument(
        "--skip-missing-cpp",
        action="store_true",
        default=True,
        help="Skip (instead of failing) trackers without a native C++ backend.",
    )
    args = parser.parse_args(argv)

    trackers = _parse_csv(args.trackers, DEFAULT_TRACKERS)
    backends = _parse_csv(args.backends, DEFAULT_BACKENDS)
    counts = _parse_int_csv(args.counts, DEFAULT_COUNTS)
    img_hw_parts = [int(x) for x in args.img_size.split(",")]
    if len(img_hw_parts) != 2:
        parser.error("--img-size must be 'height,width'")
    img_hw = (img_hw_parts[0], img_hw_parts[1])

    invalid_backends = [b for b in backends if b not in ("python", "cpp")]
    if invalid_backends:
        parser.error(f"Invalid backend(s): {invalid_backends}. Expected: python, cpp")

    rows: list[dict] = []
    for tracker_name in trackers:
        for backend in backends:
            if backend == "cpp" and tracker_name not in NATIVE_TRACKERS:
                if args.skip_missing_cpp:
                    LOGGER.info(f"Skipping {tracker_name}/cpp (no native backend).")
                    continue
                raise SystemExit(f"No native (cpp) backend for tracker '{tracker_name}'")
            for n_dets in counts:
                LOGGER.info(
                    f"Benchmarking tracker={tracker_name} backend={backend} n_dets={n_dets} ..."
                )
                try:
                    result = _measure(
                        tracker_name=tracker_name,
                        backend=backend,
                        n_dets=n_dets,
                        n_warmup=args.warmup,
                        n_frames=args.frames,
                        img_hw=img_hw,
                        seed=args.seed + n_dets,
                        reid_weights=args.reid,
                        device=args.device,
                        half=args.half,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error(
                        f"  -> failed: {exc.__class__.__name__}: {exc}"
                    )
                    rows.append(
                        {
                            "tracker": tracker_name,
                            "backend": backend,
                            "n_dets": n_dets,
                            "frames": 0,
                            "elapsed_s": 0.0,
                            "fps": 0.0,
                            "ms_per_frame": 0.0,
                            "reid_weights": str(args.reid) if tracker_name in REID_TRACKERS else "",
                            "reid_backend": "—",
                            "error": f"{exc.__class__.__name__}: {exc}",
                        }
                    )
                    continue
                LOGGER.info(
                    f"  -> {result['fps']:.2f} FPS ({result['ms_per_frame']:.3f} ms/frame)"
                )
                rows.append(result)

    _print_table(rows)

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        LOGGER.info(f"Wrote JSON results to {args.json}")
    if args.csv:
        import csv

        keys = ["tracker", "backend", "reid_backend", "reid_weights", "n_dets", "frames", "elapsed_s", "fps", "ms_per_frame", "error"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in keys})
        LOGGER.info(f"Wrote CSV results to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
