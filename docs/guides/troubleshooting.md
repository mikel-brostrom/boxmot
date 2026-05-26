# Troubleshooting and FAQ

Common problems and their resolutions when working with BoxMOT.

## Installation

### `boxmot --help` does nothing useful after install

The core `pip install boxmot` is enough for the Python API but not for many CLI workflows. Install the extra that matches the mode you want to use:

```bash
pip install "boxmot[yolo]"        # track / generate / eval with YOLO backends
pip install "boxmot[evolve]"      # tune
pip install "boxmot[research]"    # research
pip install "boxmot[onnx]"        # export --include onnx
```

See [Installation](../getting-started/installation.md#mode-specific-extras) for the full table.

### `ModuleNotFoundError: boxmot` when running a script

Run BoxMOT entry points as modules from the repo root, not as loose scripts:

```bash
# Good
uv run python -m boxmot.engine.cli --help

# Avoid
python boxmot/engine/cli.py --help
```

## Python compatibility

BoxMOT requires Python `3.10` or newer (up to `3.13`). Use `@dataclass(..., slots=True)` directly in all dataclass definitions.

## ReID and acceleration

### macOS: ReID feels slow or runs on CPU

The ONNX ReID backend selects providers from `onnxruntime.get_available_providers()`. On macOS it prefers `CoreMLExecutionProvider`. If only `CPUExecutionProvider` is available, install a runtime that ships CoreML support:

```bash
pip install onnxruntime          # or
pip install onnxruntime-silicon  # Apple Silicon optimized
```

The OpenCV-DNN ReID variant (e.g. `osnet_x0_25_msmt17_opencv.onnx`) can be faster on macOS when OpenCL is enabled in OpenCV.

### CUDA: detector or ReID falls back to CPU

Confirm both PyTorch and `onnxruntime-gpu` see the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

`CUDAExecutionProvider` must appear in the second output for the ONNX ReID backend to pick it up.

## OBB tracking

### "Detections must have 7 columns for OBB" or shape errors

OBB detections must be `(cx, cy, w, h, angle, conf, cls)` (7 columns); AABB detections are `(x1, y1, x2, y2, conf, cls)` (6 columns). Trackers infer the mode from the column count via `BaseTracker.setup_decorator`. Make sure your detector emits the correct shape and that the tracker has `supports_obb = True`.

### Track angle "snaps" or flips between frames

When extending OBB support, prefer damping over hard-resetting angular velocity each update, and resolve equivalent rectangle forms `(w, h, theta)`, `(w, h, theta + pi)`, `(h, w, theta ± pi/2)` to the candidate closest to the current state. See [Add OBB Support](../contributing/obb-support.md).

## Native C++ trackers

### `--tracker-backend cpp` fails to build on first use

Native backends compile on first use. Make sure these are installed:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

Native backends are currently available for `botsort`, `bytetrack`, `ocsort`, `occluboost`, and `sfsort`.

## Benchmark workflows

### `eval` re-runs detection every time

`generate`, `eval`, `tune`, and `research` share a cache keyed by detector + ReID + dataset. If you change benchmark, dataset, detector, or ReID, the cache key changes and the run regenerates. Keep the same combination across modes to reuse cached detections and embeddings.

### Tuning doesn't explore parameters you expect

Tuning ranges live in the per-tracker YAMLs under `boxmot/configs/trackers/`. Each tracker exposes only the parameters listed there to `boxmot tune`.

## Reporting a problem

If none of the above helps, open an issue on [GitHub](https://github.com/mikel-brostrom/boxmot/issues) with:

- the exact command you ran
- the BoxMOT version (`pip show boxmot`)
- Python, OS, and CUDA / ONNX Runtime versions
- the full stack trace
