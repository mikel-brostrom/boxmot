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
pip install "boxmot[coreml]"      # native Core ML MLProgram export/inference
pip install "boxmot[openvino]"    # export --include openvino
pip install "boxmot[tflite]"      # export --include tflite and LiteRT inference
```

See [Installation](../getting-started/installation.md#mode-specific-extras) for the full table.

### ONNX on MPS consumes excessive memory

ONNX Runtime does not provide an MPS execution provider. BoxMOT therefore runs
`.onnx` weights on CPU on macOS, even when `device="mps"` is requested. Export
the checkpoint with `--include coreml` and use the resulting
`*_coreml_model/` directory for Apple GPU/CPU execution.

The legacy ONNX Runtime Core ML execution provider is disabled by default
because transformer graph compilation can consume extreme RAM. It can be
enabled only for controlled diagnostics with
`BOXMOT_ENABLE_LEGACY_ONNX_COREML=1`; the native MLProgram path is recommended.

### `ModuleNotFoundError: boxmot` when running a script

Run BoxMOT entry points as modules from the repo root, not as loose scripts:

```bash
# Good
uv run python -m boxmot.engine.cli --help

# Avoid
python boxmot/engine/cli.py --help
```

## Python compatibility

BoxMOT supports Python 3.10 through 3.13.

## ReID and acceleration

### macOS: ReID feels slow or runs on CPU

The ONNX ReID backend intentionally selects `CPUExecutionProvider` on macOS,
even when `device="mps"` is requested. Changing the ONNX Runtime wheel does
not make ONNX use MPS. For Apple GPU/CPU acceleration, export native Core ML
and pass the resulting `*_coreml_model/` directory as the ReID weights:

```bash
boxmot export --weights model.pt --include coreml --device cpu
```

For PyTorch weights, use `device="mps"` directly. The legacy ONNX Core ML
provider remains available only through the diagnostic environment variable
described above.

### CUDA: detector or ReID falls back to CPU

Confirm both PyTorch and `onnxruntime-gpu` see the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

`CUDAExecutionProvider` must appear in the second output for the ONNX ReID backend to pick it up.

### TensorRT auto-install succeeds but import still fails

The TensorRT ReID backend and `export --include engine` try to install `nvidia-tensorrt` on first use, including NVIDIA's Python package index. If `import tensorrt` still fails afterward, check that your Python, CUDA, NVIDIA driver, and TensorRT wheel versions are compatible for the machine.

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

## Experiment workflows

### `eval` re-runs detection every time

`generate`, `eval`, `tune`, and `research` share detection and embedding
caches, but the key is more specific than detector + ReID + dataset. The root
includes benchmark, split, and detector or public-detection producer.
Embeddings additionally include their Python/C++ producer, ReID format and
runtime, weights fingerprint, preprocessing policy, and crop-schema version.
Keep those inputs and overrides identical across modes to reuse the same
artifacts.

### Replay is slow on trackers that use camera motion compensation

Most replay runs skip image loading completely, but trackers that need live image data during replay still have to read frames from the dataset.

### Tuning doesn't explore parameters you expect

Tuning ranges live alongside runtime defaults in
`boxmot/configs/trackers/<tracker>.yaml`. Runtime construction extracts each
parameter's `default`; the tuner reads its search metadata.

## Reporting a problem

If none of the above helps, open an issue on [GitHub](https://github.com/mikel-brostrom/boxmot/issues) with:

- the exact command you ran
- the BoxMOT version (`pip show boxmot`)
- Python, OS, and CUDA / ONNX Runtime versions
- the full stack trace
