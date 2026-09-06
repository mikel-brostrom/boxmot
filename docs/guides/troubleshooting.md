# Troubleshooting and FAQ

Common problems and their resolutions when working with BoxMOT.

## Installation

### `boxmot --help` does nothing useful after install

The core `pip install boxmot` is enough for the Python API but not for many CLI workflows. Install the extra that matches the mode you want to use:

```bash
pip install "boxmot[yolo]"        # track / materialize with YOLO backends
pip install "boxmot[evolve]"      # tune
pip install "boxmot[research]"    # research
pip install "boxmot[onnx]"        # export --include onnx
pip install "boxmot[coreml]"      # native Core ML MLProgram export/inference
pip install "boxmot[openvino]"    # export --include openvino
pip install "boxmot[tflite]"      # export --include tflite and LiteRT inference
```

See [Installation](../getting-started/installation.md#mode-specific-extras) for the full table.

### ONNX does not run on MPS

ONNX Runtime does not provide an MPS execution provider, so `.onnx` weights
reject `device="mps"` instead of silently running on CPU. Export the checkpoint
with `--include coreml` and use the resulting `*_coreml_model/` directory for
Apple GPU/CPU execution, or select `device="cpu"` explicitly.

### `ModuleNotFoundError: boxmot` when running a script

Run BoxMOT entry points as modules from the repo root, not as loose scripts:

```bash
# Good
uv run --no-sync python -m boxmot.engine.cli --help

# Avoid
python boxmot/engine/cli.py --help
```

## Python compatibility

BoxMOT supports Python 3.10 through 3.13.

## ReID and acceleration

### macOS: ReID feels slow or runs on CPU

The ONNX ReID backend rejects `device="mps"`; changing the ONNX Runtime wheel
does not make ONNX use MPS. For Apple GPU/CPU acceleration, export native Core
ML and pass the resulting `*_coreml_model/` directory as the ReID weights:

```bash
boxmot export --weights model.pt --include coreml --device cpu
```

For PyTorch weights, use `device="mps"` directly.

### CUDA: detector or ReID falls back to CPU

Confirm both PyTorch and `onnxruntime-gpu` see the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

`CUDAExecutionProvider` must appear in the second output. An explicit CUDA
request fails during initialization when that provider is unavailable.

### TensorRT auto-install succeeds but import still fails

The TensorRT ReID backend and `export --include engine` try to install `nvidia-tensorrt` on first use, including NVIDIA's Python package index. If `import tensorrt` still fails afterward, check that your Python, CUDA, NVIDIA driver, and TensorRT wheel versions are compatible for the machine.

## OBB tracking

### Tracker geometry mismatch errors

Construct canonical detections with `OrientedBoxes` for OBB geometry or `Boxes`
for AABB geometry. The tracker's geometry mode is fixed by `TrackerSpec`; the
inherited `BaseTracker.update()` accepts a `Detections` object and rejects a
different geometry mode. Raw 6/7-column arrays are supported only by explicit
boundary serializers, not by tracker calls.

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

### `eval` says the build is incompatible

An explicitly selected incompatible build is never modified or silently
replaced. Either omit `--build` so experiment-backed eval materializes a
compatible canonical build, or materialize the same experiment yourself and
pass the result explicitly:

```bash
boxmot eval --experiment EXPERIMENT

boxmot materialize --experiment EXPERIMENT
boxmot eval --experiment EXPERIMENT --build BUILD_ID
```

If the build came from an older direct-source or dataset-only materialization,
materialize it again through an experiment so its manifest contains the
canonical dataset, split, taxonomy, geometry, and component identity.

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
