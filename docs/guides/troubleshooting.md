# Troubleshooting and FAQ

Common problems and their resolutions when working with BoxMOT.

## Installation

### `boxmot --help` does nothing useful after install

The core `pip install boxmot` is enough for the Python API but not for many CLI workflows. Install the extra that matches the mode you want to use:

```bash
boxmot install --extra yolo                   # track / materialize with YOLO backends
boxmot install --extra evolve                 # tune with Ray
boxmot install --extra research               # research
boxmot install --extra onnx                   # export --include onnx
boxmot install --extra coreml                 # native Core ML MLProgram export/inference
boxmot install --extra onnx --extra openvino  # export --include openvino
boxmot install --extra tflite                 # export --include tflite and LiteRT inference
```

Dependency checks report missing packages without installing them. If your
application uses a specific Python environment, run
`python -m boxmot.engine.cli install --extra NAME` with that interpreter.
See [Installation](../getting-started/installation.md#mode-specific-extras)
for the full table.

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

### CUDA device index is unavailable

Device indices follow PyTorch's process-visible GPU order. If you expose only
GPU 2 before starting BoxMOT, select it as `cuda:0` inside the process:

```bash
CUDA_VISIBLE_DEVICES=2 boxmot track --source video.mp4 --device cuda:0
```

In that process, `cuda:2` is out of range because only one GPU is visible.
BoxMOT does not change `CUDA_VISIBLE_DEVICES` during device selection. See
[Device selection](../modes/track.md#device-selection) for accepted selectors.

### TensorRT ReID rejects CPU

TensorRT ReID inference requires a CUDA device. Set `--device cuda:0` in the
CLI or `device="cuda:0"` on the ReID specification, adjusting the logical index
for the GPU you want to use. An explicit CPU selection raises an error.

### TensorRT is installed but import still fails

Install TensorRT explicitly in the application environment:

```bash
python -m boxmot.engine.cli install \
  --requirement 'nvidia-tensorrt' \
  --extra-index-url https://pypi.ngc.nvidia.com
```

If `import tensorrt` still fails, check that Python, CUDA, the NVIDIA driver,
and the installed TensorRT wheel are compatible. An installed package's
metadata can satisfy a dependency check while its native runtime cannot load.
For export, also install the ONNX extra as shown in
[TensorRT setup](../modes/install.md#tensorrt).

## OBB tracking

### Tracker geometry mismatch errors

Construct canonical detections with `OrientedBoxes` for OBB geometry or `Boxes`
for AABB geometry. The tracker's geometry mode is fixed by `TrackerSpec`; the
inherited `BaseTracker.update()` rejects a `Detections` object with a different
geometry mode. For standalone box-only calls, the same fixed mode selects the
required NumPy input shape: exact AABB `N x 6` or OBB `N x 7` rows. Use
`Detections` when passing embeddings or masks, controlling sample identity
or composing a pipeline. Plain NumPy input returns `float64` AABB `M x 8` or
OBB `M x 9` rows; `Detections` input returns `Tracks`.

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
