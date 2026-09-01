# Installation

## Requirements

BoxMOT supports Python `3.10` through `3.13`.

## Basic install

```bash
pip install boxmot
boxmot --help
```

This installs the CLI, Python API, tracker implementations, and core ReID
stack. Detector-backed workflows should also install the matching detector
extra. Some backends can install a missing package on first use, but an explicit
extra gives a repeatable environment.

## Mode-specific extras

BoxMOT keeps heavier workflow dependencies optional. Install the extras that match the modes and export targets you plan to use.

| Workflow | PyPI install | Source checkout with `uv` | Notes |
| --- | --- | --- | --- |
| Tracking workflows with common YOLO backends | `pip install "boxmot[yolo]"` | `uv sync --extra yolo` | Preinstalls Ultralytics and YOLOX. |
| Detector inference with RT-DETR v2 | `pip install "boxmot[rtdetr]"` | `uv sync --extra rtdetr` | Installs the Transformers detector backend. |
| `train-reid`, `eval-reid`, and `compare-reid` | `pip install boxmot` | `uv sync` | Uses the built-in ReID training and evaluation stack. Place each selected ReID dataset under its configured `--data-dir` or `--target` path. |
| `tune` | `pip install "boxmot[evolve]"` | `uv sync --extra evolve` | Installs Ray Tune, Optuna, Plotly, and related tuning dependencies. |
| `research` | `pip install "boxmot[research]"` | `uv sync --extra research` | Installs GEPA for the code-evolution loop. |
| Detection-to-track HTTP service | `pip install "boxmot[service]"` | `uv sync --extra service` | Installs FastAPI and Uvicorn for `boxmot-service`. |
| `eval --compare-trackeval` | `pip install "boxmot[trackeval]"` | `uv sync --extra trackeval` | Adds the TrackEval reference comparison for AABB MOTChallenge datasets. |
| `export --include onnx` | `pip install "boxmot[onnx]"` | `uv sync --extra onnx` | The default export path uses ONNX. |
| `export --include coreml` | `pip install "boxmot[coreml]"` | `uv sync --extra coreml` | Native FP16 MLProgram export and inference on macOS. |
| `export --include openvino` | `pip install "boxmot[openvino]"` | `uv sync --extra openvino` | Usually paired with `--include onnx`. |
| `export --include tflite` | `pip install "boxmot[tflite]"` | `uv sync --extra tflite` | Installs both TFLite export and LiteRT inference packages. |

You can combine extras when needed:

```bash
uv sync --extra yolo --extra evolve --extra research
pip install "boxmot[yolo,evolve,research]"
```

When an optional ReID runtime is missing, BoxMOT attempts a first-use install with `uv pip install` when `uv` is available, otherwise with the active `python -m pip`. This covers ONNX Runtime, Core ML, OpenVINO, LiteRT, and NVIDIA TensorRT. Native Core ML requires macOS; TensorRT still requires a compatible CUDA/NVIDIA stack for the installed wheel to import and run correctly.

## Docker

The shared Dockerfile provides separate `cli` and `service` targets. The CLI
image includes the `yolo` and `trackeval` extras for detector, evaluation, and
interactive workflows:

```bash
docker build --target cli -f docker/Dockerfile -t boxmot/boxmot:local .
```

Run an interactive shell with the project virtual environment already on
`PATH`. Mount a host directory for videos, datasets, and generated results:

```bash
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" \
  --workdir /workspace \
  boxmot/boxmot:local
```

Inside the container, verify the CLI with `boxmot --help`. Omit `--gpus all`
for CPU-only use. GPU use requires the NVIDIA Container Toolkit and a host
driver compatible with the CUDA runtime selected by the locked PyTorch build.
Other optional workflows, such as model export or tuning, require their
corresponding extras and are not included in the default image.

Build and run the HTTP image when detections come from a separate detector:

```bash
docker build --target service -f docker/Dockerfile -t boxmot/boxmot-service:local .
docker run --rm -p 8000:8000 boxmot/boxmot-service:local
```

The service image runs as a non-root user and exposes health checks, OpenAPI at
`/docs`, and a stateful frame endpoint. See [Tracker service deployment](../guides/deployment.md)
for its detection schema and scaling model.

## Native C++ backends

Native C++ tracker backends are built lazily the first time you select `--tracker-backend cpp`. They are currently available for `botsort`, `bytetrack`, `ocsort`, `occluboost`, and `sfsort`.

Install the native build tools before using them:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

Example:

```bash
boxmot track --detector yolov8n --tracker bytetrack --tracker-backend cpp --source video.mp4
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker bytetrack --tracker-backend cpp
```

The generated build files are kept under `build/native/<tracker>/`.
For editable installs or an up-front build, compile native ReID and all live
tracker libraries with `boxmot build`, or select one with
`boxmot build --tracker bytetrack`. Cached replay executables are still built
on first `eval` or `tune` use.

## Verify the install

!!! example "Verify"

    === "CLI"

        Check the CLI:

        ```bash
        boxmot --help
        boxmot track --help
        ```

    === "Python"

        Smoke-test the Python API:

        ```python
        from boxmot import BoxMOT

        boxmot = BoxMOT(detector="yolov8n", reid="osnet_x0_25_msmt17", tracker="bytetrack")
        print(boxmot)
        ```

## Next steps

- Use [Quickstart](../index.md) for a minimal path.
- Use [Modes Overview](../modes/index.md) to decide between `track`, `generate`, `eval`, `tune`, `research`, `train-reid`, `eval-reid`, `compare-reid`, and `export`.
- Use [Native C++ Integration](../native/index.md) for native build and embedding details.
- Use the workflow table above to add the extras your workflow needs.
