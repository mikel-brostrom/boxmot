# Installation

## Requirements

BoxMOT supports Python `3.10` through `3.13`.

## Basic install

```bash
pip install boxmot
boxmot --help
```

This installs the CLI, Python API, tracker implementations, and core ReID stack
using the standard PyPI PyTorch build. Detector-backed workflows should also
install the matching detector extra. Source checkouts can choose a specific
PyTorch build as described below.

## Select a PyTorch profile

Source checkouts use mutually exclusive, lockfile-backed `cpu` and `cu130`
extras. Choose exactly one:

```bash
# CPU-only PyTorch
uv sync --extra cpu

# CUDA 13.0 PyTorch
uv sync --extra cu130
```

For a package installation, uv's pip interface can select the corresponding
PyTorch index explicitly:

```bash
uv venv
uv pip install "boxmot[cpu]" --torch-backend=cpu
# Or: uv pip install "boxmot[cu130]" --torch-backend=cu130
```

The index mappings in `tool.uv.sources` are uv project configuration and are
not honored by standard `pip`, which continues to use PyPI by default. Also,
uv does not remember a selected project extra: repeat `--extra cpu` or
`--extra cu130` on later `uv sync` commands. After syncing, use
`.venv/bin/<command>` or `uv run --no-sync <command>` to avoid unintentionally
replacing the selected PyTorch build with the default one.

## Mode-specific extras

BoxMOT keeps heavier workflow dependencies optional. The source-checkout
examples below use the CPU profile; replace `cpu` with `cu130` for CUDA 13.0.
For package installs, first select a PyTorch profile above, then add the listed
feature extras to the same environment.

| Workflow | PyPI install | Source checkout with `uv` | Notes |
| --- | --- | --- | --- |
| Tracking workflows with common YOLO backends | `pip install "boxmot[yolo]"` | `uv sync --extra cpu --extra yolo` | Preinstalls Ultralytics and YOLOX. |
| Detector inference with RT-DETR v2 | `pip install "boxmot[rtdetr]"` | `uv sync --extra cpu --extra rtdetr` | Installs the Transformers detector backend. |
| `train-reid`, `eval-reid`, and `compare-reid` | No additional extra | `uv sync --extra cpu` | Uses the built-in ReID stack with the selected PyTorch profile. Place each selected ReID dataset under its configured `--data-dir` or `--target` path. |
| `tune` | `pip install "boxmot[evolve]"` | `uv sync --extra cpu --extra evolve` | Installs Ray Tune, Optuna, Plotly, and related tuning dependencies. |
| `research` | `pip install "boxmot[research]"` | `uv sync --extra cpu --extra research` | Installs GEPA for the code-evolution loop. |
| Detection-to-track HTTP service | `pip install "boxmot[service]"` | `uv sync --extra cpu --extra service` | Installs FastAPI and Uvicorn for `boxmot-service`. |
| `eval --compare-trackeval` | `pip install "boxmot[trackeval]"` | `uv sync --extra cpu --extra trackeval` | Adds the TrackEval reference comparison for AABB MOTChallenge datasets. |
| `export --include onnx` | `pip install "boxmot[onnx]"` | `uv sync --extra cpu --extra onnx` | The default export path uses ONNX. |
| `export --include coreml` | `pip install "boxmot[coreml]"` | `uv sync --extra cpu --extra coreml` | Native FP16 MLProgram export and inference on macOS. |
| `export --include openvino` | `pip install "boxmot[openvino]"` | `uv sync --extra cpu --extra openvino` | Usually paired with `--include onnx`. |
| `export --include tflite` | `pip install "boxmot[tflite]"` | `uv sync --extra cpu --extra tflite` | Installs both TFLite export and LiteRT inference packages. |

You can combine extras when needed:

```bash
uv sync --extra cpu --extra yolo --extra evolve --extra research
pip install "boxmot[yolo,evolve,research]"
```

When an optional ReID runtime is missing, BoxMOT attempts a first-use install with `uv pip install` when `uv` is available, otherwise with the active `python -m pip`. This covers ONNX Runtime, Core ML, OpenVINO, LiteRT, and NVIDIA TensorRT. Native Core ML requires macOS; TensorRT still requires a compatible CUDA/NVIDIA stack for the installed wheel to import and run correctly.

## Docker

The shared Dockerfile provides four production targets. Both CLI images
include the `yolo` and `trackeval` extras for detector, evaluation, and
interactive workflows:

| Workload | Build target | Published image | Runtime |
| --- | --- | --- | --- |
| Full CLI | `cli-gpu` | `boxmot/boxmot:latest` | NVIDIA GPU |
| Full CLI | `cli-cpu` | `boxmot/boxmot:latest-cpu` | CPU |
| Geometry-only tracker service | `service-cpu` | `boxmot/boxmot-service:latest` | CPU only |
| ReID tracker service | `service-gpu` | `boxmot/boxmot-service:latest-gpu` | NVIDIA GPU |

Run the published GPU image with the NVIDIA Container Toolkit and a compatible
host driver:

```bash
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" \
  --workdir /workspace \
  boxmot/boxmot:latest
```

Use the CPU-suffixed image on hosts without NVIDIA GPUs:

```bash
docker run --rm -it \
  -v "$PWD:/workspace" \
  --workdir /workspace \
  boxmot/boxmot:latest-cpu
```

Inside either container, the project virtual environment is already on `PATH`;
verify it with `boxmot --help`. Other optional workflows, such as model export
or tuning, require their corresponding extras and are not included in the CLI
images.

To build the same variants locally from the repository root:

```bash
docker build --target cli-gpu -f docker/Dockerfile -t boxmot/boxmot:local .
docker build --target cli-cpu -f docker/Dockerfile -t boxmot/boxmot:local-cpu .
docker build --target service-cpu -f docker/Dockerfile -t boxmot/boxmot-service:local .
docker build --target service-gpu -f docker/Dockerfile -t boxmot/boxmot-service:local-gpu .
```

Published GPU CLI tags are `latest`, `<version>`, and `sha-<commit>`. Published
CPU CLI tags append `-cpu`, for example `<version>-cpu` and
`sha-<commit>-cpu`. The CPU service publishes canonical tags in the separate
`boxmot/boxmot-service` repository; its GPU counterpart appends `-gpu`, for
example `<version>-gpu` and `sha-<commit>-gpu`.

Run the HTTP image when detections come from a separate detector:

```bash
docker run --rm -p 8000:8000 boxmot/boxmot-service:latest
```

This CPU image supports ByteTrack, OCSort, and SFSORT without image pixels. Run
the CUDA/ReID image with a mounted checkpoint:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/models/osnet_x0_25_msmt17.pt:/models/osnet_x0_25_msmt17.pt:ro" \
  -e BOXMOT_SERVICE_REID_WEIGHTS=/models/osnet_x0_25_msmt17.pt \
  boxmot/boxmot-service:latest-gpu
```

The GPU image defaults to BotSORT and also supports StrongSORT, DeepOCSORT,
HybridSORT, BoostTrack, and OccluBoost. It requires a raw base64-encoded JPEG or
PNG in `image_base64` on every frame, including frames without detections. Both
service images run as a non-root user and expose health checks, OpenAPI at
`/docs`, and a stateful frame endpoint. They consume external detections and do
not run detector inference. See
[Tracker service deployment](../guides/deployment.md) for the request schema,
payload guidance, and scaling model.

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
