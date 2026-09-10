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
feature extras to the same environment with `boxmot install`. This installs
unmet dependencies for the existing BoxMOT distribution.

| Workflow | Add dependencies to installed BoxMOT | Source checkout with `uv` | Notes |
| --- | --- | --- | --- |
| Tracking workflows with common YOLO backends | `boxmot install --extra yolo` | `uv sync --extra cpu --extra yolo` | Installs Ultralytics and YOLOX. |
| Detector inference with RT-DETR v2 | `boxmot install --extra rtdetr` | `uv sync --extra cpu --extra rtdetr` | Installs the Transformers detector backend. |
| `train-reid`, `eval-reid`, and `compare-reid` | No additional extra | `uv sync --extra cpu` | Uses the built-in ReID stack with the selected PyTorch profile. Built-in downloads use `./datasets/reid`; explicit `--data-dir` and `--target` paths remain supported. |
| `tune` with Ray | `boxmot install --extra evolve` | `uv sync --extra cpu --extra evolve` | Installs Ray Tune, Optuna, Plotly, and related tuning dependencies. |
| `research` | `boxmot install --extra research` | `uv sync --extra cpu --extra research` | Installs GEPA for the code-evolution loop. |
| Detection-to-track HTTP service | `boxmot install --extra service` | `uv sync --extra cpu --extra service` | Installs FastAPI and Uvicorn for `boxmot-service`. |
| `eval --compare-trackeval` | `boxmot install --extra trackeval` | `uv sync --extra cpu --extra trackeval` | Adds the TrackEval reference comparison for AABB MOTChallenge datasets. |
| `export --include onnx` | `boxmot install --extra onnx` | `uv sync --extra cpu --extra onnx` | The default export path uses ONNX. |
| `export --include coreml` | `boxmot install --extra coreml` | `uv sync --extra cpu --extra coreml` | Native FP16 MLProgram export and inference on macOS. |
| `export --include openvino` | `boxmot install --extra onnx --extra openvino` | `uv sync --extra cpu --extra onnx --extra openvino` | Uses ONNX as an intermediate. |
| `export --include tflite` | `boxmot install --extra tflite` | `uv sync --extra cpu --extra tflite` | Installs both TFLite export and LiteRT inference packages. |

You can combine extras when needed:

```bash
uv sync --extra cpu --extra yolo --extra evolve --extra research
boxmot install --extra yolo --extra evolve --extra research
```

Download helpers, ReID backends, exporters, tuning, and research validate
dependencies when used and report an installation command when requirements
are missing. Install dependencies explicitly in the interpreter that runs your
application:

```bash
python -m boxmot.engine.cli install --extra onnx
```

The `install` command targets that interpreter and checks nested package
extras as well as version constraints. Use the PyTorch profile commands above
for `cpu` and `cu130`; those profile extras are not accepted by `boxmot install`.
See [Install dependencies](../modes/install.md) for custom requirements and
TensorRT setup. Native Core ML requires macOS; TensorRT requires a compatible
CUDA/NVIDIA stack.

## Docker

The shared Dockerfile provides four production targets. Both CLI images
include the `yolo` and `trackeval` extras for detector, evaluation, and
interactive workflows:

| Workload | Build target | BoxMOT v24 tag | Rolling tag |
| --- | --- | --- | --- |
| Full CUDA CLI | `cli-gpu` | `boxmot/boxmot:24.0.0` | `boxmot/boxmot:latest` |
| Full CPU CLI | `cli-cpu` | `boxmot/boxmot:24.0.0-cpu` | `boxmot/boxmot:latest-cpu` |
| CPU tracker service | `service-cpu` | `boxmot/boxmot-service:24.0.0` | `boxmot/boxmot-service:latest` |
| CUDA/ReID tracker service | `service-gpu` | `boxmot/boxmot-service:24.0.0-gpu` | `boxmot/boxmot-service:latest-gpu` |

The CPU targets install CPU-only Torch and contain no CUDA runtime. The GPU
targets install the locked CUDA 13.0 Torch wheels; they require a compatible
NVIDIA host driver and the NVIDIA Container Toolkit. Both service images use
Torch for canonical structures, but only the GPU service performs ReID
enrichment. Neither service runs a detector.

Run the published GPU image with the NVIDIA Container Toolkit and a compatible
host driver:

```bash
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" \
  --workdir /workspace \
  boxmot/boxmot:24.0.0
```

Use the CPU-suffixed image on hosts without NVIDIA GPUs:

```bash
docker run --rm -it \
  -v "$PWD:/workspace" \
  --workdir /workspace \
  boxmot/boxmot:24.0.0-cpu
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

Every target also receives a `sha-<commit>` tag, with `-cpu` or `-gpu` appended
for the suffixed variants.

For materialization and evaluation, keep raw datasets, builds, and model
artifacts on the host. This example uses the repository's MOT data layout and
persists downloads and builds across containers:

```bash
mkdir -p "$PWD/datasets/mot" "$PWD/runs/materializations" "$PWD/models"

docker run --rm --gpus all --ipc=host \
  -v "$PWD/datasets/mot:/opt/boxmot/datasets/mot" \
  -v "$PWD/runs/materializations:/materializations" \
  -v "$PWD/models:/opt/boxmot/models" \
  -e BOXMOT_BUILDS_DIR=/materializations \
  boxmot/boxmot:24.0.0 \
  boxmot materialize \
    --experiment mot17/ablation-yolox-lmbn.yaml \
    --device 0

BUILD_ID=replace-with-the-64-character-build-id

docker run --rm --gpus all --ipc=host \
  -v "$PWD/datasets/mot:/opt/boxmot/datasets/mot:ro" \
  -v "$PWD/runs/materializations:/materializations:ro" \
  -v "$PWD/models:/opt/boxmot/models:ro" \
  -e BOXMOT_BUILDS_DIR=/materializations \
  boxmot/boxmot:24.0.0 \
  boxmot eval \
    --experiment mot17/ablation-yolox-lmbn.yaml \
    --build "$BUILD_ID" \
    --tracker occluboost
```

Use the `24.0.0-cpu` image without `--gpus all` and materialize with
`--device cpu` on CPU-only hosts. See the
[deployment guide](../guides/deployment.md) for the mount and service details.

Run the HTTP image when detections come from a separate detector:

```bash
docker run --rm -p 8000:8000 boxmot/boxmot-service:latest
```

This CPU image supports ByteTrack, OcSort, and SFSORT without image pixels. Run
the CUDA/ReID image with a mounted checkpoint:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/models/osnet_x0_25_msmt17.pt:/models/osnet_x0_25_msmt17.pt:ro" \
  -e BOXMOT_SERVICE_REID_WEIGHTS=/models/osnet_x0_25_msmt17.pt \
  boxmot/boxmot-service:latest-gpu
```

The GPU image defaults to BotSort and also supports StrongSort, DeepOcSort,
HybridSort, BoostTrack, and OccluBoost. It requires a raw base64-encoded JPEG or
PNG in `image_base64` on every frame, including frames without detections. Both
service images run as a non-root user and expose health checks, OpenAPI at
`/docs`, and a stateful frame endpoint. They consume external detections and do
not run detector inference. See
[Tracker service deployment](../guides/deployment.md) for the request schema,
payload guidance, and scaling model.

## Native C++ backends

The CPU and GPU CLI Docker images bundle native ReID and the C API tracker
backends for `botsort`, `bytetrack`, `ocsort`, `occluboost`, and `sfsort`.
`--tracker-backend cpp` therefore works in those images without a compiler or
CMake. The service images use Python tracker backends.

For a normal host installation, native backends are built lazily the first time
you select `--tracker-backend cpp`.

Install the native build tools before using them:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

Example:

```bash
boxmot track --detector yolov8n --tracker bytetrack --tracker-backend cpp --source video.mp4
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID --tracker bytetrack --tracker-backend cpp
```

The generated build files are kept under `build/native/<tracker>/`.
For editable installs or an up-front build, compile native ReID and all live
tracker libraries with `boxmot build`, or select one with
`boxmot build --tracker bytetrack`. Evaluation and tuning stream keyed Parquet
rows through the same live native API; there are no cache replay executables.

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
        import boxmot
        from boxmot.trackers import TrackerSpec

        tracker = boxmot.create_tracker(TrackerSpec(name="bytetrack"))
        print(boxmot.__version__, tracker.requirements)
        ```

## Next steps

- Use [Quickstart](../index.md) for a minimal path.
- Use [Modes Overview](../modes/index.md) to decide between `track`, `materialize`, `eval`, `tune`, `research`, `train-reid`, `eval-reid`, `compare-reid`, `export`, and `build`.
- Use [Native C++ Integration](../native/index.md) for native build and embedding details.
- Use the workflow table above to add the extras your workflow needs.
