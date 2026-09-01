# Docker images

BoxMOT uses one shared multi-stage Dockerfile for four independently built
images:

| Target | Suggested tag | Contents |
| --- | --- | --- |
| `cli-gpu` | `boxmot/boxmot:latest` | Full detector, ReID, CLI, and evaluation stack with CUDA 13.0 PyTorch |
| `cli-cpu` | `boxmot/boxmot:latest-cpu` | The same full stack with CPU-only PyTorch |
| `service-cpu` | `boxmot/boxmot-service:latest` | Non-root CPU geometry-only detection-to-track HTTP service |
| `service-gpu` | `boxmot/boxmot-service:latest-gpu` | Non-root CUDA/ReID detection-to-track HTTP service |

The CPU and CUDA selections come from mutually exclusive, lockfile-backed `cpu`
and `cu130` extras in the root project. Docker, local development, and CI all
consume the same `pyproject.toml` and `uv.lock`. The CPU service installs only
the minimal `service-runtime` group and runs BoxMOT directly from source, so it
contains neither PyTorch nor CUDA and uses headless OpenCV. The GPU service
selects the root CUDA/ReID and HTTP extras, without detector or evaluation
extras. A final `default` stage aliases `cli-gpu`, so no-target builds still
produce the full CUDA image.

The GPU target intentionally starts from the same Python slim base as the CPU
target. Its locked `+cu130` PyTorch wheels provide the matching user-space CUDA
libraries, avoiding a second CUDA runtime from a PyTorch base image. The host
still needs a compatible NVIDIA driver and NVIDIA Container Toolkit.

## Build

Run builds from the repository root:

```bash
docker build --target cli-gpu -f docker/Dockerfile -t boxmot/boxmot:local .
docker build --target cli-cpu -f docker/Dockerfile -t boxmot/boxmot:local-cpu .
docker build --target service-cpu -f docker/Dockerfile -t boxmot/boxmot-service:local .
docker build --target service-gpu -f docker/Dockerfile -t boxmot/boxmot-service:local-gpu .
```

The default build is equivalent to `--target cli-gpu`:

```bash
docker build -f docker/Dockerfile -t boxmot/boxmot:local .
```

When dependencies change, regenerate the single root lock with the same uv
version pinned in the Dockerfile and CI helper:

```bash
uvx --from uv==0.12.4 uv lock
```

## Run

Run the CUDA CLI image with the NVIDIA Container Toolkit:

```bash
docker run --rm -it --ipc=host --gpus all boxmot/boxmot:local
```

Run the CPU CLI image:

```bash
docker run --rm -it --ipc=host boxmot/boxmot:local-cpu
```

Run the CPU geometry-only detection-to-track service:

```bash
docker run --rm -p 8000:8000 boxmot/boxmot-service:local
curl --fail http://127.0.0.1:8000/healthz
```

It supports ByteTrack, OCSort, and SFSORT and does not need image pixels. Run
the CUDA/ReID service with an NVIDIA GPU and a mounted checkpoint:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/models/osnet_x0_25_msmt17.pt:/models/osnet_x0_25_msmt17.pt:ro" \
  -e BOXMOT_SERVICE_REID_WEIGHTS=/models/osnet_x0_25_msmt17.pt \
  boxmot/boxmot-service:local-gpu
```

The GPU service defaults to BotSORT and also supports StrongSORT, DeepOCSORT,
HybridSORT, BoostTrack, and OccluBoost. Its request must contain a raw
base64-encoded JPEG or PNG in `image_base64` for every frame, even when
`detections` is empty. Base64 increases the compressed payload by roughly 33%,
so prefer compressed JPEG for high-volume streams and enforce request-size
limits at ingress.

Neither service runs detector inference. Keep one service process per
container; the GPU process loads and warms one ReID model shared by its tracker
sessions and defaults to one concurrent tracker update. Scale with multiple
containers and route every stream/session consistently to the same instance.
See the [deployment guide](../docs/guides/deployment.md) for the request schema,
state model, and scaling constraints.

A GitHub Actions runner is intentionally not included. A self-hosted runner is
CI infrastructure with a different security and lifecycle model, not a BoxMOT
runtime image.
