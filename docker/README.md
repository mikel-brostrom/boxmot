# Docker images

BoxMOT uses one shared multi-stage Dockerfile for three independently built
images:

| Target | Suggested tag | Contents |
| --- | --- | --- |
| `cli-gpu` | `boxmot/boxmot:latest` | Full detector, ReID, CLI, and evaluation stack with CUDA 13.0 PyTorch |
| `cli-cpu` | `boxmot/boxmot:latest-cpu` | The same full stack with CPU-only PyTorch |
| `service` | `boxmot/boxmot-service:latest` | Non-root, geometry-only detection-to-track HTTP service |

The CPU and CUDA selections come from mutually exclusive, lockfile-backed `cpu`
and `cu130` extras in the Docker-only project. This keeps the published BoxMOT
dependency contract unchanged. The service installs only the minimal
`service-runtime` group and runs BoxMOT directly from its source package. It
therefore contains neither PyTorch nor CUDA, and uses headless OpenCV. A final
`default` stage aliases `cli-gpu`, so no-target builds still produce the full
CUDA image.

The GPU target intentionally starts from the same Python slim base as the CPU
target. Its locked `+cu130` PyTorch wheels provide the matching user-space CUDA
libraries, avoiding a second CUDA runtime from a PyTorch base image. The host
still needs a compatible NVIDIA driver and NVIDIA Container Toolkit.

## Build

Run builds from the repository root:

```bash
docker build --target cli-gpu -f docker/Dockerfile -t boxmot/boxmot:local .
docker build --target cli-cpu -f docker/Dockerfile -t boxmot/boxmot:local-cpu .
docker build --target service -f docker/Dockerfile -t boxmot/boxmot-service:local .
```

The default build is equivalent to `--target cli-gpu`:

```bash
docker build -f docker/Dockerfile -t boxmot/boxmot:local .
```

When image dependencies change, regenerate the Docker-specific lock with the
same uv version pinned in the Dockerfile:

```bash
uvx --from uv==0.12.4 uv lock --project docker
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

Run the detection-to-track service:

```bash
docker run --rm -p 8000:8000 boxmot/boxmot-service:local
curl --fail http://127.0.0.1:8000/healthz
```

The production service should remain one process per container. Scale with
multiple containers and route every stream/session consistently to the same
instance. See the [deployment guide](../docs/guides/deployment.md) for the
request schema, state model, and scaling constraints.

A GitHub Actions runner is intentionally not included. A self-hosted runner is
CI infrastructure with a different security and lifecycle model, not a BoxMOT
runtime image.
