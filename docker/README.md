# Docker images and Compose

Use the [Docker integration guide](../docs/integrations/docker.md) for video jobs,
CPU/GPU inference, Linux webcam previews, Ollama, pinned models, cache ownership,
database backups/upgrades, and cleanup.

- [Continuous cameras on Kubernetes](../docs/integrations/kubernetes.md)
- [HTTP tracking service deployment](../docs/guides/deployment.md)
- [Real Compose inference and recovery test](../tests/e2e/README.md)

## Images

One multi-stage Dockerfile provides four targets:

| Target | Contents | Published tags |
| --- | --- | --- |
| `cli-cpu` | Detection, tracking, attributes and evaluation with CPU PyTorch | `boxmot/boxmot:<version>-cpu`, `sha-<commit>-cpu`, `latest-cpu` |
| `cli-gpu` | Full CLI with CUDA 13.0 PyTorch | Build locally |
| `service-cpu` | Non-root, headless detection-to-track HTTP service | `boxmot/boxmot-service:<version>`, `sha-<commit>`, `latest` |
| `service-gpu` | Non-root HTTP service with CUDA ReID | Build locally |

`<version>` is the release version in the root `pyproject.toml`. Prefer image
digests when deploying. A build without `--target` selects `cli-gpu`.

From the repository root:

```bash
docker build --target cli-cpu -f docker/Dockerfile -t boxmot/boxmot:local-cpu .
docker build --target cli-gpu -f docker/Dockerfile -t boxmot/boxmot:local-gpu .
docker build --target service-cpu -f docker/Dockerfile -t boxmot/boxmot-service:local .
docker build --target service-gpu -f docker/Dockerfile -t boxmot/boxmot-service:local-gpu .
```

## Build cache and dependencies

CPU and CUDA dependencies use the root `pyproject.toml`, `uv.lock`, and pinned uv
version. The mutually exclusive `cpu` and `cu130` extras select PyTorch wheels;
the CUDA wheels provide user-space libraries without a second CUDA base image.
Run `uv lock` when dependencies change.

System packages, native compilation, dependency environments and application
installation have separate layers. The dependency-manifest stage normalizes
only the local BoxMOT version, then checks those exact inputs with
`uv lock --check --offline` before `uv sync --frozen`. Version-only releases
reuse dependency layers; wheel builds retain the real release version.

CLI images install a wheel containing native libraries built inside Docker.
Service images copy their source and dependency environment separately.
Compilers remain in build stages. `.dockerignore` excludes host-generated
native binaries, CMake outputs, datasets, weights and local environment files.

Release/manual image CI builds and smoke-tests CPU targets before publishing.
BuildKit intermediate layers use separate `mode=max` cache scopes per target;
uv cache mounts speed up builds locally but are not exported as download caches
between workflow jobs. GPU execution requires a suitable host and is not covered
by the hosted CPU image workflow.

## Compose files

| File | Purpose |
| --- | --- |
| `compose.attributes.yaml` | Complete CPU vLLM tracking stack and optional standalone services |
| `compose.attributes.gpu.yaml` | GPU vLLM and CUDA tracking override |
| `compose.attributes.ollama.yaml` | CPU Ollama tracking override with model digest verification |
| `compose.webcam.yaml` | Linux camera and X11 access; layer last |

GPU and Ollama are alternative provider overrides. Use Compose 2.24.4 or newer.
The [integration guide](../docs/integrations/docker.md) is the canonical source
for commands and settings. PR checks render all six provider/input combinations;
weekly/manual CI exercises real CPU inference and streaming recovery.
