# Docker

Use Docker images for finite video jobs, batch evaluation, and the HTTP tracking
service. Use the [Helm deployment](kubernetes.md) for continuous cameras with
PostgreSQL, model serving, ownership, recovery and retention. The same chart
supports local Minikube and cloud Kubernetes.

Compose files and the former Kustomize deployment have been retired. Removing
them from the checkout does not stop existing containers or delete their
volumes. Back up existing PostgreSQL data and follow the
[Helm migration and schema-upgrade procedure](kubernetes.md#upgrade-the-schema-and-install)
before moving clients. Keep previous configuration and image references while
migrating; do not delete a database volume to resolve a schema mismatch.

## Build images

Run from the repository root:

```bash
docker build --target cli-cpu -f docker/Dockerfile -t boxmot/boxmot:local-cpu .
docker build --target cli-gpu -f docker/Dockerfile -t boxmot/boxmot:local-gpu .
```

The CPU image uses CPU-only PyTorch; the GPU image uses the locked CUDA 13.0
PyTorch wheels. GPU execution requires a compatible NVIDIA host driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
A build without `--target` selects `cli-gpu`.

The CLI images include detection, tracking, evaluation and attribute client
dependencies, native ReID, and C++ backends for BotSort, ByteTrack, OcSort,
OccluBoost and SFSORT. Export and tuning may need their
[additional extras](../getting-started/installation.md#mode-specific-extras).
The images do not start a PostgreSQL or model server.

Published CPU images use `boxmot/boxmot:<version>-cpu` and
`boxmot/boxmot:sha-<commit>-cpu`; `<version>` comes from `pyproject.toml`.
Use `repository@sha256:...` for repeatable deployments. GPU images are built
separately and still require inference validation on the intended hardware.

Release CI builds and smoke-tests CPU images once, retains their archives for
30 days, and publishes the verified candidates after the package and code
release succeed. Publication checks the archive checksum, source commit,
version, target and Docker image ID before pushing. Manual publication must
select the prior gate's `candidate_run_id` and matching release tag/commit;
it cannot silently rebuild a replacement. See
[release gates and recovery](../contributing/ci.md#release-gates).

## Run a CPU video job

Copy a video to `data/video.mp4`, then run from your working directory:

```bash
mkdir -p data models runs .cache

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" --workdir /workspace \
  -e HOME=/tmp -e XDG_CACHE_HOME=/workspace/.cache \
  boxmot/boxmot:local-cpu \
  boxmot track --source data/video.mp4 --detector yolo26n \
    --tracker bytetrack --device cpu --project runs/docker --save --save-json
```

The bind mount preserves inputs, downloaded models, caches and results. Running
with your UID/GID keeps created host files owned by your account. Built-in
profiles with explicit `models/...` paths use the working directory; bare
checkpoint names use the [model-path policy](../python/index.md#model-files-and-logging).
`XDG_CACHE_HOME` keeps installed-package caches under the mounted directory.

Results go into a new `runs/docker/exp/` directory, incrementing its name on
subsequent runs. Use `--exist-ok` only when deliberately reusing an output
folder. Omit `--save` if you only need JSONL records.

For an interactive shell, replace the trailing `boxmot track ...` command with
`bash` and add `-it` to `docker run`. The environment is already on `PATH`:

```bash
boxmot --help
```

## Run a GPU video job

After building `local-gpu`, use the same host folders:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" --workdir /workspace \
  -e HOME=/tmp -e XDG_CACHE_HOME=/workspace/.cache \
  boxmot/boxmot:local-gpu \
  boxmot track --source data/video.mp4 --detector yolo26n \
    --tracker botsort --device 0 --project runs/docker-gpu --save --save-json
```

`--gpus all` exposes the host GPUs; `--device 0` selects one logical device
inside the container. For continuous GPU cameras or a shared GPU attribute
server, configure the [Helm GPU settings](kubernetes.md#storage-scheduling-and-cloud-deployment).
A built-in webcam can be published as a network stream using the
[camera connection instructions](kubernetes.md#configure-cameras-and-dependencies).

## Attributes and services

Attribute-enabled jobs need an existing PostgreSQL database and an Ollama or
vLLM endpoint reachable from the container. Set `BOXMOT_ATTRIBUTES_DATABASE_URL`
and pass `--attributes PROFILE --attributes-host URL`; see
[Track attributes](../config/attributes.md) for profile and local database setup.
`127.0.0.1` inside a container refers to that container, so use reachable service
addresses instead of the host guide's loopback URLs. Use the Helm chart to
operate the complete camera stack.

The Dockerfile also provides `service-cpu` and `service-gpu`. These non-root HTTP
services consume external detections; their settings, checkpoint mounts and
build commands are in [Service deployment](../guides/deployment.md#build-and-run).

## Batch materialization and evaluation

Keep the working directory mounted at the same container path in both commands
so saved references remain valid:

```bash
mkdir -p datasets/mot models runs/materializations .cache

docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" --workdir /workspace \
  -e HOME=/tmp -e XDG_CACHE_HOME=/workspace/.cache \
  -e BOXMOT_BUILDS_DIR=/workspace/runs/materializations \
  boxmot/boxmot:local-gpu \
  boxmot materialize --experiment mot17/ablation-yolox-lmbn.yaml --device 0

BUILD_ID=replace-with-the-64-character-build-id

docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" --workdir /workspace \
  -e HOME=/tmp -e XDG_CACHE_HOME=/workspace/.cache \
  -e BOXMOT_BUILDS_DIR=/workspace/runs/materializations \
  boxmot/boxmot:local-gpu \
  boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml \
    --build "$BUILD_ID" --tracker occluboost
```

Use the build ID printed by `materialize`. `BOXMOT_BUILDS_DIR` directs lookup to
the persistent mount. Optional dataset packages and large downloads are not
baked into the image. Build the relevant image target locally when needed.
