# Docker

Use the CLI images to run tracking and batch workflows in a container. For
stateful HTTP tracking from external detections, use the separate
[service deployment guide](../guides/deployment.md).

## Run the CPU CLI

Mount your working directory so input videos and saved outputs remain on the
host:

```bash
docker run --rm -it \
  -v "$PWD:/workspace" \
  --workdir /workspace \
  boxmot/boxmot:latest-cpu
```

Inside the container, the virtual environment is already on `PATH`:

```bash
boxmot --help
boxmot track --source video.mp4 --detector yolov8n --tracker bytetrack --device cpu --save
```

The CLI images include the `yolo` and `trackeval` extras, native ReID, and the
C++ backends for BotSort, ByteTrack, OcSort, OccluBoost, and SFSORT. Select
`--tracker-backend cpp` without installing a compiler or CMake. Export, tuning,
and other optional workflows need their [additional extras](../getting-started/installation.md#mode-specific-extras).

## Build a CLI image

From a repository checkout, choose a target in `docker/Dockerfile`:

```bash
docker build --target cli-cpu -f docker/Dockerfile -t boxmot/boxmot:local-cpu .
docker build --target cli-gpu -f docker/Dockerfile -t boxmot/boxmot:local-gpu .
```

The CPU image uses CPU-only Torch. The GPU image uses the locked CUDA 13.0
Torch wheels and requires a compatible NVIDIA host driver and the NVIDIA
Container Toolkit:

```bash
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" \
  --workdir /workspace \
  boxmot/boxmot:local-gpu
```

The Dockerfile also provides `service-cpu` and `service-gpu` targets. Their
runtime settings, checkpoint mounts, and build commands are in
[Service deployment](../guides/deployment.md#build-and-run). Service images use
Python tracker backends and consume detections from an external detector.

## Reproducible batch jobs

Keep datasets, materialized builds, model artifacts, and evaluation outputs on
the host. After building `local-gpu` above, run:

```bash
mkdir -p "$PWD/datasets/mot" "$PWD/runs/materializations" "$PWD/models"

docker run --rm --gpus all --ipc=host \
  -v "$PWD/datasets/mot:/opt/boxmot/datasets/mot" \
  -v "$PWD/runs:/opt/boxmot/runs" \
  -v "$PWD/models:/opt/boxmot/models" \
  -e BOXMOT_BUILDS_DIR=/opt/boxmot/runs/materializations \
  boxmot/boxmot:local-gpu \
  boxmot materialize \
    --experiment mot17/ablation-yolox-lmbn.yaml \
    --device 0

BUILD_ID=replace-with-the-64-character-build-id

docker run --rm --gpus all --ipc=host \
  -v "$PWD/datasets/mot:/opt/boxmot/datasets/mot:ro" \
  -v "$PWD/runs:/opt/boxmot/runs" \
  -v "$PWD/models:/opt/boxmot/models:ro" \
  -e BOXMOT_BUILDS_DIR=/opt/boxmot/runs/materializations \
  boxmot/boxmot:local-gpu \
  boxmot eval \
    --experiment mot17/ablation-yolox-lmbn.yaml \
    --build "$BUILD_ID" \
    --tracker occluboost
```

Use the build ID printed by `materialize`. Matching mount points preserve
source references and let `--build` resolve under `BOXMOT_BUILDS_DIR`. The
writable `runs` mount also retains evaluation results. Built-in dataset and
model paths are relative to `/opt/boxmot`.

On CPU-only hosts, use `boxmot/boxmot:latest-cpu`, remove `--gpus all`, and
materialize with `--device cpu`.

For repeatable runs, pin an image version or commit tag instead of a rolling
alias. The release workflow publishes CPU CLI tags as `<version>-cpu`,
`sha-<commit>-cpu`, and `latest-cpu`, and CPU service tags as `<version>`,
`sha-<commit>`, and `latest`. Build the GPU targets locally when needed.
