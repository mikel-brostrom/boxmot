# Docker

Run finite video jobs and Linux webcam previews with Compose. For continuous
camera ownership, recovery and retention, use the [streaming deployment](kubernetes.md).
For HTTP tracking from external detections, see [service deployment](../guides/deployment.md).

## Tracking and vLLM with Compose

The base [`compose.attributes.yaml`](https://github.com/mikel-brostrom/boxmot/blob/master/docker/compose.attributes.yaml)
runs BoxMOT, PostgreSQL and vLLM on the CPU: YOLO26n detection, ByteTrack tracking
and `Qwen/Qwen3.5-0.8B` attributes. Python dependencies are inside the images.
Use Docker with Compose **2.24.4 or newer** for all documented overlays.
Run commands from the repository root in a terminal with Docker access.
If Docker requires sudo, pass settings with `sudo env NAME=value docker ...`;
the webcam launcher handles this automatically.

### Process a video

```bash
export BOXMOT_UID="$(id -u)" BOXMOT_GID="$(id -g)"
mkdir -p data models runs
cp /path/to/video.mp4 data/video.mp4

docker compose -f docker/compose.attributes.yaml up --build \
  --abort-on-container-exit --exit-code-from boxmot boxmot
```

BoxMOT starts after PostgreSQL and vLLM pass their health checks. Initial builds,
model downloads and CPU initialization may take several minutes. The server
health check does not warm up image inference. On completion or Ctrl+C, Compose
stops the job and its dependencies.

Input `data/` is mounted read-only at `/inputs`. Detector/ReID downloads live in
`models/`; outputs live in `runs/compose/exp/`. Subsequent runs use `exp2/`,
`exp3/`, etc., preserving earlier video, tracks and attributes. All exports from
one run share its directory. `BOXMOT_RUN_NAME` changes the parent directory.
Pass `--exist-ok` only when deliberately reusing and overwriting an output run:

```bash
docker compose -f docker/compose.attributes.yaml run --rm boxmot \
  --source /inputs/video.mp4 --exist-ok
```

Unlike `up --abort-on-container-exit`, `run` leaves dependency servers running.
Use the [shutdown command](#results-and-shutdown) afterward.

### Show a Linux webcam preview

Run as your normal user from a Linux desktop terminal with `DISPLAY` set:

```bash
./docker/webcam.sh
```

The launcher creates output directories, passes your UID/GID, detects the camera
device group, and layers `compose.webcam.yaml` over the CPU stack. It maps
`/dev/video0` and the X11/Xwayland socket. It asks for sudo only when Docker
requires it. If X11 rejects your user, run
`xhost +si:localuser:$(id -un)` from the desktop terminal and retry.
Press `q`, `Esc`, or Ctrl+C to stop.

```bash
VIDEO_DEVICE=/dev/video2 BOXMOT_RUN_NAME=webcam-test ./docker/webcam.sh
```

### GPU detection, ReID, and VLM inference

The GPU overlay selects `cli-gpu`, BoT-SORT, CUDA detection/ReID and a GPU vLLM
server. The host needs a driver compatible with CUDA 13.0 PyTorch and the NVIDIA
Container Toolkit. See [Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/).
Stop the CPU stack before switching: both model services publish port `8001`.

```bash
export BOXMOT_UID="$(id -u)" BOXMOT_GID="$(id -g)"
mkdir -p data models runs

docker compose -f docker/compose.attributes.yaml \
  -f docker/compose.attributes.gpu.yaml up --build \
  --abort-on-container-exit --exit-code-from boxmot boxmot
```

Both containers share host GPU 0 by default; `VLLM_GPU_MEMORY_UTILIZATION=0.5`
budgets vLLM memory without isolating GPU resources. With two GPUs, set
`BOXMOT_GPU_ID=0 VLLM_GPU_ID=1`. BoxMOT still uses device `0` inside its container.
Outputs default to `runs/compose-gpu/exp/`, then numbered directories.
The small `person-appearance-vllm-cpu` attribute preset is also used here;
the server determines the VLM hardware.

For GPU webcam inference, add the webcam overlay last:

```bash
export VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video0}"
export VIDEO_GID="$(stat -c '%g' "$VIDEO_DEVICE")"
docker compose -f docker/compose.attributes.yaml \
  -f docker/compose.attributes.gpu.yaml -f docker/compose.webcam.yaml \
  up --build --abort-on-container-exit --exit-code-from boxmot boxmot
```

Use the same files for subsequent logs, stop and down commands. GPU and Ollama
are alternative provider overlays; do not combine them.

### Settings and logs

| Variable | Default / purpose |
| --- | --- |
| `BOXMOT_UID`, `BOXMOT_GID` | `1000`; set to your host UID/GID for writable outputs. |
| `BOXMOT_RUN_NAME` | `compose`, `compose-gpu`, or `compose-ollama`; parent under `runs/`. |
| `BOXMOT_INPUT_DIR` | `../data`, relative to the base Compose file; use an absolute path elsewhere. |
| `BOXMOT_SOURCE` | `/inputs/video.mp4`, or `0` for the webcam overlay. |
| `BOXMOT_DETECTOR` | `yolo26n`; profile or weights path inside the container. |
| `BOXMOT_TRACKER` | `bytetrack`, or `botsort` with the GPU overlay. |
| `BOXMOT_REID` | `osnet-x0-25-msmt17`, used by appearance trackers. |
| `BOXMOT_GPU_ID`, `VLLM_GPU_ID` | `0`; host GPUs exposed to the respective containers. |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.5`; vLLM GPU memory budget. |
| `VLLM_CPU_IMAGE`, `VLLM_GPU_IMAGE`, `OLLAMA_IMAGE` | Override the digest-pinned server images. |
| `VLLM_MODEL_REVISION` | Pinned Hugging Face commit for the small Qwen model. |
| `OLLAMA_MODEL_DIGEST` | Expected full digest of the installed `qwen3.5:9b` model. |
| `VLLM_PORT`, `OLLAMA_PORT` | Loopback ports `8001`, `11434`. |
| `BOXMOT_POSTGRES_PORT` | Loopback database port `5432`. |
| `BOXMOT_POSTGRES_USER`, `BOXMOT_POSTGRES_PASSWORD`, `BOXMOT_POSTGRES_DB` | Local defaults `boxmot`, `boxmot-local`, `boxmot`. |
| `COMPOSE_PROJECT_NAME` | Override `boxmot-attributes` to isolate another stack; also select unused host ports. |

Database settings initialize an empty volume only; changing them does not alter
existing PostgreSQL users or passwords. Use URL-safe credential values because
Compose embeds them in the client's connection URL. Host clients configure
[their own database URL](../config/attributes.md#configure-postgresql).
Container clients use service DNS and internal ports, independent of host ports.

```bash
docker compose -f docker/compose.attributes.yaml logs --tail 100 -f boxmot vllm-cpu postgres
```

### Pinned models and cache ownership

Server images use immutable digests. Both vLLM services pin the model with
`--revision`; changing the image does not implicitly update its weights.
The defaults live in the Compose files. The CPU image/revision match the
[recorded inference run](https://github.com/mikel-brostrom/boxmot/blob/master/deploy/kubernetes/VALIDATION.md).
GPU and Ollama image manifests were registry-verified; inference on those
backends still needs acceptance testing on the target hardware.

vLLM owns a persistent `vllm-models` named volume shared by the CPU and GPU
variants of a Compose project. It no longer writes into your personal Hugging
Face cache. Old host downloads remain untouched; the new volume downloads its
pinned snapshot on first use. Ollama uses `ollama-models`. BoxMOT's own caches
remain under the host-owned `models/` mount. Changing projects creates separate
server caches and databases.

To prefill the vLLM volume from an existing host snapshot without giving the
server write access to your home, stop vLLM and run:

```bash
docker compose -f docker/compose.attributes.yaml run --rm --no-deps \
  -v "$HOME/.cache/huggingface/hub:/seed:ro" --entrypoint sh vllm-cpu \
  -c 'mkdir -p /root/.cache/huggingface/hub && cp -a /seed/. /root/.cache/huggingface/hub/'
```

Update image digests and model revisions deliberately, rerun the integration
test, and keep the previous references for rollback. An override can use any
image reference, but use a digest for repeatable deployments.

### CPU resource use

vLLM uses a 256 MiB KV cache, a 2,048-token context, one active request, and an
image preprocessing cap of 262,144 pixels. Prefix caching and the multimodal
processor cache are disabled. OpenMP reserves two CPU cores by default; set
`VLLM_CPU_OMP_THREADS_BIND` and `VLLM_CPU_NUM_OF_RESERVED_CPU` for your host.
Weights and runtime buffers need additional RAM. Use `docker stats --no-stream`,
`free -h`, and `vmstat 1 5` to check memory pressure before increasing timeouts.

## Choose an attribute provider

Use the base stack for CPU vLLM, the GPU overlay for CUDA vLLM, or the Ollama
overlay below. Every provider requires PostgreSQL. The webcam overlay can be
added last to any one of these configurations.

### Ollama

Prepare the pinned model once, then use the same complete-stack invocation as
vLLM. Run these commands in the same terminal:

```bash
export COMPOSE_FILE=docker/compose.attributes.yaml:docker/compose.attributes.ollama.yaml
export BOXMOT_UID="$(id -u)" BOXMOT_GID="$(id -g)"
mkdir -p data models runs
cp /path/to/video.mp4 data/video.mp4

docker compose up -d --wait postgres ollama
docker compose exec ollama ollama pull qwen3.5:9b
docker compose up --build --abort-on-container-exit --exit-code-from boxmot boxmot
```

Before tracking, the BoxMOT container checks `/api/tags` against the full
`OLLAMA_MODEL_DIGEST` from the overlay. A missing model or changed tag fails
with an explicit error. Keep the prepared model volume to reproduce the pinned
revision. If the registry tag changes, restore your retained model volume or
validate the new model and explicitly update the digest; do not bypass the
check accidentally by copying a new digest without testing inference.
The server health check only checks availability, so it can start before the
model is pulled. This profile uses the 9B model on the CPU and needs more memory
than the small vLLM profile.

### vLLM CPU

For repeated jobs sharing already started servers:

```bash
export COMPOSE_FILE=docker/compose.attributes.yaml
export BOXMOT_UID="$(id -u)" BOXMOT_GID="$(id -g)"
docker compose build boxmot
docker compose up -d --wait --wait-timeout 1800 postgres vllm-cpu
docker compose run --rm --no-deps boxmot --source /inputs/video.mp4
```

### Results and shutdown

Results include annotated video, `tracks.jsonl`, and `attributes.jsonl` in each
unique run directory. PostgreSQL retains attributes and run history in its
`postgres-data` volume. The named model caches and host outputs survive:

```bash
docker compose --profile '*' down
```

Set `COMPOSE_FILE` as above or pass the same `-f` files used to start the stack.
Adding `-v` deletes **all named volumes**, including the database and both model
caches. Run history uses new session IDs for each finite job.

## Upgrade an existing Compose database

Version-1 attribute databases need an explicit upgrade before this release can
use them. Fresh databases initialize automatically. Keep your previous image
reference and configuration. Stop **all** clients of this database, including
host CLI jobs, other Compose projects and any streaming/maintenance workers.
Keep them stopped until the upgrade succeeds.

Select your normal `COMPOSE_FILE` (including the GPU or Ollama overlay if used),
then run from a Docker-enabled terminal:

```bash
set -e
umask 077
docker compose build boxmot
docker compose stop boxmot
docker compose up -d --wait postgres
mkdir -p runs/backups
backup="runs/backups/attributes-$(date -u +%Y%m%dT%H%M%SZ).dump"
docker compose exec -T postgres sh -ec \
  'exec pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > "$backup"
docker compose exec -T postgres pg_restore --list < "$backup" > /dev/null

docker compose run --rm --no-deps --entrypoint boxmot boxmot stream-db upgrade
docker compose up --abort-on-container-exit --exit-code-from boxmot boxmot
```

The one-off container uses the existing Compose database URL and bypasses
tracking/model startup. The additive upgrade preserves runs, crops and outcomes
and is safe to rerun. It allows ten minutes per SQL statement and thirty seconds
to acquire locks. Validate restore of the backup separately before a production
upgrade. Do not delete the database volume to resolve a schema mismatch.
Restart any other stopped clients after the upgrade succeeds.

Old version-1 clients reject the upgraded database. Rolling back to them requires
restoring the matching backup into a separate PostgreSQL database and pointing
the old deployment to it; keeping the upgraded volume protects later data.
Releases supporting the same schema version can reuse the database.

## Verify the integration

Run the inexpensive six-combination check without starting containers:

```bash
uv run --no-sync python -m tests.ci.compose_contract
```

Relevant PRs run this check in CI. A weekly/manual workflow runs actual CPU
inference and recovery using the [Compose end-to-end harness](https://github.com/mikel-brostrom/boxmot/blob/master/tests/e2e/README.md).
GPU inference remains a hardware acceptance check.

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

The CLI images include the `yolo`, `trackeval`, and `attributes` extras, native
ReID, and the C++ backends for BotSort, ByteTrack, OcSort, OccluBoost, and SFSORT. Select
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
