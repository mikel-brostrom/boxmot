# Docker

Use the CLI images to run tracking and batch workflows in a container. For
stateful HTTP tracking from external detections, use the separate
[service deployment guide](../guides/deployment.md).

For video tracking with Ollama, vLLM CPU, OpenAI, Anthropic, or Infistar attributes, see
[Choose an attribute provider](#choose-an-attribute-provider).

## Tracking and vLLM with Compose

The repository's
[`docker/compose.attributes.yaml`](https://github.com/mikel-brostrom/boxmot/blob/master/docker/compose.attributes.yaml)
runs the detector, tracker, and attribute client in a BoxMOT container alongside
a vLLM server and PostgreSQL. Detection and inference use the CPU; the default
models are YOLO26n and `Qwen/Qwen3.5-0.8B`, with ByteTrack for tracking. Python dependencies are installed
inside the images, so the host only needs Docker and
[Docker Compose](https://docs.docker.com/compose/install/).

Run the following commands from a repository checkout. Stop any manually started
server occupying port `8001` first. PostgreSQL uses loopback port `5432`; set
`BOXMOT_POSTGRES_PORT` if that port is already in use. The webcam launcher handles
Docker access;
omit `sudo` from the manual commands if your Docker setup does not require it.

### Process a video

Create host directories and copy your input video into `data/video.mp4`:

```bash
mkdir -p data models runs
cp /path/to/video.mp4 data/video.mp4

sudo env BOXMOT_UID="$(id -u)" BOXMOT_GID="$(id -g)" \
  HF_CACHE_DIR="$HOME/.cache/huggingface" \
  docker compose -f docker/compose.attributes.yaml up --build \
  --abort-on-container-exit --exit-code-from boxmot boxmot
```

Compose builds the BoxMOT CPU image and waits for vLLM and PostgreSQL health
checks before starting tracking. The first build, model downloads, and initialization
can take several minutes. `HF_CACHE_DIR` reuses your existing Hugging Face
downloads, including when Docker runs through `sudo`. The health check confirms
server availability; it does not warm up image inference.

The default input is `/inputs/video.mp4` inside the container. Host `data/` is
mounted read-only at `/inputs`; `models/` retains detector/ReID downloads and
`runs/compose/exp/` receives the annotated video, `tracks.jsonl`, and
`attributes.jsonl`. Matching your host UID/GID keeps these outputs writable by
your user. PostgreSQL retains attribute values and history in its `postgres-data`
named volume. When processing finishes, Compose stops all three containers. Press
`Ctrl+C` to stop earlier.

Model caches persist under `models/.cache/`, and Ultralytics settings under
`models/Ultralytics/`. Compose supplies a runtime username and an explicit
PyTorch cache path so your host UID does not need an `/etc/passwd` entry
inside the image.

### Show a Linux webcam preview

Run this from your Linux desktop terminal so `DISPLAY` is set:

```bash
./docker/webcam.sh
```

The launcher creates `data/`, `models/`, and `runs/`, uses your host UID/GID and
Hugging Face cache, and detects the camera's device group. It builds and starts
BoxMOT, vLLM, and PostgreSQL using
[`docker/compose.webcam.yaml`](https://github.com/mikel-brostrom/boxmot/blob/master/docker/compose.webcam.yaml)
for camera and X11/Xwayland display access. Run the launcher as your normal user;
it asks for your sudo password only if Docker requires it.

By default, it maps `/dev/video0` and adds `--source 0 --show`. The container
runs as your host user and shares the X11 socket. If the display rejects that
user, allow it from your desktop terminal
with `xhost +si:localuser:$(id -un)` and rerun the command. Press `q` or `Esc` in
the preview, or `Ctrl+C` in the terminal, to stop the job. Completed attributes and saved outputs remain on
the host. The CPU profile sends the first eligible crop immediately, with its
longest side limited to 640 pixels and no storyboard padding. Attribute labels
appear after the first model response; latency depends on the CPU and available RAM.

### Settings and logs

Set environment variables before the launcher to adjust a run:

```bash
VIDEO_DEVICE=/dev/video2 BOXMOT_RUN_NAME=webcam-test ./docker/webcam.sh
```

For the manual video-file command, pass these variables after `sudo env`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `BOXMOT_RUN_NAME` | `compose` | Results go to `runs/<name>/exp/`; choose a new name for each run to preserve earlier outputs. |
| `BOXMOT_INPUT_DIR` | `../data` | Host input directory, relative to the Compose file; use an absolute path for another location. |
| `BOXMOT_SOURCE` | `/inputs/video.mp4`, or `0` with the webcam override | Source path or camera index inside the container. |
| `BOXMOT_DETECTOR` | `yolo26n` | Detector profile, or path to existing weights inside the container. |
| `BOXMOT_TRACKER` | `bytetrack` | Tracker name. |
| `BOXMOT_REID` | `osnet-x0-25-msmt17` | ReID profile for trackers that use it. |
| `VIDEO_DEVICE` | `/dev/video0` | Host webcam mapped to `/dev/video0`; the launcher detects its device group. |
| `HF_CACHE_DIR` | `$HOME/.cache/huggingface` with the launcher | Host Hugging Face model cache. |
| `BOXMOT_POSTGRES_PORT` | `5432` | PostgreSQL port published on the host's loopback address. |
| `BOXMOT_POSTGRES_USER` | `boxmot` | Local development database user, created when the volume is empty. |
| `BOXMOT_POSTGRES_PASSWORD` | `boxmot-local` | Local development database password, set when the volume is empty. |
| `BOXMOT_POSTGRES_DB` | `boxmot` | Database created when the volume is empty. |

The Compose database credentials are local development defaults. These settings
configure the PostgreSQL container and the BoxMOT container's connection URL
together. Use URL-safe values for the user, password, and database settings.
Changing initialization settings does not change users or passwords in an existing
database volume; manage those changes in PostgreSQL. The webcam launcher preserves
these settings when it invokes Docker through `sudo`.

BoxMOT reads its connection URL from `BOXMOT_ATTRIBUTES_DATABASE_URL`. Compose
sets this to the `postgres` service on port `5432`, independent of the host port.
Host Python or CLI processes must
[set their own database URL](../config/attributes.md#configure-postgresql).

The default detector and ReID profiles download into the writable `models/`
mount. To reuse an existing `models/yolov8n.pt`, pass
`BOXMOT_DETECTOR=models/yolov8n.pt`. For other downloads, use a built-in model
profile or a custom YAML with its checkpoint path under `models/` and a download
URI.

The container connects to `http://vllm-cpu:8000` through
`--attributes-host`, overriding the packaged profile's host URL. This connection
uses Compose networking and does not depend on the published host port.

Inspect startup or inference logs from another terminal:

```bash
sudo env HF_CACHE_DIR="$HOME/.cache/huggingface" \
  docker compose -f docker/compose.attributes.yaml logs --tail 100 -f boxmot vllm-cpu postgres
```

For server-only Compose commands, Ollama, cache settings, and cleanup, see
[Track attributes](../config/attributes.md#docker-compose).

### CPU resource use

The vLLM service uses a 256 MiB KV cache, a 2,048-token context and prefill batch,
and one active request. Image preprocessing is capped at 262,144 pixels,
including its startup probe; the model's default permits 16,777,216 pixels.
Its multimodal processor cache is disabled because successive track crops
are distinct images. Automatic OpenMP thread binding
reserves two CPU cores for tracking and other work. These settings reduce
resource contention; model weights and runtime buffers still need additional RAM.

Keep available host RAM for all three containers and other applications. Sustained
swapping or heavy disk page-ins can make a first response take minutes. Check
host memory and container usage while the request is running:

```bash
free -h
vmstat 1 5
sudo docker stats --no-stream
```

In `free`, inspect the `available` column. In `vmstat`, sustained nonzero `si`
or `so` indicates active swapping; allocated swap alone does not show current
pressure. Stop unnecessary memory-heavy processes or reduce their memory use
before retrying. Raising the request timeout does not make inference faster.

## Choose an attribute provider

`docker/compose.attributes.yaml` provides PostgreSQL plus local Ollama and vLLM
CPU servers. Every attribute run needs PostgreSQL. OpenAI, Anthropic, and
Infistar are hosted model APIs, so those options run only BoxMOT and PostgreSQL
locally.
The commands below run detection and tracking on the CPU and process a video
file. For the existing vLLM webcam workflow, use the
[Linux webcam launcher](#show-a-linux-webcam-preview).

| Provider | Attribute profile | Endpoint used inside the BoxMOT container |
| --- | --- | --- |
| [Ollama](https://ollama.com) | `person-appearance-ollama` | `http://ollama:11434` |
| [vLLM](https://vllm.ai) CPU (x86-64) | `person-appearance-vllm-cpu` | `http://vllm-cpu:8000` |
| [OpenAI](https://developers.openai.com/) | `person-appearance-openai` | `https://api.openai.com/v1` |
| [Anthropic](https://platform.claude.com/) | `person-appearance-anthropic` | `https://api.anthropic.com/v1` |
| [Infistar](https://www.infistar.cc/register?aff=VSCF47E6&ref_source=link) | `person-appearance-infistar` | `https://infistar.ai/v1` |

### Prepare the image and input

Run these examples from the repository root as your Docker-enabled user.
For Infistar, first replace `REPLACE_WITH_YOUR_VISION_MODEL_ID` in
`boxmot/configs/attributes/person-appearance-infistar.yaml` with an exact model
ID available to your API key. Follow the
[Infistar setup](../config/attributes.md#run-with-infistar) to select a model
supporting image inputs and JSON output. Profiles are packaged into the
image, so rebuild it after changing the model ID.

```bash
export COMPOSE_FILE=docker/compose.attributes.yaml
export BOXMOT_UID="$(id -u)" BOXMOT_GID="$(id -g)"
export HF_CACHE_DIR="$HOME/.cache/huggingface"

mkdir -p data models runs "$HF_CACHE_DIR"
cp /path/to/video.mp4 data/video.mp4
docker compose build boxmot
```

Keep these environment variables in the terminal used for the following
commands. Host `data/` is mounted at `/inputs`, and `models/` and `runs/`
preserve downloads and results. PostgreSQL keeps its data in a persistent named
volume. Compose supplies the BoxMOT container's database URL automatically.
Choose one provider below.

### Ollama

Start PostgreSQL and the model server, then download the profile's model:

```bash
docker compose up -d --wait postgres ollama
docker compose exec ollama ollama pull qwen3.5:9b

docker compose run --rm --no-deps --entrypoint boxmot boxmot track \
  --source /inputs/video.mp4 --detector yolo26n --tracker bytetrack --device cpu \
  --attributes person-appearance-ollama \
  --attributes-host http://ollama:11434 \
  --save --save-json --project /opt/boxmot/runs/ollama
```

### vLLM CPU

Start PostgreSQL and the x86-64 CPU service, then wait for initialization:

```bash
docker compose up -d --wait --wait-timeout 1800 postgres vllm-cpu

docker compose run --rm --no-deps --entrypoint boxmot boxmot track \
  --source /inputs/video.mp4 --detector yolo26n --tracker bytetrack --device cpu \
  --attributes person-appearance-vllm-cpu \
  --attributes-host http://vllm-cpu:8000 \
  --save --save-json --project /opt/boxmot/runs/vllm
```

Inside Compose, the client uses service port `8000`; the published host port
`8001` is for clients outside Docker.

### OpenAI

Start PostgreSQL, export your OpenAI API key, and pass it to the BoxMOT container:

```bash
export OPENAI_API_KEY='YOUR_OPENAI_API_KEY'
docker compose up -d --wait postgres

docker compose run --rm --no-deps \
  -e OPENAI_API_KEY --entrypoint boxmot boxmot track \
  --source /inputs/video.mp4 --detector yolo26n --tracker bytetrack --device cpu \
  --attributes person-appearance-openai \
  --save --save-json --project /opt/boxmot/runs/openai
```

The profile uses `gpt-6-luna` through the native OpenAI Responses API. It sends
track crops to OpenAI and keeps the queue and results in local PostgreSQL.
See [OpenAI setup](../config/attributes.md#run-with-openai) for model and schema
configuration. No local model server is needed.

### Anthropic

Start PostgreSQL, export your Anthropic API key, and pass it to the BoxMOT container:

```bash
export ANTHROPIC_API_KEY='YOUR_ANTHROPIC_API_KEY'
docker compose up -d --wait postgres

docker compose run --rm --no-deps \
  -e ANTHROPIC_API_KEY --entrypoint boxmot boxmot track \
  --source /inputs/video.mp4 --detector yolo26n --tracker bytetrack --device cpu \
  --attributes person-appearance-anthropic \
  --save --save-json --project /opt/boxmot/runs/anthropic
```

The profile uses `claude-haiku-5-5` through the native Anthropic Messages API. It
sends track crops to Anthropic and keeps the queue and results in local PostgreSQL.
See [Anthropic setup](../config/attributes.md#run-with-anthropic) for model and schema
configuration. No local model server is needed.

### Infistar

With the configured model ID included in the image, start PostgreSQL, then export
the key and pass it to the BoxMOT container:

```bash
export INFISTAR_API_KEY='YOUR_INFISTAR_API_KEY'
docker compose up -d --wait postgres

docker compose run --rm --no-deps \
  -e INFISTAR_API_KEY --entrypoint boxmot boxmot track \
  --source /inputs/video.mp4 --detector yolo26n --tracker bytetrack --device cpu \
  --attributes person-appearance-infistar \
  --save --save-json --project /opt/boxmot/runs/infistar
```

The client sends track crops to Infistar's HTTPS endpoint and uses the local
PostgreSQL database for its queue and results. No local model server is needed.

### Results and shutdown

These commands override the Compose service's entrypoint to select the
provider, detector, and output directory explicitly. `--no-deps` prevents
Compose from automatically starting dependencies, so each workflow starts
PostgreSQL and any selected local model service explicitly. See the
[Compose run reference](https://docs.docker.com/reference/cli/docker/compose/run/).

Annotated video, `tracks.jsonl`, and `attributes.jsonl` are saved in an `exp/`
run directory under `runs/ollama/`, `runs/vllm/`, `runs/openai/`,
`runs/anthropic/`, or `runs/infistar/`. Choose a different `--name` or `--project`
to preserve outputs from repeated runs.
The BoxMOT container is removed when
the command finishes; any server started with `up -d` stays running.

Stop the servers and remove the Compose containers when finished:

```bash
docker compose --profile '*' down
```

Model downloads and output bind mounts are preserved, along with the Ollama model
volume and PostgreSQL `postgres-data` volume. The database retains run history
after containers stop; new sessions create new run IDs rather than resuming old
work. Adding `-v` to `down` deletes both named volumes, including the database.
See [server logs and settings](../config/attributes.md#logs-settings-and-shutdown)
for startup troubleshooting.

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
