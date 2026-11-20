# Track Attributes

Use [Ollama](https://ollama.com) or [vLLM](https://vllm.ai) to describe visible
properties of each track while BoxMOT continues tracking.
The optional attribute enricher sends one crop or combines chronological crops
of one track into a storyboard, then validates the model's answer against your
JSON Schema. Track identities and observation metadata remain application-owned.
Every attribute run requires PostgreSQL for its queue, latest values, and result
history.

For continuous network cameras, use the [Kubernetes streaming workflow](../integrations/kubernetes.md).
It adds ownership leases, bounded queues, durable event replay and seven-day
retention. Existing attribute databases require an explicit `boxmot stream-db upgrade`
before running clients from this release. Follow the backup, stop-client and
upgrade procedure for [Helm](../integrations/kubernetes.md#upgrade-the-schema-and-install).

## Attribute extraction options

Choose a model service and pass its matching profile to `--attributes`:

| Option and setup | `--attributes` value | Model in the example | Server URL |
| --- | --- | --- | --- |
| [Ollama](#run-with-ollama) | `person-appearance-ollama` | `qwen3.5:9b` | `http://localhost:11434` |
| [vLLM on GPU](#run-with-vllm) | `person-appearance-vllm` | `Qwen/Qwen3.5-9B` | `http://localhost:8000` |
| [vLLM on x86-64 CPU (Helm)](#vllm-on-x86-64-cpu) | `person-appearance-vllm-cpu` | `Qwen/Qwen3.5-0.8B` | `http://127.0.0.1:8001` |

All three profiles are included in `boxmot/configs/attributes/`.
Attribute profiles select the model service; use `--device` separately to select
the device for detection and tracking.

### Quick commands from a source checkout

First [install the client dependencies](#install-the-client-dependencies),
[configure PostgreSQL](#configure-postgresql), and complete the setup linked in
the table. Keep PostgreSQL and the selected model service available, then
run one of these commands from the repository root:

```bash
# Ollama
uv run --no-sync python -m boxmot.engine.cli track \
  --source 0 --attributes person-appearance-ollama --show --save --save-json

# vLLM on GPU
uv run --no-sync python -m boxmot.engine.cli track \
  --source 0 --attributes person-appearance-vllm --show --save --save-json

# vLLM on CPU
uv run --no-sync python -m boxmot.engine.cli track \
  --source 0 --attributes person-appearance-vllm-cpu --show --save --save-json

```

`--source 0` selects your webcam; replace `0` with a video path to process a
file. `--show` displays the preview, `--save` saves the annotated video, and
`--save-json` adds track records with available attributes to `tracks.jsonl`.
Every attribute-enabled run also writes `attributes.jsonl` in its run directory;
see [Outputs](#outputs) for details about when results become available.

## Install the client dependencies

Install the optional Python dependencies in your BoxMOT environment:

```bash
boxmot install --extra attributes
```

For a source checkout, add `--extra attributes` to your existing `uv sync`
command and repeat your selected extras. For example:

```bash
uv sync --extra cpu --extra yolo --extra evolve --extra service --extra attributes \
  --group dev --group test --group docs
```

Replace `--extra cpu` with `--extra cu130` for CUDA 13.0. Continue using
`uv run --no-sync` for commands from the checkout to preserve that profile.

Both backends use the same `attributes` extra; no provider SDK is required.
Run the model server separately; BoxMOT does not install a server or download
its models automatically.
Keep vLLM in its own environment: its headless OpenCV dependency can overwrite
BoxMOT's GUI build and disable `--show`. See the
[OpenCV preview fix](../guides/troubleshooting.md#opencv-preview-reports-the-function-is-not-implemented)
if this has already happened.

## Configure PostgreSQL

Attribute extraction requires a PostgreSQL connection. Set
`BOXMOT_ATTRIBUTES_DATABASE_URL` in the environment that runs BoxMOT; there is
no implicit database URL. For a local development database, run the
[official PostgreSQL 17 image](https://hub.docker.com/_/postgres):

```bash
docker run --detach --name boxmot-postgres \
  --publish 127.0.0.1:5432:5432 \
  --env POSTGRES_USER=boxmot --env POSTGRES_PASSWORD=boxmot-local \
  --env POSTGRES_DB=boxmot \
  --volume boxmot-postgres-data:/var/lib/postgresql/data \
  --health-cmd='pg_isready -U boxmot -d boxmot' \
  --health-interval=5s --health-timeout=3s --health-retries=20 postgres:17

docker exec boxmot-postgres pg_isready -U boxmot -d boxmot
export BOXMOT_ATTRIBUTES_DATABASE_URL='postgresql://boxmot:boxmot-local@127.0.0.1:5432/boxmot'
```

Wait for `pg_isready` to report that the server accepts connections before
starting BoxMOT; repeat it while the new database initializes. These credentials
are for local development. The database is published only on the host's loopback
address and persists in the `boxmot-postgres-data` named volume. Stop it with
`docker stop boxmot-postgres` and restart with `docker start boxmot-postgres`.
Removing the container leaves the named volume intact. Environment credentials
initialize an empty volume only; changing them does not update existing users.

For another database, set the URL to its host, database, and credentials. The
database user must be able to create and use the `boxmot_attributes` schema and
its tables. Percent-encode reserved characters in URL credentials.

The profile setting `database_url_env` selects the environment variable to read;
its default is `BOXMOT_ATTRIBUTES_DATABASE_URL`. Store the variable name in YAML
and keep the connection URL in the environment:

```yaml
database_url_env: BOXMOT_ATTRIBUTES_DATABASE_URL
```

The same database can hold multiple runs. Queue entries, latest values, and event
history are isolated by run ID within `boxmot_attributes`. Completed jobs release
their crop data; latest values and history remain after a clean close. A new
session creates a new run ID and does not resume an earlier run's pending work.

Host processes use the published loopback port above. Kubernetes clients use
the database service address from their Secret; see the
[Helm database setup](../integrations/kubernetes.md#start-on-minikube).
A container's `127.0.0.1` does not address its host or another container.
Use the database hostname reachable from the process running BoxMOT.

## Run with Ollama

Install [Ollama](https://docs.ollama.com/quickstart) separately, start its server with
`ollama serve` if it is not already running, and download the example model.
Complete the [PostgreSQL setup](#configure-postgresql) before starting tracking:

```bash
ollama pull qwen3.5:9b
boxmot track --source video.mp4 --attributes person-appearance-ollama --save --save-json
```

The built-in `person-appearance-ollama` profile describes visible upper-body and leg
clothing, footwear, headwear, carried items, and pose, with unknown values and
confidence for each field. It uses
[`qwen3.5:9b`](https://ollama.com/library/qwen3.5:9b) at
`http://localhost:11434` and filters for class ID `0`, the person class in COCO.
Use a custom profile when your detector's class map differs.

## Run with vLLM

Set up [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/) in a
separate environment or container with a vision-language model supported by
your installed vLLM version. Follow the
[Qwen3.5 model instructions](https://huggingface.co/Qwen/Qwen3.5-9B#serving-qwen35)
for the example model. In that server environment, start:

```bash
vllm serve Qwen/Qwen3.5-9B --port 8000 --reasoning-parser qwen3 --max-model-len 8192
```

Adjust the context limit and server deployment to fit your hardware. Keep image
input enabled. Wait for model downloads and startup warmup to finish, then check
readiness and the served model name from the BoxMOT environment:

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/models
```

Use the server address and host port from your profile if they differ. Once both
checks succeed and [PostgreSQL is configured](#configure-postgresql), run BoxMOT
from its own environment:

```bash
boxmot track --source video.mp4 --attributes person-appearance-vllm --save --save-json
```

The `person-appearance-vllm` profile uses `Qwen/Qwen3.5-9B` at
`http://localhost:8000`, with the same appearance schema and person-class filter
as `person-appearance-ollama`. The model name must match the model served by vLLM;
use a custom profile if you configure a different served model name.

All backends work with [prompted EdgeTAM](../trackers/edgetam.md):

```bash
boxmot track --tracker edgetam --source video.mp4 \
  --attributes person-appearance-ollama --save --save-json
```

## vLLM on x86-64 CPU

The [Helm deployment](../integrations/kubernetes.md) provides the tested CPU
vLLM configuration, pinned serving image and model revision, persistent cache,
and resource limits. Install it in Minikube or your Kubernetes environment.
To use that model server from a host-side BoxMOT job, forward its service in a
separate terminal (replace the release and namespace names if different):

```bash
kubectl --namespace boxmot-helm port-forward service/boxmot-vllm 8001:8000
```

Complete [PostgreSQL setup](#configure-postgresql) for the host client, then run:

```bash
curl --fail http://127.0.0.1:8001/health
curl --fail http://127.0.0.1:8001/v1/models
boxmot track --source 0 --attributes person-appearance-vllm-cpu \
  --show --save --save-json
```

The default server serves `Qwen/Qwen3.5-0.8B`. Its matching
`person-appearance-vllm-cpu` profile sends the first eligible crop immediately,
limits its longest side to 640 pixels, and allows 600 seconds for a response.
Single crops are sent without padding or upscaling into a storyboard. Attributes
appear after the first analysis completes; server health does not prove that
first-request model compilation has finished.

Keep sufficient CPU and RAM for model serving and tracking. The bundled server
is shared by every configured camera and any host clients, so additional jobs
increase its queue. See the [deployment validation record](https://github.com/mikel-brostrom/boxmot/blob/master/deploy/helm/VALIDATION.md)
for measured inference times and test limits. CPU serving does not change the
client's `--device` setting.

The chart owns its model-cache PVC and does not mount the host's personal
Hugging Face cache. Inspect initialization through Kubernetes:

```bash
kubectl --namespace boxmot-helm get pods
kubectl --namespace boxmot-helm logs deployment/boxmot-vllm --tail=100
```

Use the [Ollama setup](#run-with-ollama) for a separately managed Ollama server.
Model-server changes need matching `model` and `host` fields in a custom
attribute profile. Keep image digests and model revisions with deployment
values, and rerun real inference before promoting an update.

## Customize a profile

`--attributes` accepts a built-in profile name or a YAML path. Packaged profiles
live in `boxmot/configs/attributes/`. Save this example as `visible-color.yaml`:

```yaml
backend: ollama
model: qwen3.5:9b
host: http://localhost:11434
database_url_env: BOXMOT_ATTRIBUTES_DATABASE_URL
prompt: >-
  Describe the main visible clothing color of this tracked person. Return
  Unknown with low confidence when clothing is hidden or ambiguous. Use
  agreement across the chronological crops. Return only the schema fields.
schema:
  type: object
  additionalProperties: false
  required: [color, confidence]
  properties:
    color:
      type: string
      description: Main clothing color, or Unknown when not visible.
    confidence:
      type: number
      minimum: 0
      maximum: 1
class_ids: [0]
display_fields: [color]
sample_every: 30
batch_size: 5
request_timeout_s: 120
retries: 2
max_crop_side: 640
jpeg_quality: 90
think: false
```

```bash
boxmot track --source video.mp4 --attributes visible-color.yaml --save-json
```

`backend` selects the request protocol. Defaults and supported output formats are:

| `backend` | Default `host` | Default API key variable | `response_format` choices |
| --- | --- | --- | --- |
| `ollama` (default) | `http://localhost:11434` | None | `json_schema` |
| `vllm` | `http://localhost:8000` | None | `json_schema`, `json_object` |

Override only the connection URL without copying a profile with
`--attributes-host URL`, for example:

```bash
boxmot track --source video.mp4 --attributes person-appearance-vllm-cpu \
  --attributes-host http://127.0.0.1:8002 --show
```

This requires `--attributes` and keeps the selected profile's model, schema,
and sampling settings. For vLLM, `host` can include a
trailing `/v1`. To use the custom schema above with vLLM, change its connection
settings to:

```yaml
backend: vllm
model: Qwen/Qwen3.5-9B
host: http://localhost:8000/v1
```

`api_key_env` selects an optional environment variable for bearer
authentication; it contains the variable name, not the key. For example, set
`api_key_env: VLLM_API_KEY` for a vLLM server requiring authentication. If the
selected variable is missing or empty, initialization fails.

`think: false` disables thinking for Ollama and sets vLLM's
`chat_template_kwargs.enable_thinking` to false. For vLLM this depends on the
served model's chat template supporting that setting, as Qwen3.5 does.
Set `think: null` to omit the backend's thinking option and leave the choice to
the server.

`response_format` defaults to `json_schema`, which sends the schema for
server-side structured output. For vLLM models supporting JSON mode,
`response_format: json_object` requests JSON in the prompt and validates the
response against the schema locally. Ollama accepts only `json_schema`.

Omit `class_ids` to analyze all classes. `display_fields` selects schema fields
for video labels and supports dotted paths for nested values; an empty list
disables attribute labels. The prompt and schema apply to every eligible track
in the run. Model responses cannot set a track ID or redirect attributes to
another track. Schemas must have an object root; any schema references must be
local. Returned JSON is validated locally before it becomes a result. Ollama
requests use [`/api/generate`](https://docs.ollama.com/api/generate) with the
schema in `format`. vLLM requests use `/v1/chat/completions` with a base64
[image URL](https://docs.vllm.ai/en/latest/features/multimodal_inputs/#online-serving)
and the configured
[`response_format`](https://docs.vllm.ai/en/latest/features/structured_outputs/).
Both backends send one JPEG image per request with streaming disabled: a single
crop or a storyboard containing multiple crops.

## Sampling and completion

Sampling starts at a track's first eligible observation and repeats every
30 processed frames by default. A current detection or a nonempty current mask
is required; prediction-only rows keep their previous attributes. Crops come
from original frames, with clipped box bounds, enclosing bounds for OBBs, or
current mask bounds. OBB crops retain their image orientation for pose analysis.

One worker analyzes batches of up to five crops. Compressed crops and pending
work live in PostgreSQL, isolated by run ID. Each track can have one running batch
and one pending batch; newer pending samples replace older pending samples without
moving that track to the back of the queue. This limits memory usage while
preserving work for short-lived tracks. Partial batches are flushed when a
track disappears, the sequence resets, or input ends.

For a continuously visible track first sampled at frame 0, the default
`batch_size: 5` and `sample_every: 30` collect frames 0, 30, 60, 90, and 120 before
starting analysis. Set `batch_size: 1` in a custom profile for an earlier first
request, with less temporal evidence for pose or activity. A single crop keeps
its encoded dimensions, bounded by `max_crop_side`, without a title, padding,
or upscaling. Multiple crops are composed into a labeled chronological storyboard.

At normal EOF, tracking stops accepting frames and drains outstanding analysis,
showing the remaining backlog. This can take longer than video processing;
there is no overall drain deadline. Transient request failures receive two
retries by default, with exponential backoff capped at 30 seconds. A valid
`Retry-After` header on HTTP 429 or server errors can extend the delay; other
tracks can continue while the affected batch waits.
Invalid responses produce failure events and leave previous valid attributes
available. Each eligible track receives an analysis attempt or an explicit
failure/cancellation outcome, not necessarily a successful description.

Interrupting a run cancels outstanding work and preserves completed events.
Sequence generations isolate reused track IDs, including results arriving
after a reset. A temporarily absent track returning in the same sequence keeps
its identity. Closing a session retains the run's latest values and event history
in PostgreSQL. Starting another session creates a new run ID; it does not claim
or resume work from earlier runs.

## Outputs

Attribute-enabled tracking writes `attributes.jsonl` in the run directory even
without `--save-json`. Events record run/sequence/track identity, source samples,
the model, and validated attribute values, or an explicit failure/cancellation.
`TrackRun.attributes_path` points to this journal.
PostgreSQL also retains the run's latest values and event history independently
of the exported JSONL files.

With `--save-json`, `tracks.jsonl` also contains the latest available attributes
for each frame. Saved video and previews display the selected fields on wrapped
lines below each track label, respecting label visibility controls. Fields appear
when the first analysis completes; collecting the configured crop batch and
waiting for the model response can delay the first label. Video and frame snapshots
use results available at that moment; later results remain in
`attributes.jsonl` and do not rewrite earlier frames. Attributes do not change
association, packed NumPy track layouts, or MOT text outputs.

If the preview has no attribute fields, check server readiness as above and
inspect the run's `attributes.jsonl` for failed requests. Failures also produce
terminal warnings; consecutive identical errors are suppressed there, while every
failure remains in the journal. An empty journal can mean the first batch is still
collecting crops or waiting for inference. The live tracking status shows
completed analyses, pending batches, and failed analyses on a separate line.
Pending batches include crop collection, queued requests, and active inference;
completed counts include refreshes of previously analyzed tracks.

## Python

Set `BOXMOT_ATTRIBUTES_DATABASE_URL` before starting your Python process, using
the same [PostgreSQL setup](#configure-postgresql) as the CLI. A custom
`AttributeConfig` can select a different variable through `database_url_env`.

For an engine-owned run, pass `attributes="person-appearance-ollama"`,
`attributes="person-appearance-vllm"`, or your YAML path to
`boxmot.engine.tracking.workflow.run_track`. It also accepts an enricher
instance and owns that instance's completion and shutdown. When supplying a
profile name, YAML path, or configuration, `attributes_host="http://vllm-cpu:8000"`
overrides its server URL. This override cannot be used with a caller-owned
enricher instance.

For caller-owned frames, manage the enricher explicitly:

```python
from boxmot.attributes import create_attribute_enricher
from boxmot.pipelines import TrackingPipeline

# detector and tracker are configured components; frames yields canonical Frame values.
with create_attribute_enricher("person-appearance-ollama") as attributes:
    pipeline = TrackingPipeline(
        detector=detector, tracker=tracker, attributes=attributes,
    )
    for frame in frames:
        result = pipeline.step(frame)
        for track_id, description in result.attributes.items():
            print(track_id, description.values, description.samples)
        for event in attributes.poll():
            consume_event(event)
    for event in attributes.finish():
        consume_event(event)
```

`consume_event` is your application's event handler. `result.attributes` is a
detached snapshot keyed by track ID; subsequent completions cannot mutate an
earlier result. `poll()` returns completed events without waiting; consuming
the `finish()` iterator drains the remaining work, including late results.

Standalone tracker loops can call `attributes.update(frame, tracks)` with
canonical `Frame` and `Tracks` values, then use the same event API. Reset the
pipeline between sequences as usual; it also resets the attribute generation.
Attribute extraction is optional and its dependencies are loaded only when
enabled. CLI help and ordinary tracking do not require the `attributes` extra
or a database connection.
