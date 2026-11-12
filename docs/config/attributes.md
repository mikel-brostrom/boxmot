# Track Attributes

Use [Ollama](https://ollama.com), [vLLM](https://vllm.ai), or a compatible hosted API
such as [Infistar](https://www.infistar.cc/register?aff=VSCF47E6&ref_source=link) to describe visible
properties of each track while BoxMOT continues tracking. The optional attribute
enricher sends one crop or combines chronological crops of one track into a
storyboard, then validates the model's answer against your JSON Schema. Track
identities and observation metadata remain application-owned.
Every attribute run requires PostgreSQL for its queue, latest values, and result
history, including runs that use a hosted model service.

## Attribute extraction options

Choose a model service and pass its matching profile to `--attributes`:

| Option and setup | `--attributes` value | Model in the example | Server URL |
| --- | --- | --- | --- |
| [Ollama](#run-with-ollama) | `person-appearance-ollama` | `qwen3.5:9b` | `http://localhost:11434` |
| [vLLM on GPU](#run-with-vllm) | `person-appearance-vllm` | `Qwen/Qwen3.5-9B` | `http://localhost:8000` |
| [vLLM on x86-64 CPU (Docker)](#vllm-on-x86-64-cpu) | `person-appearance-vllm-cpu` | `Qwen/Qwen3.5-0.8B` | `http://127.0.0.1:8001` |
| [Infistar hosted API](#run-with-infistar) | `person-appearance-infistar` | Your selected vision model ID | `https://infistar.ai/v1` |

All four profiles are included in `boxmot/configs/attributes/`. For Infistar,
set the template's model ID and `INFISTAR_API_KEY` as shown in its setup section.
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

# Infistar (after setting the template's model ID and the API key)
uv run --no-sync python -m boxmot.engine.cli track \
  --source 0 --attributes person-appearance-infistar --show --save --save-json
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

Both backends use the same `attributes` extra. For local inference, run the model
server separately; BoxMOT does not install a server or download its models automatically.
Keep vLLM in its own environment: its headless OpenCV dependency can overwrite
BoxMOT's GUI build and disable `--show`. See the
[OpenCV preview fix](../guides/troubleshooting.md#opencv-preview-reports-the-function-is-not-implemented)
if this has already happened.

## Configure PostgreSQL

Attribute extraction requires a PostgreSQL connection. Set
`BOXMOT_ATTRIBUTES_DATABASE_URL` in the environment that runs BoxMOT; there is
no implicit database URL. For a local development database, the repository's
Compose service uses the [official PostgreSQL 17 image](https://hub.docker.com/_/postgres):

```bash
docker compose -f docker/compose.attributes.yaml up -d --wait postgres
export BOXMOT_ATTRIBUTES_DATABASE_URL='postgresql://boxmot:boxmot-local@127.0.0.1:5432/boxmot'
```

These credentials are local development defaults. The database is published only
on the host's loopback address and persists in the `postgres-data` named volume.
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

Compose supplies a URL using `postgres:5432` to its BoxMOT container. Host
processes use `127.0.0.1` and the published port instead. See the
[database settings](../integrations/docker.md#settings-and-logs) for port and
credential overrides, and update the host URL to match any overrides.

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

Both backends work with [prompted EdgeTAM](../trackers/edgetam.md):

```bash
boxmot track --tracker edgetam --source video.mp4 \
  --attributes person-appearance-ollama --save --save-json
```

## Run with Infistar

[Infistar's API](https://doc.infistar.ai/en/api-overview) accepts OpenAI-compatible
Chat Completions requests at `https://infistar.ai/v1/chat/completions`.
BoxMOT connects through its `vllm` client; Infistar runs the model service, so
you do not need a local vLLM installation. The
[PostgreSQL database](#configure-postgresql) is still required.

[Register with Infistar](https://www.infistar.cc/register?aff=VSCF47E6&ref_source=link),
create an API key in **Console → Token Management**, and set it in
the shell that will run BoxMOT. List the models available to that key:

```bash
export INFISTAR_API_KEY='YOUR_INFISTAR_API_KEY'
curl --fail --silent --show-error https://infistar.ai/v1/models \
  -H "Authorization: Bearer $INFISTAR_API_KEY"
```

Choose an exact `data[].id` from the response. Check that the model supports
Chat Completions, image inputs, and JSON mode in
[Infistar's model catalog](https://infistar.ai/pricing).
[Listing a model](https://doc.infistar.ai/en/api-reference/models) confirms access,
but does not guarantee those capabilities. BoxMOT sends a JPEG crop or storyboard as a
base64 `image_url` and sets `temperature: 0`, so the selected route must accept
that image format and sampling setting.

In a source checkout, open
`boxmot/configs/attributes/person-appearance-infistar.yaml` and replace
`REPLACE_WITH_YOUR_VISION_MODEL_ID` in its `model` field with the exact model ID.
The packaged template already includes these connection settings and the same
appearance schema as the other provider profiles:

```yaml
backend: vllm
host: https://infistar.ai/v1
model: REPLACE_WITH_YOUR_VISION_MODEL_ID
api_key_env: INFISTAR_API_KEY
response_format: json_object
think: null
```

`response_format: json_object` selects the
[JSON mode documented by Infistar](https://doc.infistar.ai/en/integration-guides/advanced-features).
BoxMOT includes the schema in the prompt and validates the returned JSON locally.
`think: null` omits vLLM's model-specific `chat_template_kwargs` option.
The API key stays in the named environment variable; do not put it in the YAML.
The client sends an explicit JSON instruction in the system message and uses
`image_url` content for the JPEG, as described in Infistar's
[vision and JSON guidance](https://doc.infistar.ai/en/integration-guides/advanced-features).

For rate limits and temporary server failures, BoxMOT honors a valid
`Retry-After` header before retrying the affected batch. Infistar documents
this behavior in its
[rate-limit guidance](https://doc.infistar.ai/troubleshooting/rate-limits-concurrency).

After setting the model ID and exporting the API key, try a short video from
the repository root before a longer run:

```bash
uv run --no-sync python -m boxmot.engine.cli track \
  --source video.mp4 --attributes person-appearance-infistar --save --save-json
```

This sends track crops to Infistar. Inspect `attributes.jsonl` in the run
directory for completed results or request errors. The profile analyzes COCO
person tracks and collects up to five crops per request. See
[Sampling and completion](#sampling-and-completion) for when analysis starts.

For a package installation or a separate custom configuration, copy the full
`person-appearance-infistar.yaml` template, remove its `id`, set its `model`,
and pass the copy's path to `--attributes`. YAML model names are literal strings;
`api_key_env` is the field that reads an environment variable. The model
placeholder must be replaced before the profile can run.

## Docker Compose

The repository includes
[`docker/compose.attributes.yaml`](https://github.com/mikel-brostrom/boxmot/blob/master/docker/compose.attributes.yaml)
for PostgreSQL and CPU model servers. Install
[Docker Compose](https://docs.docker.com/compose/install/), then run the commands
below from the repository root. Keep BoxMOT in your host environment for webcam
access and `--show` when using these server-only commands. To run BoxMOT in a
container with Ollama, vLLM CPU, or Infistar, follow
[Choose an attribute provider](../integrations/docker.md#choose-an-attribute-provider).
The [full Docker Compose workflow](../integrations/docker.md#tracking-and-vllm-with-compose)
also covers a Linux webcam preview with vLLM.
Start PostgreSQL and one model backend by their service names. The commands
below use the local development database defaults from
[Configure PostgreSQL](#configure-postgresql).

### vLLM on x86-64 CPU

Stop any previously started server using host port `8001` before starting Compose:

```bash
docker compose -f docker/compose.attributes.yaml up -d \
  --wait --wait-timeout 1800 postgres vllm-cpu
export BOXMOT_ATTRIBUTES_DATABASE_URL='postgresql://boxmot:boxmot-local@127.0.0.1:5432/boxmot'
boxmot track --source 0 --attributes person-appearance-vllm-cpu \
  --show --save --save-json
```

This serves `Qwen/Qwen3.5-0.8B` at `http://127.0.0.1:8001`. The matching
`person-appearance-vllm-cpu` profile sends the first eligible crop immediately,
limits its longest side to 640 pixels, and allows 600 seconds for a response.
Single crops are sent without padding or upscaling into a storyboard. Attributes
appear after the first analysis completes; the health check does not run image
inference or guarantee that its first-request initialization has finished.

The Compose server limits its caches and reserves CPU cores for tracking. Keep
enough available host RAM for both processes: sustained swapping can stretch
inference into minutes. See [CPU resource use](../integrations/docker.md#cpu-resource-use)
for the defaults and diagnostic commands.

The server reuses `$HOME/.cache/huggingface`. If Docker requires `sudo`, preserve
your user's cache path explicitly to avoid downloading the model again:

```bash
sudo env HF_CACHE_DIR="$HOME/.cache/huggingface" \
  docker compose -f docker/compose.attributes.yaml up -d \
  --wait --wait-timeout 1800 postgres vllm-cpu
```

### Ollama on CPU

Use an available host port `11434`, start Ollama, and download its model before
running BoxMOT:

```bash
docker compose -f docker/compose.attributes.yaml up -d --wait postgres ollama
docker compose -f docker/compose.attributes.yaml exec ollama ollama pull qwen3.5:9b
export BOXMOT_ATTRIBUTES_DATABASE_URL='postgresql://boxmot:boxmot-local@127.0.0.1:5432/boxmot'
boxmot track --source 0 --attributes person-appearance-ollama --show --save --save-json
```

Ollama stores models in the persistent `ollama-models` named volume. Its health
check confirms that the server responds; the separate `pull` command makes the
model available. This uses the 9B Ollama profile; choose a custom profile and
pull its matching model if you need a smaller model.

### Logs, settings, and shutdown

Downloads and initialization can take several minutes. If `--wait` times out,
the container continues running; inspect its status and startup log:

```bash
docker compose -f docker/compose.attributes.yaml ps -a
docker compose -f docker/compose.attributes.yaml logs --tail 100 -f vllm-cpu
```

Use `ollama` or `postgres` instead of `vllm-cpu` for their logs. Environment variables
`VLLM_PORT`, `OLLAMA_PORT`, and `HF_CACHE_DIR` override the host ports and
Hugging Face cache location. `VLLM_CPU_IMAGE` selects an alternative CPU image.
If you change the server's model or host port, use a custom attribute YAML with
matching `model` and `host` values. When using `sudo`, pass these variables with
`sudo env` as in the cache example above.

Stop and remove the Compose containers with:

```bash
docker compose -f docker/compose.attributes.yaml --profile '*' down
```

This preserves the Hugging Face cache, Ollama model volume, and PostgreSQL data
volume. `down -v` also deletes the local database and its retained attribute
history, as well as the Ollama downloads.

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

`backend` accepts `ollama` (the default) or `vllm`. If `host` is omitted, it
defaults to `http://localhost:11434` for Ollama and `http://localhost:8000` for
vLLM. Override only the connection URL without copying a profile with
`--attributes-host URL`, for example:

```bash
boxmot track --source video.mp4 --attributes person-appearance-vllm-cpu \
  --attributes-host http://127.0.0.1:8002 --show
```

This requires `--attributes` and keeps the selected profile's model, schema,
and sampling settings. For vLLM, `host` can include a trailing `/v1`. To use the
custom schema above with vLLM, change its connection settings to:

```yaml
backend: vllm
model: Qwen/Qwen3.5-9B
host: http://localhost:8000/v1
```

For a server requiring bearer authentication, add `api_key_env: VLLM_API_KEY`
to the profile and set that environment variable before starting BoxMOT. This
setting works with either backend; it contains the variable name, not the key.
If the named variable is missing or empty, initialization fails.

`think: false` disables thinking for Ollama and sets vLLM's
`chat_template_kwargs.enable_thinking` to false. For vLLM this depends on the
served model's chat template supporting that setting, as Qwen3.5 does.
Set `think: null` to omit the backend's thinking option and leave the choice to
the server.

For the `vllm` backend, `response_format` defaults to `json_schema`, which sends
the schema for server-side structured output. Set `response_format: json_object`
for providers supporting JSON mode instead: BoxMOT requests JSON in a system
message, includes the schema in the user prompt, and validates the response
locally. The Ollama backend always sends the schema through its native `format`
field and does not accept `response_format: json_object`.

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
Both send one JPEG image per request with streaming disabled: a single crop or
a storyboard containing multiple crops.

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
