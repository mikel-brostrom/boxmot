# Continuous cameras on Kubernetes

`boxmot stream` runs one network camera with embedded attribute extraction,
shared model serving, and an existing PostgreSQL database. The examples in
`deploy/kubernetes` provide CPU and GPU profiles and save JSON metadata only.
The finite `boxmot track` workflow remains available for files and previews.

## Requirements

- Kubernetes, `kubectl` with [Kustomize](https://kubernetes.io/docs/tasks/manage-kubernetes-objects/kustomization/), and a registry accessible to the nodes.
- PostgreSQL 16 or newer, reachable from camera and maintenance pods. The examples **do not install PostgreSQL**.
- A storage class supporting `ReadWriteMany` for outputs, and `ReadWriteOnce` for model caches. Set `storageClassName` explicitly for your cluster.
- Network routes from pods to the cameras and model server, and access to model repositories during preparation.
- Images containing this change, pinned by digest. Published older BoxMOT images do not contain `stream`.

The CPU example uses the `cli-cpu` BoxMOT target and an x86-64 vLLM CPU image
supporting `Qwen/Qwen3.5-0.8B`. Confirm the serving image supports the model and
BF16 on your CPU; use the image provider's CPU requirements when selecting nodes.
CPU inference is useful for low frame rates and functional validation; measure
your camera workload before choosing resources.

The GPU example uses `cli-gpu` (CUDA 13.0 PyTorch) and a compatible vLLM CUDA
image. Install a compatible NVIDIA driver, container runtime integration, and
the [NVIDIA device plugin](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/).
Each camera requests one GPU and vLLM requests another: one camera needs **two
allocatable GPUs**. They may occupy different nodes. Disable GPU time slicing
for this resource pool if physically separate devices are required. Configure
GPU node selectors, tolerations, and `runtimeClassName` for your cluster.

## Build and publish

Run from the repository root, replacing the registry and release name:

```bash
docker buildx build -f docker/Dockerfile --target cli-cpu \
  -t registry.example.com/boxmot:stream-cpu --push .
docker buildx build -f docker/Dockerfile --target cli-gpu \
  -t registry.example.com/boxmot:stream-gpu --push .
docker buildx imagetools inspect registry.example.com/boxmot:stream-cpu
```

Record the resulting digest. Select and test a vLLM CPU or CUDA image with the
same small-model arguments used by the existing [Compose setup](docker.md).
Mirror that image to your registry if needed and record its digest as well.
Supply registry credentials through your cluster's image-pull configuration.

The base deliberately contains unresolved image names. Use the renderer, which
rejects mutable tags, to produce deployable manifests:

```bash
export CAMERA_IMAGE='registry.example.com/boxmot@sha256:REPLACE_WITH_REAL_DIGEST'
export VLM_IMAGE='registry.example.com/vllm@sha256:REPLACE_WITH_REAL_DIGEST'
uv run --no-sync python -m deploy.kubernetes.render --profile cpu \
  --camera-image "$CAMERA_IMAGE" --vlm-image "$VLM_IMAGE" > /tmp/boxmot.yaml
```

Choose `--profile gpu` with GPU image digests for the GPU overlay. The digest
must contain 64 hexadecimal characters. Keep the rendered manifests with your
release configuration. Raw `kubectl kustomize deploy/kubernetes/overlays/cpu`
and `.../gpu` are also useful for structural review, but still contain image
placeholders.

## Configure cameras, storage, and Secrets

Edit the overlay's `stream.yaml` (the CPU overlay uses `base/stream.yaml`).
Secrets supply URLs; keep credentials out of ConfigMaps:

```yaml
camera_id: camera-01
source_url_env: BOXMOT_CAMERA_URL
attributes: person-appearance-vllm-cpu
attributes_host: http://vllm:8000
output_dir: /outputs
health_port: 8080
retention_days: 7
tracking:
  detector: yolo26n
  tracker: bytetrack
  reid: osnet-x0-25-msmt17
  device: cpu
```

The GPU profile selects BoT-SORT and `device: "0"`, the GPU exposed inside the
camera container. The small attribute profile works with either server device.
For external Ollama, set `attributes: person-appearance-ollama` and
`attributes_host` to its reachable URL, prepare that profile's model on Ollama,
and omit the vLLM Deployment, Service, and PVC from your customized deployment.

The initial workflow accepts network URLs such as RTSP and HTTP. Local video
files and USB camera indices belong to `track`. Camera IDs allow letters,
digits, underscores and hyphens, up to 63 characters. Unknown stream settings
are rejected. Detector/ReID profiles and tracker options use the existing
engine configuration resolution; custom profile paths must exist inside the
container, usually through another ConfigMap mount.

Create the namespace and pre-existing Secrets. The files below should contain
only the corresponding URL, without a trailing newline. Use your secret manager
or a protected temporary directory, and remove the local files afterward:

```bash
kubectl apply -f deploy/kubernetes/base/namespace.yaml
kubectl -n boxmot-streams create secret generic attributes-database \
  --from-file=url=/secure/postgres-url
kubectl -n boxmot-streams create secret generic camera-01 \
  --from-file=url=/secure/camera-url
```

The PostgreSQL role needs permission to create/upgrade `boxmot_attributes` and
read/write its tables and sequences. Use a TLS PostgreSQL URL appropriate to
your deployment. The database must preserve state across pod replacements.

Default PVC requests are 20 GiB per camera for output, 10 GiB for its detector
and ReID cache, and 20 GiB for the shared VLM cache. Output storage must implement
durable file and directory `fsync` and atomic rename. Camera workers run as
UID/GID 10001; ensure volumes are writable by that identity. The output
maintenance Job shares the output claim, so it requires RWX across nodes.
RWO-only installations need their own maintenance scheduling/storage design.

Size output storage as measured JSON bytes/second × 604800 seconds plus
headroom for open segments and replay. Seven-day retention is a minimum age
for deletion, not a hard disk quota: unacknowledged work is protected. PostgreSQL
needs room for at most 256 MiB of queued crops per camera, retained events,
indexes and WAL. Monitor both database and PVC free space.

## Prepare models

The camera profiles download missing detector/ReID checkpoints into
`/opt/boxmot/models`; vLLM uses `HF_HOME=/models/huggingface`. Persist these caches
before admitting production traffic. The startup probes allow up to thirty
minutes for loading, including initial downloads. Increase this deliberately
if preparation needs longer.

For restricted networks, stage the checkpoints named by
`boxmot/configs/detectors/yolo26n.yaml` and
`boxmot/configs/reid/osnet-x0-25-msmt17.yaml` into the camera model PVC using a
temporary preparation pod. Stage a complete Hugging Face snapshot of
`Qwen/Qwen3.5-0.8B` into the VLM cache PVC using the selected serving image's
`hf download Qwen/Qwen3.5-0.8B` command with that `HF_HOME`. Mount the same claims
and use the camera UID for camera files. Stop the preparation pods before
starting Deployments using RWO caches. Pin the model revision when preparing
your release; the image digest alone does not pin downloaded model weights.

For private model repositories, provide `HF_TOKEN` through a separate Secret.
Prepare caches and verify inference using the exact images and hardware you
will deploy. No model weights or credentials are included in the repository.

## Upgrade the schema and install

The additive schema version 2 preserves version 1 runs, crops and outcomes. New
databases can initialize automatically. An existing version 1 schema requires
an explicit upgrade; application startup will not reset or silently migrate it.

1. Back up PostgreSQL and stop all existing attribute clients, including finite `track` jobs. Suspend maintenance during the upgrade.
2. Create the namespace and Secrets, then render the dedicated upgrade Job.
3. Wait for the Job to succeed before starting the new camera and maintenance workloads.

```bash
uv run --no-sync python -m deploy.kubernetes.render --profile cpu \
  --camera-image "$CAMERA_IMAGE" --vlm-image "$VLM_IMAGE" --upgrade > /tmp/boxmot-upgrade.yaml
kubectl apply -f /tmp/boxmot-upgrade.yaml
kubectl -n boxmot-streams wait --for=condition=complete \
  job/attribute-schema-upgrade --timeout=900s
kubectl apply --dry-run=server -f /tmp/boxmot.yaml
kubectl apply -f /tmp/boxmot.yaml
kubectl -n boxmot-streams rollout status deployment/vllm --timeout=1800s
kubectl -n boxmot-streams rollout status deployment/camera-01 --timeout=1800s
```

The upgrade can also run directly with `boxmot stream-db upgrade`. It is
idempotent. Before another upgrade, archive the previous Job's logs and delete
that completed Job so Kubernetes can create a new one with changed images.
The upgrade allows ten minutes per SQL statement and thirty seconds to acquire
locks; the Job has a fifteen-minute deadline. Plan a longer maintenance window
and adjust these limits before upgrading unusually large histories.

Keep previous image digests and configuration for rollback. Version 1 clients
reject the version 2 schema: rolling back to them requires restoring the
matching database backup, not deleting or downgrading the live schema. Rollbacks
between releases that both support version 2 can retain the database.

## Ownership and failure behavior

Each camera uses one Deployment replica with `Recreate`. PostgreSQL leases
renew every five seconds and expire after thirty seconds according to database
time. A dedicated connection renews ownership; streaming database transactions
check the current owner and epoch. Failed renewal stops admission and publishing
and terminates the worker. Acquisition cancels abandoned pending/running jobs,
releases their crops, and retains completed outcomes. Late obsolete responses
cannot commit.

This controls persisted work. During network partitions, two processes may
temporarily access the physical camera. Unique session directories prevent
writers from appending to each other's files; a file segment from an obsolete
owner can exist even though its database acknowledgment is fenced.

Capture runs in a killable subprocess with one replaceable frame slot, bounded
to 32 MiB by default (`max_frame_bytes`). Source indices and timestamps survive
dropped frames. OpenCV's FFmpeg backend receives ten-second
[open/read timeouts](https://docs.opencv.org/4.11.0/d4/d15/group__videoio__flags__base.html).
The supervisor replaces stuck processes and reconnects indefinitely with
exponential backoff, jitter and a thirty-second cap. Every reconnection resets
the tracker and starts a new attribute session.

`/healthz` reports supervisor progress; `/readyz` additionally requires ownership
and recently completed frames. Camera outages make the pod unready while it
reconnects. Database and output failures terminate the process. Model failures
produce explicit failed events after bounded retries while tracking continues.
Model availability is not reported as camera availability.

SIGTERM/SIGINT stop capture and submissions, cancel inference, and flush
committed results. A twenty-second watchdog bounds application shutdown; pod
termination grace is thirty seconds. If cleanup cannot finish, unacknowledged
events remain in PostgreSQL for the next owner or maintenance.

## Outputs and retention

Files live under `/outputs/<camera_id>/<session_id>/`, with separate
`tracks-*.jsonl` and `attributes-*.jsonl` segments. Identity is the combination
of camera, session, generation and track ID. Attribute values appear in the
attribute event segments instead of being repeated in every track frame.
Capture wall time is exported as
`captured_at_unix_s`; `timestamp_s` retains the source's media timing when
available. No videos or image files are exported; crops are transient database
queue payloads and disappear after completion or cancellation.

Segments rotate before the next write after 64 MiB or one hour. Open segments
end in `.jsonl.open`; clean close synchronizes and renames them. Read complete
newline-terminated records from orphaned `.open` files after a crash and ignore
an incomplete final line. Synchronized records may exist in either suffix.

Attribute events have stable `event_id` values. Export writes and synchronizes
the JSONL batch before acknowledging it in PostgreSQL. A crash after sync and
before acknowledgment replays the same IDs into new segments; consumers **must
deduplicate by event ID**. Track metadata has no replay queue; its
`camera_id/session_id/sample_id` provenance identifies each processed frame.

Each camera defaults to 1,000 outstanding attribute batches and 256 MiB of
queued crops (`max_pending_jobs`, `max_crop_bytes`). Overflow cancels the oldest
pending batches with an explicit capacity reason. Accepted work is never
silently dropped. These limits bound the inference backlog; events waiting for
successful export remain durable.

`attribute-maintenance` runs every five minutes. It abandons expired sessions
and removes acknowledged history older than seven days, preserving active track
state, pending jobs and unacknowledged events. `camera-01-output-maintenance`
runs hourly, including when that camera never restarts. For an offline camera,
it briefly acquires ownership and exports abandoned outcomes, so cleanup does
not depend on another camera connection. A worker restarting during that short
lease may need to retry acquisition. Maintenance conservatively skips file
deletion while any camera event remains unacknowledged and protects open files
while a session is active. The running camera also prunes closed historical
segments hourly. Monitor failed maintenance Jobs.

## Add cameras and validate capacity

Create a separate Deployment, ConfigMap, camera Secret, output claim, model
claim, and output-maintenance CronJob for each camera. Change names, selectors,
`camera_id`, Secret references and claim references consistently. Keep one
replica per camera; scaling that Deployment creates competing owners. Share the
vLLM Service and database, and retain one global database-maintenance CronJob.
Increase serving capacity based on measured backlog age before adding cameras.

The health JSON and thirty-second status logs include processed/dropped frames,
reconnects, pending batches, queued crop bytes, oldest batch age, and mean frame
time over a fixed 300-frame window. Streaming does not retain lifetime sets of
track IDs or profiler samples. Supplied ByteTrack and BoT-SORT implementations
bound removed-track archives; validate memory behavior separately for custom
trackers.

Before production, run a CPU soak with representative cameras for at least
24 hours. Observe pod RSS, frame progress, queue age, database/PVC growth, and
maintenance completion. Interrupt the camera, model service and database, then
delete a camera pod during inference and verify fresh sessions and event replay.
Test competing owners and storage failures. Check that queues stay within their
caps and memory stabilizes after warmup. Test retention against aged fixtures
before relying on seven-day cleanup.

Render and schema-validate both overlays, then use server dry-run against the
target cluster. On GPU hardware verify that camera and vLLM receive distinct
devices and both models start. Unit tests use synthetic frames, real PostgreSQL
transactions and injected models; they do not replace these cluster checks.

## Command reference

::: mkdocs-click
    :module: boxmot.engine.commands.stream
    :command: stream
    :prog_name: boxmot stream
    :depth: 1
