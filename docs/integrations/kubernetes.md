# Continuous cameras on Kubernetes

`boxmot stream` runs one network camera with embedded attribute extraction,
shared model serving, and PostgreSQL. The Helm chart in `deploy/helm/boxmot`
generates one worker and output-maintenance job per camera, with optional vLLM
and local PostgreSQL. It saves JSON metadata only. The finite `boxmot track`
workflow remains available for files and previews.

Helm is the supported deployment path for Minikube, AKS and EKS. Compose and
Kustomize manifests have been retired. Existing installations keep their data;
see [Migrate an existing deployment](#migrate-an-existing-deployment) before switching.

## Requirements

- Kubernetes, `kubectl`, Helm with OCI registry support, and a registry accessible to the nodes.
- PostgreSQL 16 or newer, reachable from camera and maintenance pods. Use the optional bundled database for local validation, and an externally managed database for production.
- RWX output storage with durable file and directory `fsync` and atomic rename; RWO model caches. The single-node Minikube example uses RWO outputs.
- Network routes from pods to cameras and the model server, plus model-repository access during preparation.
- Images from this implementation, pinned by digest. Older BoxMOT images may not contain `stream`.

The CPU defaults use the `cli-cpu` BoxMOT target, a pinned x86-64 vLLM CPU
image, and a pinned revision of `Qwen/Qwen3.5-0.8B`. Verify the serving image's
CPU instruction requirements on your nodes. CPU inference is useful for
functional validation and low frame rates; measure your workload before sizing
production resources.

## Build once and identify a release

Build the camera image from the code you intend to deploy. Replace the registry
and candidate tag, then record the resulting digest:

```bash
docker buildx build -f docker/Dockerfile --target cli-cpu \
  -t registry.example.com/boxmot:stream-candidate --push .
docker buildx imagetools inspect registry.example.com/boxmot:stream-candidate
```

The chart requires `image: repository@sha256:<64 hex digits>`; it has no default
camera image. Save that reference in a release values file:

```yaml
# release-images.yaml: replace this placeholder with your published digest.
image: registry.example.com/boxmot@sha256:YOUR_64_HEX_DIGEST
```

Record any overrides to `vllm.image`, `vllm.model`, `vllm.revision` and
`postgres.image` in the same file. The chart pins its default serving/database
images and model revision. An image digest does not pin downloaded weights.
Keep these five identities with each deployment:

| Identity | Purpose |
| --- | --- |
| Chart version | Kubernetes templates and values interface; versioned independently. |
| `appVersion` | Informational BoxMOT application version; never substitutes for an image digest. |
| Image digests | Exact camera, serving and optional database container artifacts. |
| Model revision | Exact Hugging Face model snapshot served by vLLM. |
| Environment values commit | Camera IDs, resources, storage, scheduling and Secret references. |

Promote the same tested digests from local validation to staging and production.
Environment tags are convenient pointers; keep digest references in Helm values.
The image-promotion workflow retags the BoxMOT image from a successful
integration run and retains the serving/model identities without rebuilding.
It does not apply a Helm release to a cloud cluster.

Maintainers can start CPU validation and then promote its successful run:

```bash
gh workflow run helm.yml -f image='boxmot/boxmot@sha256:YOUR_64_HEX_DIGEST'
gh workflow run promote-images.yml -f validation_run_id=SUCCESSFUL_RUN_ID \
  -f environment=staging
```

Use `environment=production` for the subsequent promotion. The workflow records
the tested image/model identities and chart source commit, and verifies that
the promoted image digest matches. The chart commit does not establish the
source commit of an independently supplied image; retain its build provenance
as well. Promotion requires the repository's Docker Hub credentials; configure
GitHub environments if deployment approval gates are required.

Scheduled validation needs repository variable `BOXMOT_HELM_IMAGE`, with optional
`BOXMOT_HELM_VLLM_IMAGE` and `BOXMOT_HELM_RUNNER`. Its runner must have Docker,
8 CPUs, at least 16 GiB RAM and 25 GiB free disk for the isolated cluster and
model downloads. GPU execution remains a hardware-specific check.

Chart releases use tags `boxmot-chart-X.Y.Z` and publish to
`oci://ghcr.io/<repository-owner>/charts/boxmot`. Select an explicit chart
version when consuming a published package:

```bash
helm upgrade --install boxmot oci://ghcr.io/YOUR_OWNER/charts/boxmot \
  --version X.Y.Z --namespace boxmot-streams --create-namespace \
  -f environment.yaml -f release-images.yaml --wait --timeout 30m
```

Do not assume a chart package exists until its release workflow has succeeded.
The publishing workflow requires the tag to match `Chart.yaml` and a commit on
the default branch; it refuses to replace an existing chart version. Increase
the chart version when releasing template, default-value or application-version
metadata changes. Package release preparation synchronizes `appVersion` with
the package version; it leaves the chart version independently managed.
The source-chart commands below work before publication. Registry credentials
belong in the cluster's image-pull configuration, referenced with
`imagePullSecrets` when necessary.

## Start on Minikube

Use an existing cluster or create a separate development profile. The example
budgets suit an initial one-camera trial on an 8-CPU, 10-GiB node; reserve host
memory and disk for the cluster, downloaded weights and retained metadata.

```bash
minikube start -p boxmot-helm --cpus 8 --memory 10240
kubectl create namespace boxmot-helm
cp deploy/helm/examples/minikube.yaml environment.yaml
```

Create the Secrets outside Helm. In a protected local directory, prepare files
containing only the camera URL, PostgreSQL password, and PostgreSQL URL, without
trailing newlines. For release `boxmot`, the bundled database URL is
`postgresql://boxmot:PERCENT_ENCODED_PASSWORD@boxmot-postgres:5432/boxmot`; the
password must match the separate password file. Use your secret manager in
production, and remove local secret files after creation.

```bash
kubectl -n boxmot-helm create secret generic attributes-database \
  --from-file=url=/secure/postgres-url --from-file=password=/secure/postgres-password
kubectl -n boxmot-helm create secret generic camera-01 \
  --from-file=url=/secure/camera-url
helm lint deploy/helm/boxmot -f environment.yaml -f release-images.yaml
helm template boxmot deploy/helm/boxmot --namespace boxmot-helm \
  -f environment.yaml -f release-images.yaml > /tmp/boxmot-helm.yaml
kubectl -n boxmot-helm apply --dry-run=server -f /tmp/boxmot-helm.yaml
helm upgrade --install boxmot deploy/helm/boxmot --namespace boxmot-helm \
  -f environment.yaml -f release-images.yaml --wait --timeout 30m
kubectl -n boxmot-helm rollout status deployment/boxmot-camera-01 --timeout=1800s
```

An existing schema version 1 database needs the [explicit upgrade](#upgrade-the-schema-and-install)
before camera clients start. A new database initializes automatically. The
PostgreSQL role must be able to create the `boxmot_attributes` schema and
read/write its tables and sequences.

Network camera URLs include RTSP and HTTP. USB camera indices and local files
belong to `track`; publish a built-in webcam as a network stream reachable from
the cluster before using it here. The CPU integration harness supplies its own
bundled fixture camera for repeatable validation.

## Configure cameras and dependencies

Camera map keys are stable IDs, using lowercase DNS labels of at most 20
characters. Release names or `fullnameOverride` must be DNS labels of at most
30 characters; the release/fullname and camera ID lengths combined must not
exceed 44, so generated CronJob names fit Kubernetes limits. The chart rejects
invalid IDs instead of truncating them. A
second camera requires a Secret and another map entry:

```yaml
cameras:
  camera-01:
    sourceSecret: {name: camera-01, key: url}
  loading-bay:
    sourceSecret: {name: loading-bay-camera, key: stream-url}
    config:
      tracking:
        detector: yolo26n
        tracker: bytetrack
        device: cpu
        classes: [0]
      retention_days: 7
    resources:
      requests: {cpu: "2", memory: 2Gi}
      limits: {cpu: "4", memory: 4Gi}
```

Each entry produces its Deployment, ConfigMap, output/model claim references
and output-maintenance CronJob. Settings merge over `cameraDefaults`; set an
unwanted default camera entry to `null` when layering values. The chart owns
`camera_id`, `source_url_env`, `attributes_host`, `output_dir` and `health_port`.
Other supported stream settings belong in `config`. Detector/ReID profiles and
tracker settings use the normal engine configuration resolution; custom paths
must already exist in the selected image.

Keep each camera at one replica with `Recreate`. Stable IDs identify PostgreSQL
leases across releases and namespaces: two releases sharing a database and
camera ID compete for the same camera. Use a separate database for staging or
validation. Do not attach an HPA to camera Deployments.

The default vLLM service is internal. For a separately operated server, omit the
bundled deployment and supply its reachable endpoint:

```yaml
vllm:
  enabled: false
  externalEndpoint: https://vlm.internal.example
postgres:
  enabled: false
database:
  existingSecret: production-attributes-database
  urlKey: url
```

Selecting both bundled vLLM and an external endpoint is rejected. For external
Ollama, additionally set `cameraDefaults.config.attributes` to
`person-appearance-ollama` and prepare that profile's model on the server. The
external server operator owns its image and model revision. Camera credentials
and database URLs are injected by individual Secret keys; the chart does not
store them in Helm values or ConfigMaps.

For authenticated model serving, `cameraDefaults.secretEnv` (or a per-camera
`secretEnv`) injects individual Secret keys. The attribute configuration must
also select that environment variable through `api_key_env`; the built-in
profiles do not enable API-key authentication. This complete inline example
uses the existing Secret `model-api`, key `token`, for both client and bundled
vLLM server, and extracts one clothing field:

```yaml
cameraDefaults:
  secretEnv:
    ATTRIBUTE_API_TOKEN: {name: model-api, key: token}
  config:
    attributes:
      backend: vllm
      model: Qwen/Qwen3.5-0.8B
      api_key_env: ATTRIBUTE_API_TOKEN
      class_ids: [0]
      sample_every: 30
      batch_size: 1
      request_timeout_s: 600
      prompt: Describe visible upper-body clothing and its main color. Use Unknown if unclear.
      schema:
        type: object
        additionalProperties: false
        required: [upper_body_clothing]
        properties:
          upper_body_clothing: {type: string, minLength: 1, maxLength: 80}
vllm:
  secretEnv:
    VLLM_API_KEY: {name: model-api, key: token}
```

`VLLM_API_KEY` enables [vLLM API-key authentication](https://docs.vllm.ai/en/stable/getting_started/quickstart/).
For an external server, apply the client portion and configure authentication
on that server separately. Keep literal secrets out of `env`, values files and
command arguments. `env` and `secretEnv` cannot define the same variable;
chart-owned variables cannot be overridden. Set an inherited `secretEnv` entry
to `null` to remove it.

With bundled vLLM, the camera's attribute model must match `vllm.model`.
The default `person-appearance-vllm-cpu` profile expects `Qwen/Qwen3.5-0.8B`;
`person-appearance-vllm` expects `Qwen/Qwen3.5-9B`. Changing the served model
requires a matching profile or complete inline attribute configuration, a
real revision belonging to that model, and suitable serving resources.
The chart rejects known profile/backend mismatches and model-name overrides in
`vllm.args`. A custom attribute file must already exist in the image and contain
the matching model/backend and, when needed, `api_key_env`; Helm cannot inspect
that file's contents.

Config changes update a checksum and restart the affected camera. External
Secret content is not visible to Helm: after rotation, increment
`cameras.<id>.secretRevision` for a camera URL, or root `secretRevision` for
shared database/model credentials, then upgrade. Coordinate PostgreSQL password
rotation with the database itself; changing a Secret does not change the
password in an already initialized database.

## Storage, scheduling and cloud deployment

Every persistence block supports `storageClass`, `size`, `accessModes`,
`existingClaim` and `retain`. Empty `storageClass` uses the cluster default;
`"-"` explicitly requests no class. Default capacities are 20 GiB of output
and 10 GiB of models per camera, plus 20 GiB for vLLM. A pre-existing claim is
referenced without being created or deleted. Chart-created PVCs default to
`retain: true` and are kept on uninstall; changing that policy requires an
explicit data-lifecycle decision. Retained claims remain your responsibility
and should be deliberately reused with `existingClaim` when reinstalling.

Camera workers use UID/GID 10001. Ensure their volumes are writable and support
durable file/directory `fsync` and atomic rename. Output maintenance shares the
camera claim, so production output storage needs RWX across nodes. RWO output
claims are suitable for the single-node Minikube example; multi-node RWO use
requires an explicit placement/storage design for both workloads. PVC access
modes alone do not establish filesystem durability.

Camera, vLLM, PostgreSQL and maintenance each accept independent `resources`,
`nodeSelector`, `tolerations`, `affinity`, `runtimeClassName`, `serviceAccount`,
`podLabels` and `podAnnotations`. Service-account tokens are not automatically
mounted. Create or reference accounts and add provider annotations/labels as
needed; cloud identity associations and permissions are provisioned separately.

Start an AKS or EKS environment from `deploy/helm/examples/aks.yaml` or
`eks.yaml`, replacing the referenced storage and Secrets with provisioned
resources. Those examples use an external PostgreSQL URL and existing RWX
output claims. Set up node pools, CSI drivers, managed database backups, TLS,
private camera connectivity and cloud identity outside Helm. Check actual
storage-class names and availability in your cluster before installation. The
AKS example uses [Azure Disk model caches](https://learn.microsoft.com/en-us/azure/aks/create-volume-azure-disk)
and an operator-provisioned [Azure Files output claim](https://learn.microsoft.com/azure/aks/create-volume-azure-files).
The EKS example uses [EBS model caches](https://docs.aws.amazon.com/eks/latest/userguide/ebs-csi.html)
and an existing [EFS output claim](https://docs.aws.amazon.com/eks/latest/userguide/efs-csi.html).
EKS Auto Mode uses a different EBS provisioner from the standard CSI driver;
configure the referenced storage class for your node type.
Remote cameras need a reachable private route or an appropriately secured
publisher; a laptop's device path cannot be mounted into a cloud node.

GPU serving and GPU tracking are independent. Add
`deploy/helm/examples/gpu-vllm.yaml` after your environment values and supply a
compatible CUDA `vllm.image` digest to keep camera tracking on CPU while vLLM
requests one GPU. For GPU tracking, use a `cli-gpu` BoxMOT image, set the camera's
`config.tracking.device: "0"`, and add `nvidia.com/gpu: 1` to that camera's
resource requests and limits. One GPU camera plus one GPU vLLM server needs
two allocatable GPUs. Configure NVIDIA drivers, runtime/device plugin,
selectors and tolerations for the chosen pools; verify device assignment on
real hardware.

The chart can enable ingress policies with `networkPolicy.enabled: true` on a
cluster whose CNI enforces NetworkPolicy. They restrict bundled PostgreSQL to
this release's database clients and bundled vLLM to its camera workers.
`networkPolicy.additionalVllmIngress` admits explicitly selected extra clients,
such as a metrics scraper. These policies do not restrict egress, secure an
external server, or establish private cloud connectivity.

If Prometheus Operator and its `ServiceMonitor` CRD already exist, set
`monitoring.serviceMonitor.enabled: true` to scrape bundled vLLM's `/metrics`.
Configure `monitoring.serviceMonitor.labels` for your Prometheus selector, and
allow the scraper through `additionalVllmIngress` when ingress policies are
enabled. Prometheus itself is not installed by this chart.

Camera workers expose `/metrics` on the existing health port. Enable their
`PodMonitor` and optional alert rules separately; match the labels selected by
your Prometheus installation:

```yaml
monitoring:
  podMonitor:
    enabled: true
    labels: {release: prometheus}
  rules:
    enabled: true
    labels: {release: prometheus}
    for: 5m
    maxBacklogAgeSeconds: 300
    maxDroppedFrameRatio: 0.5
```

This requires the `PodMonitor` and `PrometheusRule` CRDs. Rules report unavailable
cameras, failed scrapes, aging attribute backlogs, high frame-drop ratios and
failed attribute outcomes. Tune thresholds against your camera FPS and latency
requirements. `boxmot_stream_frames_total` and
`boxmot_stream_dropped_frames_total` are process counters; use `rate(...[5m])`
to compare processed FPS with decoded frames discarded before tracking.
`boxmot_stream_oldest_batch_age_seconds`, `pending_batches` and
`queued_crop_bytes` describe backlog. `boxmot_stream_frame_processing_mean_seconds`
covers the last 300 processed frames. Attribute outcome counters measure
acknowledged exports, including replay from earlier sessions; use persisted
event IDs for unique counts. They do not measure model request latency. Monitor
vLLM latency and database/PVC growth separately.

## Prepare models

Camera profiles download detector/ReID checkpoints into `/opt/boxmot/models`;
vLLM uses `HF_HOME=/models/huggingface`. Persist these caches before admitting
production traffic. Startup probes allow thirty minutes for loading and first
downloads. Prepare caches using the exact images and hardware being deployed.

In restricted networks, stage the checkpoints named in
`boxmot/configs/detectors/yolo26n.yaml` and
`boxmot/configs/reid/osnet-x0-25-msmt17.yaml` using a temporary preparation pod.
Stage the complete Hugging Face snapshot selected by `vllm.model` and
`vllm.revision` into its cache with the serving image's `hf download` command
and `--revision`. Use UID 10001 for camera files, and stop preparation pods
before starting Deployments that use the same RWO claims. Private repositories
can use `vllm.hfTokenSecret: {name: model-token, key: token}`. The chart passes the
pinned revision as both model and tokenizer revisions to `vllm serve`; weights
and credentials are not in the chart.

## Upgrade the schema and install

Schema version 2 preserves version 1 runs, crops and outcomes. New databases
initialize automatically; an existing version 1 database requires an explicit
upgrade. Helm never runs migrations as an install/upgrade hook.

1. Stop every attribute client using this database, including finite `track` jobs and other releases. Suspend camera Deployments and maintenance schedules first. Suspending a CronJob does not stop a Job already running; wait for those Jobs and camera pods to finish.
2. Back up PostgreSQL using your managed database's backup procedure or `pg_dump`. Verify the backup is restorable before proceeding.
3. Enable the one-off migration with a unique operation ID and wait for completion.
4. Archive its logs, disable migration, and restore the camera/maintenance settings.

For the one-camera example, use the same environment and release-image files
for every phase. Explicitly suspend any additional camera with a per-camera
`suspend: false` override as well:

```bash
helm upgrade --install boxmot deploy/helm/boxmot --namespace boxmot-helm \
  -f environment.yaml -f release-images.yaml \
  --set cameraDefaults.suspend=true --set maintenance.suspend=true
kubectl -n boxmot-helm get pods,jobs
```

After all clients stop, back up the bundled local database; store backups
outside the cluster and protect them as application data:

```bash
umask 077
kubectl -n boxmot-helm exec deployment/boxmot-postgres -- \
  pg_dump -U boxmot -d boxmot --format=custom > boxmot-before-upgrade.dump
helm upgrade boxmot deploy/helm/boxmot --namespace boxmot-helm \
  -f environment.yaml -f release-images.yaml \
  --set cameraDefaults.suspend=true --set maintenance.suspend=true \
  --set migration.enabled=true --set-string migration.id=v2-001
kubectl -n boxmot-helm wait --for=condition=complete \
  job/boxmot-db-upgrade-v2-001 --timeout=900s
kubectl -n boxmot-helm logs job/boxmot-db-upgrade-v2-001 > schema-upgrade.log
helm upgrade boxmot deploy/helm/boxmot --namespace boxmot-helm \
  -f environment.yaml -f release-images.yaml \
  --set migration.enabled=false --wait --timeout 30m
```

The chart rejects migration mode if any configured camera or maintenance
schedule is not suspended. This validates desired configuration; it cannot
prove other clients have stopped. The Job runs `boxmot stream-db upgrade`, which
is idempotent, with a fifteen-minute deadline. SQL statements allow ten minutes
and lock acquisition thirty seconds. Plan larger histories explicitly; use a
new migration ID for each operation so immutable Job templates do not block it.

Keep the previous chart, values and image digests for rollback. `helm rollback`
restores Kubernetes configuration; it cannot reverse a database migration.
Version 1 clients reject version 2 and need their matching database backup
restored before application rollback. Releases supporting the same schema can
retain the database. Avoid automatic rollback during an explicit migration.

## Migrate an existing deployment

The repository no longer ships Compose or Kustomize manifests, launchers or CI
jobs. Dockerfiles remain for building the images Helm runs and for standalone
finite tracking jobs. Removing source files does not stop any existing
containers or delete Docker volumes, Kubernetes resources or database records.
The historical [Kubernetes validation record](https://github.com/mikel-brostrom/boxmot/blob/master/deploy/kubernetes/VALIDATION.md)
remains available as evidence for its recorded revision.

1. Record the old source revision, camera IDs, image/model identities, database URL Secret, output/model locations and retention policy. Preserve the old manifests outside the new checkout for recovery.
2. Validate the Helm release against a separate namespace and fresh database first. Use the recovery harness below; never use its disposable database for production data.
3. Stop the old camera clients and suspend their maintenance schedules. Wait for active Jobs, workers and finite attribute clients to exit. Back up PostgreSQL and output files before cutover; verify the backup is restorable.
4. For an existing Kubernetes database, reuse its URL Secret and set `postgres.enabled: false`. Reuse each output/model PVC through `existingClaim` in the **same namespace**. Helm does not adopt or move old resources. For a new namespace or cluster, restore the database and copy outputs to provisioned claims, preserving camera/session paths and UID/GID 10001 ownership.
5. For Compose, restore its PostgreSQL backup into the target database, copy persistent output files from Docker volumes/bind mounts, and reference those new claims and Secrets. Docker named volumes cannot be mounted as Kubernetes PVCs. Model caches may be copied or downloaded again using the pinned revisions.
6. Render the full Helm values with cameras and maintenance suspended. Run the explicit schema upgrade only if required, then enable the cameras. Confirm fresh sessions, completed attributes and exports before removing the old stopped workloads. Retain backups and old volumes until the cutover has been accepted.

Do not run old and new workers for the same camera/database simultaneously.
Keep database retention consistent across releases sharing that database. The
new `stream-db maintain` performs only global database cleanup;
`stream-outputs maintain --camera-id ID --output-dir /outputs --retention-days N`
handles one camera's export reconciliation and file retention. Custom cron
commands using the old combined flags must be updated.

Chart 0.2.0 requires an application image containing `stream-outputs`, live
`timestamp_source` selection, and the camera metrics endpoint. Build and test
that image from the same checkout; an older digest with the same application
version cannot supply these changes.

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
to 32 MiB by default (`max_frame_bytes`). Source indices and elapsed time survive dropped frames.
`timestamp_source: capture` is the live default: monotonic time at decoder
receipt, relative to the current connection. This avoids fictitious FPS/PTS
reported by some MJPEG backends. It is arrival timing, not sensor exposure
time. Use `timestamp_source: media` only when the source provides verified,
strictly increasing presentation timestamps; missing, reset or invalid PTS
causes a reconnect and fresh tracking session. Nominal camera FPS is never a
fallback for live timestamps. Offline video timing is unchanged. Tracker
`variable_dt` remains an explicit opt-in. OpenCV's FFmpeg backend receives ten-second
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
`captured_at_unix_s`; `timestamp_s` follows the configured capture/media
timestamp policy. No videos or image files are exported; crops are transient database
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

Size output storage as measured JSON bytes/second × 604800 seconds plus
headroom for open segments and replay. Seven-day retention is a minimum age
for deletion, not a hard disk quota: unacknowledged work is protected.
PostgreSQL also needs space for queued crops, retained events, indexes and WAL.
Monitor both database and PVC free space.

`maintenance.retentionDays` controls global database history retention.
Each camera's `config.retention_days` independently controls its files. The
per-camera job never scans or deletes global database history. Agree on one
database policy for all releases sharing the same database; use separate
databases when different history retention is required.

`<release>-db-gc` runs every five minutes. It abandons expired sessions
and removes acknowledged history older than seven days, preserving active track
state, pending jobs and unacknowledged events. `<release>-<camera>-out-gc`
runs hourly, including when that camera never restarts. For an offline camera,
it briefly acquires ownership and exports abandoned outcomes, so cleanup does
not depend on another camera connection. A worker restarting during that short
lease may need to retry acquisition. Maintenance conservatively skips file
deletion while any camera event remains unacknowledged and protects open files
while a session is active. The running camera also prunes closed historical
segments hourly. Monitor failed maintenance Jobs.

## Validate capacity and recovery

Increase serving capacity based on measured backlog age before adding cameras.
Tracking FPS and attribute requests per second are separate measurements: the
attribute worker samples tracks and completes requests asynchronously. Camera
readiness does not establish attribute generation latency or accuracy.

The health JSON, Prometheus endpoint and thirty-second status logs include processed/dropped frames,
reconnects, pending batches, queued crop bytes, oldest batch age, and mean frame
time over a fixed 300-frame window. Queue aggregates refresh at most every
five seconds after processed frames; scrapes do not query PostgreSQL. Streaming does not retain lifetime sets of
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

Chart contract tests render CPU, GPU and external-service configurations and
reject invalid values. Use server dry-run against the target cluster as well.
The real CPU integration harness installs a fixture camera, checks persisted
attributes and metadata, upgrades the release and checks recovery. Run it with
a tested image from this checkout:

```bash
uv run --no-sync python -m tests.e2e.helm_streaming --help
uv run --no-sync python -m tests.e2e.helm_capacity --help
```

The independent capacity scenario loops moving bundled video frames across
multiple camera workers. It waits for real attributes from every camera,
then measures processed FPS, dropped frames, completed attribute events and
backlog over the selected interval. Increase from two to four cameras with
identical image/resources to compare runs. See the
[E2E guide](https://github.com/mikel-brostrom/boxmot/blob/master/tests/e2e/README.md)
for commands, thresholds and recorded evidence. A short looping fixture is a
repeatable load test; it does not establish real-camera accuracy or production
capacity.

On GPU hardware verify that requested devices are available and both models
start. Cluster provisioning, managed database failover, network routes and the
monitoring stack remain operator responsibilities. Validate the chart's
optional ingress policies and ServiceMonitor against your CNI and Prometheus
configuration before production traffic.

## Command reference

::: mkdocs-click
    :module: boxmot.engine.commands.stream
    :command: stream
    :prog_name: boxmot stream
    :depth: 1
