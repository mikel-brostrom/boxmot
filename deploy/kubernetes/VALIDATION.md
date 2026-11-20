# Local validation

The newer [Helm validation record](../helm/VALIDATION.md) covers chart rendering,
an isolated installation and upgrade, recovery, and persistent-volume retention.

## Built-in webcam on Minikube — 2026-11-20

The existing `boxmot-local` profile was restarted and tested with the host's
integrated `/dev/video0` camera. FFmpeg captured 640 × 480 MJPEG at 30 FPS and
supplied a temporary HTTP MJPEG relay at **5 FPS**. The relay bound only the
private Minikube Docker bridge address, `192.168.49.1:18090`; the camera pod
successfully reached it. No audio or video recording was enabled. BoxMOT wrote
its usual track/attribute metadata and temporarily stored queued person crops.

Only the `camera-01` source Secret and its rollout were changed. Images, model
revision, detector settings, attribute profile, CPU limits and PVCs were the
same as the November 18 deployment below. The original source was backed up
privately before switching. DNS and database startup failures during cluster
resume recovered before the measurement windows.

Two separate windows were measured: approximately **09:26:57–09:31:59 UTC**,
including the first attribute request, then **09:37:38–09:39:40 UTC**
after repeated attribute completions. Capture continued between the windows.

| Measurement | Initial five-minute window | Warm two-minute window |
| --- | --- | --- |
| Tracking throughput | 1,161 frames / 300.00 seconds: **3.87 FPS** | 392 frames / 119.94 seconds: **3.27 FPS** |
| Capture frames dropped | 339, **22.6%** | 208, **34.7%** |
| Ready samples | 61/61 | 25/25 |
| Reconnects / workload restarts / OOM events | 0 / 0 / 0 | 0 / 0 / 0 |
| Completed attributes during window | 1 | 2 |
| Mean vLLM request latency | **280.16 seconds**, first request | **61.64 seconds**, two requests |
| Mean capture-to-persist latency | **284.96 seconds**, first result | **63.91 seconds**, two results |
| Camera mean CPU use | 2.00 cores | 2.00 cores |
| vLLM mean CPU use | 2.11 cores | 3.85 cores |

The webcam session produced **nine successful attribute results for one tracked
identity** by the final database snapshot. All nine contained attribute values,
all nine exactly matched their exported JSONL payloads, and none was awaiting
export. The session export contained 2,727 frame records with one track per
record, without duplicate samples or conflicting payloads. Counts are scoped
to the webcam session; the database also contains earlier fixture runs. The
broader consistency check matched all 45 historical-plus-webcam snapshot events.
This verifies persistence and export, not the semantic accuracy of the model's
descriptions or general tracking accuracy.

After the first result, the next eight results arrived roughly once per minute;
their mean capture-to-persist latency was **67.68 seconds**, range
**62.01–74.97 seconds**. Mean database-to-export delay across all nine results
was 0.199 seconds. Attributes are asynchronous per-track analyses, not generated
for every processed frame. Warm vLLM requests spent 42.53 seconds in prefill and
19.10 seconds in decode on average.

CPU means use cgroup deltas. In the warm window, camera and vLLM used about
439.8 MiB and 5.97 GiB of sampled memory respectively, from three metrics-server
samples. The camera exhausted its two-core quota and was throttled in 99.7% of
active CFS periods; vLLM was throttled in 42.1%. These are scheduling-period
fractions, not wall-time fractions. The deployment therefore did not sustain
the 5 FPS input under concurrent inference. This webcam workload differs from
the repeated 200 × 360 fixture, so the FPS difference is not evidence of an
optimization.

After verification, the original fixture URL was restored, `camera-01` rolled
out successfully, and all four deployments were available. The webcam session
closed, with its completed attributes preserved. The host webcam relay,
FFmpeg process and both temporary port forwards were stopped. Minikube remains
running on the fixture source.

Ignored local artifacts are under `runs/webcam-20261120/`: `measurements/` holds
the first window; `warm-measurements/` holds the second window, the session-only
summary and output verification. The relay, reversible source switch, cleanup
wrapper and reproduction notes are retained there. The source Secret backup
and actual webcam metadata remain local; no images or attribute values are
included in this validation record. This check covers one built-in camera on
one local CPU node, not GPU execution, cloud networking or a long-running soak.

## CPU tuning experiments — 2026-11-20

These are **host-only detector measurements**, not a new Kubernetes deployment
benchmark. The `boxmot-local` API returned `no route to host`, and Docker socket
access was denied. No deployment settings were changed. The missing Python
3.11.15 interpreter was restored with uv 0.12.4 into its original `/tmp` location,
reusing the existing locked CPU environment without resynchronizing packages.

The canonical BoxMOT detector used the same cached YOLO26n model and 200 × 360
fixture as the deployment. Each configuration ran three randomized rounds of
20 measured predictions after five warm-up predictions per round. Timing
excludes capture, tracking, PostgreSQL, vLLM, output synchronization and startup.
Normal execution mode was retained. The environment was Python 3.11.15,
Torch 2.12.1+cpu, Ultralytics 8.4.58 and OpenCV 4.11.0.

With process affinity restricted to logical CPUs `0,2` (two distinct physical
cores), the focused six-versus-two-thread comparison produced:

| Detector size | Torch threads | Mean prediction time | Mean CPU time per prediction |
| --- | --- | --- | --- |
| 640 | 6 | 94.72 ms | 138.63 ms |
| 640 | 2 | 61.47 ms | 122.86 ms |
| 480 | 6 | 61.61 ms | 89.75 ms |
| 480 | 2 | 35.17 ms | 70.32 ms |

Affinity is not a Kubernetes CFS quota and does not reproduce concurrent VLM
load. In a separate unrestricted-host comparison, six threads were faster at
640 pixels (39.87 ms versus 58.08 ms for two threads), but consumed substantially
more CPU time (239.19 ms versus 116.14 ms). Thread allocation therefore needs
validation against the actual container budget; these timings must not be
reported as improved streaming FPS.

The focused comparison preserved detector outputs exactly across thread counts
at each size. At 480 pixels, all five person detections in this fixture remained;
the highest-confidence person's box had 0.975 IoU with the 640-pixel reference
and confidence 0.849 versus 0.844. These detector counts include low-confidence
background people and are different from the single confirmed track in the
deployment. An initial 320-pixel experiment reduced detections from five to
three, so it is not the preferred first size change. One repeated image cannot
establish production recall or tracking accuracy.

The source-level hypothesis that Ultralytics forces eight threads was rejected:
BoxMOT passes a `torch.device`, which takes an early return before that setting.
Fresh detector initialization retained six Torch threads on the unrestricted
host and two with two-core affinity. The actual deployed thread count remains
unmeasured. Explicit `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2` also retained two
Torch threads after detector initialization in a separate local smoke check.

Further analysis of the November 18 Prometheus snapshots separated the six
completed warm attribute requests: **32.89 seconds prefill**, **22.52 seconds
decode**, and **0.00135 seconds server queue** on average, with 625 input and 149
output tokens per completed request. These use completed-request histogram
deltas; token-counter and first-token deltas have different boundary semantics.
The measurements prioritize model compute over server-queue tuning.

The next live experiments should change one factor at a time: camera thread
allocation, then detector size 480; separately test a shorter attribute prompt
while retaining structured-output validation, a lower vision minimum pixel
budget, or vLLM CPU quota 4 → 5 with existing binding unchanged. These VLM changes
are hypotheses, not measured improvements. Check attribute correctness as well
as latency, and keep database/export consistency checks in each live run.

The ignored `runs/deployment_tuning_20261120/` directory contains the benchmark,
logs, raw timings, input hashes and candidate camera patches. All **1,020 timed
predictions** across the three experiments completed. Reproduction commands:

```bash
UV_CACHE_DIR=/tmp/boxmot-review-uv uv run --no-sync python \
  -m runs.deployment_tuning_20261120.detector_benchmark \
  --cpus 0,2 --samples 20 --output /tmp/boxmot-detector-grid.json
UV_CACHE_DIR=/tmp/boxmot-review-uv uv run --no-sync python \
  -m runs.deployment_tuning_20261120.detector_benchmark \
  --sizes 640 480 --threads 6 2 --samples 20 \
  --output /tmp/boxmot-detector-host.json
UV_CACHE_DIR=/tmp/boxmot-review-uv uv run --no-sync python \
  -m runs.deployment_tuning_20261120.detector_benchmark \
  --cpus 0,2 --sizes 640 480 --threads 6 2 --samples 20 \
  --output /tmp/boxmot-detector-two-cores.json
```

## Minikube CPU deployment — 2026-11-18

The solution was deployed to the isolated `boxmot-local` Minikube profile,
namespace `boxmot-streams`, with real detector, attribute-model and PostgreSQL
execution. This supersedes the earlier limitation that no Kubernetes cluster
had been exercised. Existing application databases were not used.

The host is an AMD Ryzen 5 4600H with 6 cores/12 threads and approximately
15 GiB RAM. Minikube 1.39.0 runs Kubernetes/kubectl 1.37.2 using Docker and
containerd, with local container limits of 8 CPU and 10000 MB memory. The local
deployment uses a dedicated PostgreSQL 17 database and four `standard`
ReadWriteOnce hostPath PVCs. These are single-node storage, and their requested
capacities are not enforced filesystem quotas.

The BoxMOT image is the previously validated Compose build; its application
code has not changed. Images were imported locally and selected by digest:

- BoxMOT: `boxmot/boxmot@sha256:66df3468dfa1b707f442491de7250893b2eca929889bcfb5d5764ebacd82e268`.
- vLLM: `vllm/vllm-openai-cpu@sha256:4154467a886f8b7ffa9dbab32f873f5ca4528409d042a6ec866e8888223263d3`.
- PostgreSQL: `postgres@sha256:0d938e975177bef9f1f61f3f9dbcf2e0aa0a139568f66d3415ed8892eb97f6ac`.

Tracking uses YOLO26n and ByteTrack on CPU. vLLM serves
`Qwen/Qwen3.5-0.8B` in BF16 at model revision
`2fc06364715b967f1860aea9cf38778875588b17`. The camera fixture repeats one
200 × 360 JPEG over HTTP MJPEG at a nominal 5 FPS. This is a functional workload,
not an accuracy evaluation or a representative camera-capacity benchmark.
The camera requests 1 CPU/1 GiB and is limited to 2 CPU/2 GiB; vLLM requests
2 CPU/4 GiB and is limited to 4 CPU/8 GiB.

The first vLLM startup exposed a Kubernetes-specific defect: the Service named
`vllm` injected `VLLM_PORT=tcp://...`, while vLLM expects a numeric port. The
worker exited with a configuration error, not an OOM kill. The shared pod
template now sets `enableServiceLinks: false`; both CPU and GPU rendered
contracts verify it. The focused Kubernetes tests passed **3 tests**. Separately,
the local vLLM memory limit was raised from 6 to 8 GiB after startup diagnostics
reported no headroom at the lower limit.

With images and models cached, the corrected vLLM pod reached Ready **81 seconds**
after creation; the camera pod reached Ready after **16 seconds**. These are pod
creation-to-readiness intervals, not first attribute-result latency.

The five-minute measurement ran from approximately **19:30:07 to 19:35:09 UTC**
while tracking and attribute inference were both active:

| Measurement | Observed result |
| --- | --- |
| Tracking throughput | 905 frames over 299.97 seconds: **3.02 FPS** |
| Capture frames dropped | 593, or **39.6%** of processed plus dropped frames |
| Availability | 61/61 ready samples; zero reconnects or workload restarts |
| Successful attributes | Six during the window; seven total at the database snapshot |
| vLLM request latency | **55.43 seconds mean**, six completed requests |
| Capture-to-persisted attribute latency | **62.21 seconds mean**, range 58.75–67.34 seconds, six window results |
| First attribute result | **263.51 seconds** after its source capture, before the measurement window |
| Camera CPU and memory | 1.99 CPU cores mean; 422.9 MiB sampled mean memory |
| vLLM CPU and memory | 3.96 CPU cores mean; 5.98 GiB sampled mean memory |
| Container lifetime memory peaks | Camera 704.9 MiB; vLLM 6.19 GiB, including startup |
| Database/export consistency | All seven snapshot events matched exported JSONL payloads; none awaiting export |
| Backlog at final database snapshot | One running batch, age 53.11 seconds; 6,048 bytes of crops; no pending batches |

CPU means come from cgroup counter deltas; memory means use six distinct
metrics-server samples. Neither worker had an OOM event during the window.
The camera and vLLM were throttled in 97.4% and 58.1% of active CFS scheduling
periods respectively. Those percentages describe scheduling periods. Both
workers used nearly their full CPU quotas, and this configuration did not
sustain the fixture's nominal 5 FPS. Capture-to-result latency includes
application scheduling and inference; the database-to-export delay averaged
0.184 seconds across all seven completed events.

The standalone local measurement and verification scripts were run as follows,
with loopback port forwards already active:

```bash
.venv/bin/python runs/minikube-20261118T191126Z/collect_metrics.py \
  --duration 300 --interval 5 \
  --camera-url http://127.0.0.1:18080 --vlm-url http://127.0.0.1:18000 \
  > runs/minikube-20261118T191126Z/measurement.log 2>&1
.venv/bin/python runs/minikube-20261118T191126Z/verify_outputs.py
kubectl --context boxmot-local -n boxmot-streams get pods -o json
```

Local manifests, startup logs, model references and measurement tools are
retained under `runs/minikube-20261118T191126Z/`. These ignored artifacts are not
committed. In particular, `vllm-startup-failure.log` preserves the original error,
and `deployment-summary.json` records the local configuration. Measurements are
in `measurements/report.json`; raw health/resource samples, PostgreSQL snapshots
and vLLM metrics are retained alongside it. The export check in
`measurements/output-snapshot/verification.json` has `verified: true`, with seven
matching events and no missing acknowledged events or payload mismatches.

This run does not cover GPU inference, a physical camera, multiple Kubernetes
nodes, or a 24-hour soak. The earlier remaining checks still apply to those
areas.

## Compose reliability follow-up — 2026-11-18

The Compose defaults now pin the previously validated CPU vLLM image and
Hugging Face revision `2fc06364715b967f1860aea9cf38778875588b17`. The model cache
uses a Docker volume, prefilled for this check from a read-only host mount.
The GPU vLLM 0.33.0 and Ollama image digests were checked against their registry
manifests; their inference remains untested on this host. The Ollama overlay
checks the installed model's full manifest digest before tracking starts.

The updated checkout was rebuilt and passed **all nine real CPU integration
stages** with the pinned model and named cache:

```bash
UV_CACHE_DIR=/tmp/boxmot-review-uv uv run --no-sync python -m tests.e2e.compose_attributes --profile cpu
```

Image: `boxmot/boxmot:boxmot-e2e-c13c965779`. Evidence:
`runs/boxmot-e2e-c13c965779/result.json` and its command logs and exports.
The report has `error: null` and no cleanup errors. The finite video produced
real tracks and completed attributes in PostgreSQL and JSONL. Streaming
inference, ownership exclusion, camera reconnection, cancellation during
inference, database outage/recovery and acknowledged export consistency all
passed. Service startup took 131 seconds and the finite pipeline 236 seconds.
All test containers were removed; the test database and model cache volumes
remain for inspection and reuse.

The commands below used Python 3.11 and `UV_CACHE_DIR=/tmp/boxmot-review-uv`:

```bash
uv run --no-sync python -m tests.ci.compose_contract
uv run --no-sync python -m pytest tests/unit/engine/test_attributes.py \
  tests/unit/engine/test_prompted_tracking.py tests/unit/engine/test_runner_sinks_v24.py -q
uv run --no-sync python -m pytest tests/unit/engine/test_cli_v24.py \
  tests/unit/test_compose_contract.py tests/unit/test_docker_webcam_launcher.py \
  tests/unit/engine/streaming/test_compose_cleanup.py -q
uv run --no-sync python -m pytest tests/unit/test_docker_cache_layers.py \
  tests/unit/test_docker_dependency_manifest.py tests/unit/test_docker_native_runtime.py \
  tests/unit/test_docker_workflow_v24.py -q
uv run --no-sync python -m pytest tests/unit/engine/test_prompted_tracking.py -k prompted_attributes -q
uv run --no-sync python -m boxmot.engine.cli track --help
uv run --no-sync python -m tests.e2e.compose_attributes --profile gpu --prepare-only
uv run --no-sync mkdocs build --strict
bash -n docker/webcam.sh
git diff --check
```

All six actual Compose merges passed (CPU/GPU/Ollama, with and without webcam).
The first pytest selection passed 74 tests with two PostgreSQL-fixture skips
because `BOXMOT_TEST_POSTGRES_URL` was unset. The other selections passed 182,
76 and 2 tests respectively. The final two tests additionally exercise prompted
tracking with its default JSON sink and shared attribute directory; one replaces
a case in the first selection. CLI help, GPU preparation, shell syntax, Ruff on
all changed Python files, whitespace and the strict documentation build passed.
GPU preparation evidence is in `runs/boxmot-e2e-d39d16548c/`.

The documented backup and upgrade commands were also run against an isolated
PostgreSQL 17 database containing version-1 tracks and completed attributes.
The custom-format archive was readable, the one-off `stream-db upgrade` command
succeeded twice and preserved both latest attributes and events, and restoring
the backup into a separate database preserved the original version-1 schema
and values. Commands and results are in
`runs/boxmot-e2e-upgrade-70e4763b/result.json`; its containers and database volume
were removed. No existing application database was upgraded.

## Docker and PostgreSQL 17 checks — 2026-11-18

Docker access was enabled for this session. The CPU CLI image was built from
the checkout, and real YOLO detection, vLLM inference and PostgreSQL were
exercised in isolated Compose projects. Existing application databases and
unrelated containers were left unchanged.

### Environment and reproducibility

- Python 3.11.15, uv 0.12.4, Docker 29.1.3, Compose 2.40.3.
- AMD Ryzen 5 4600H, 6 cores/12 threads, approximately 15 GiB RAM. This CPU
  has AVX2 but no AVX512-BF16 acceleration.
- Built BoxMOT image: `boxmot/boxmot:boxmot-e2e-08f69c407b` (`cli-cpu`).
- vLLM 0.33.0, `vllm/vllm-openai-cpu:latest-x86_64`, image digest
  `sha256:4154467a886f8b7ffa9dbab32f873f5ca4528409d042a6ec866e8888223263d3`.
- Qwen/Qwen3.5-0.8B, BF16; real model weights were already cached.
- PostgreSQL 17.12 in a dedicated test volume.

The Python environment was restored with the locked dependency set:

```bash
UV_CACHE_DIR=/tmp/boxmot-uv-cache UV_PYTHON_INSTALL_DIR=/tmp/boxmot-python \
  uv sync --frozen --python 3.11 --extra cpu --extra yolo --extra evolve \
  --extra service --extra attributes --group dev --group test --group docs
```

The reusable [Compose test](../../tests/e2e/README.md) builds the current image
by default:

```bash
uv run --no-sync python -m tests.e2e.compose_attributes --profile cpu
```

Subsequent test iterations reused the successfully built image with
`--image boxmot/boxmot:boxmot-e2e-08f69c407b`; no runtime Python source changed
after that build. The test uses a bundled pedestrian fixture as both a short
video and a repeating HTTP MJPEG camera. It does not measure model accuracy
or physical-camera compatibility.

### Completed real-model run

The fresh run **passed all nine stages**, exited with status 0 and required no
manual intervention:

```bash
UV_CACHE_DIR=/tmp/boxmot-uv-cache uv run --no-sync python -m tests.e2e.compose_attributes \
  --profile cpu --image boxmot/boxmot:boxmot-e2e-08f69c407b
```

Evidence is retained locally in `runs/boxmot-e2e-73f64e8988/result.json`, numbered
command logs, and the `finite/` and `streams/` output directories. The record has
`error: null`. All of this project's containers were removed successfully;
its database volume, image and output artifacts remain available for inspection.

| Check | Observed result |
| --- | --- |
| Fixture/configuration and service startup | Compose configuration valid; PostgreSQL and real vLLM healthy |
| Finite video pipeline | 24 frame records with real person tracks; one completed attribute result persisted and exported |
| Network camera pipeline | Real OpenCV HTTP capture, YOLO, ByteTrack, vLLM, database persistence and acknowledged export |
| Competing owner | Rejected with `OwnershipLost`; lease identity and session records unchanged |
| Camera interruption | Readiness cleared while liveness stayed healthy; reconnect created a fresh session |
| SIGTERM during inference | Observed running batch explicitly cancelled, no outstanding camera jobs, exit 0 without OOM; within the 22-second test bound around the 20-second application deadline |
| PostgreSQL outage/recovery | Worker exited nonzero; restart after lease expiry created a fresh session, retained completed event IDs/payloads and exported a new completed result |
| Final exports | 492 streaming frame records across five sessions; two completed and six cancelled attribute events; every acknowledged streaming event matched its JSONL payload |

The finite pipeline took 233 seconds including cold inference; model service
startup took 67 seconds with cached weights. These are observations from this
fixture and host, not throughput or capacity guarantees. The later cleanup-only
hardening was tested separately with failure injection; it does not change any
pipeline or recovery assertion from this successful run.

### Findings and corrections

The original CPU settings caused lengthy Triton compilation for Mamba prefix
cache copies on the first inference request. The CPU Compose service and CPU
Kubernetes base now use `--disable-prefix-caching`. The validated profile retains
BF16; an experimental FP32 run failed because this CPU attention backend rejects
FP32, and that configuration was discarded. Cold inference still takes several
minutes on this older CPU.

Early test attempts also exposed harness problems: profiled services were
omitted from log collection/teardown, a cross-file service extension lost its
input mounts, and Compose attempted to start an intentionally paused model
dependency during a worker restart. These were corrected in the harness.
Interrupted or failed attempts are retained as failures in their own
`runs/boxmot-e2e-*/result.json` files.

The installed-image release contract now includes `stream` and `stream-db`.
It passed against the built image from `/tmp`, outside the checkout:

```bash
docker run --rm -i --workdir /tmp --entrypoint python \
  boxmot/boxmot:boxmot-e2e-08f69c407b -I - --check-cli-help \
  < tests/ci/release_contract.py
```

### Regression and manifest checks

The following selection passed **196 tests** against PostgreSQL 17.12, using
a separate test database selected through `BOXMOT_TEST_POSTGRES_URL`:

```bash
uv run --no-sync python -m pytest \
  tests/unit/attributes tests/unit/engine/streaming \
  --ignore=tests/unit/engine/streaming/test_kubernetes.py \
  --ignore=tests/unit/engine/streaming/test_soak.py -q
```

The Docker build/configuration selection passed **76 tests**:

```bash
uv run --no-sync python -m pytest \
  tests/unit/test_docker_workflow_v24.py tests/unit/test_docker_cache_layers.py \
  tests/unit/test_docker_dependency_manifest.py tests/unit/test_docker_native_runtime.py -q
```

After the CPU argument change, the affected `test_docker_workflow_v24.py`
selection passed again (**20 tests**, included in the preceding 76).
The Kubernetes contract selection passed **3 tests**. Both overlays and the
upgrade Job rendered successfully with kubectl 1.35.0; kubeconform 0.8.0 reported
**21 valid resources, zero invalid, zero errors and zero skipped** against
Kubernetes 1.35.0 schemas. Placeholder digest-shaped image references were used
for structural checks; no images were published and no cluster was modified.
Exact rendering and validation commands are recorded in
`runs/boxmot-e2e-9ef41a9d6e/kubernetes-commands.json`.

The final GPU Compose configuration also passed:

```bash
uv run --no-sync python -m tests.e2e.compose_attributes --profile gpu --prepare-only
```

Its merged stream service preserves the NVIDIA device request, all three
mounts, the base environment and the nonroot user. Evidence is under
`runs/boxmot-e2e-dfcfc2abed/`. This command starts no containers.

Harness cleanup failure injection passed **5 tests**: a log collection timeout
must not prevent other cleanup steps, and cleanup/report failures must not mask
the original test failure or produce a false success. Final Ruff, byte
compilation, whitespace checks and the strict documentation build also passed:

```bash
uv run --no-sync python -m pytest tests/unit/engine/streaming/test_compose_cleanup.py -q
uv run --no-sync ruff check tests/e2e tests/ci/release_contract.py \
  tests/unit/engine/streaming/test_compose_cleanup.py
uv run --no-sync python -m compileall -q tests/e2e
uv run --no-sync mkdocs build --strict
git diff --check
```

### Remaining infrastructure checks

- An actual `docker run --gpus device=0 ... nvidia-smi -L` allocation attempt
  failed with `could not select device driver "" with capabilities: [[gpu]]`.
  The host exposes one GTX 1650 with 4 GiB VRAM and driver 580.178.04, but Docker
  has no NVIDIA runtime configured. GPU image/model startup and separate GPU
  allocation remain untested; the latter also requires a second GPU.
- No Kubernetes cluster was available. Actual scheduling, PVC behavior, probes
  in a cluster, deployment/rollback and a CPU pod soak remain untested.
- A physical camera, external Ollama inference, a 24-hour memory/storage soak,
  and seven days of wall-clock retention were not exercised. Dedicated tests
  cover simulated retention, queue limits, rotation, export failures/replay,
  blocked capture, stale results and bounded collections; these are distinct
  from the real-model Compose test.
- The Python 3.10–3.13 execution matrix remains a CI check.

## Earlier implementation checks — 2026-11-17

The record below describes the earlier implementation pass. Its Docker-access
and host-driver limitations were superseded by the checks above; its missing
cluster and GPU execution coverage still applies.

Validation used Python 3.11.15, uv 0.12.4, CPU PyTorch 2.12.1,
OpenCV 4.11.0, and an isolated PostgreSQL 16.15 instance under `/tmp`.
No existing application database or live cluster was modified.

### Environment

uv 0.12.4 was already installed. The initial `pip install uv==0.12.4` attempt
was rejected by the system interpreter's externally managed environment policy.
The repository environment was then prepared successfully with:

```bash
UV_CACHE_DIR=/tmp/boxmot-uv-cache UV_PYTHON_INSTALL_DIR=/tmp/boxmot-python \
  uv sync --python 3.11 --extra cpu --extra yolo --extra evolve \
  --extra service --extra attributes --group dev --group test --group docs
```

Tests used `BOXMOT_TEST_POSTGRES_URL` pointing to that isolated database. The
sandbox initially blocked local sockets; database and process/HTTP integration
tests were rerun with local connectivity enabled. These are resolved setup
failures, not skipped database coverage.

### Checks and results

The following focused regression selection passed **382 tests**:

```bash
uv run --no-sync python -m pytest \
  tests/unit/attributes tests/unit/engine/streaming \
  tests/unit/engine/test_workflow_pipeline.py \
  tests/unit/engine/test_tracking_profiling_v24.py \
  tests/unit/engine/test_sources_v24.py \
  tests/unit/engine/test_latest_camera_source.py \
  tests/unit/engine/test_cli_startup.py \
  tests/unit/engine/test_runner_sinks_v24.py \
  tests/unit/engine/test_variable_dt_cli.py -q
```

After additional recovery tests and final fixes, this selection passed **140
tests** (it overlaps the preceding selection):

```bash
uv run --no-sync python -m pytest \
  tests/unit/attributes/test_streams.py tests/unit/attributes/test_session.py \
  tests/unit/engine/streaming \
  --ignore=tests/unit/engine/streaming/test_soak.py \
  tests/unit/engine/test_attributes.py \
  tests/unit/engine/test_track_timing_report.py \
  tests/unit/engine/test_batch_profiling.py -q
```

Targeted final checks also passed:

| Command suffix after `uv run --no-sync python -m pytest` | Result |
| --- | --- |
| `tests/unit/attributes/test_streams.py -q` | 13 passed; includes lock-wait expiry, renewal loss, takeover, additive migration and offline-camera export/retention |
| `tests/unit/engine/streaming/test_soak.py tests/unit/trackers/box/test_bytetrack_memory.py -q` | 5 passed; 10,000 real CPU tracker updates each for ByteTrack and BoT-SORT, bounded archives and collectible retired tracks |
| `tests/unit/engine/streaming/test_outputs.py -q` | 5 passed; sync failure, partial-write poisoning, duplicate replay, rotation and cleanup |
| `tests/unit/engine/streaming/test_kubernetes.py -q` | 3 passed with kubectl on PATH; immutable images, both profiles, GPU allocation and upgrade Job |
| `tests/unit/engine/streaming/test_runtime.py tests/unit/engine/test_attributes.py tests/unit/engine/test_runner_sinks_v24.py -q` | 58 passed; reconnects/SIGTERM and separate streaming output preserve finite-track behavior |

The final retention review added ownership locking around bounded cleanup pages
and scoped longer timeouts to explicit schema upgrades. The following affected
selection produced 19 passes and one test assertion error (the new test used
dictionary access on an `AttributeResult`). Correcting that assertion and
rerunning the affected test passed:

```bash
uv run --no-sync python -m pytest \
  tests/unit/attributes/test_streams.py \
  tests/unit/engine/streaming/test_runtime.py \
  tests/unit/engine/streaming/test_kubernetes.py -q
uv run --no-sync python -m pytest \
  tests/unit/attributes/test_streams.py::test_retention_skips_camera_with_inflight_admission -q
```

This covers concurrent admission during retention, cleanup beyond a 1,000-row
page, upgrade-only timeout settings, and the final 900-second upgrade Job.

The standalone profiler test also processes 30,000 synthetic samples while
retaining only aggregate counters and a 300-frame window. The runtime integration
test uses real PostgreSQL and sends SIGTERM during blocked asynchronous inference.
These tests inject perception/model backends; they do not load production models.

Both final overlays and the schema-upgrade Job were rendered with kubectl 1.35.0
and placeholder *digest-shaped* image references for structural validation.
kubeconform 0.8.0 reported **21 valid resources, zero invalid, zero errors, zero
skipped**:

```bash
kubeconform -strict -summary -kubernetes-version 1.35.0 \
  /tmp/boxmot-cpu.yaml /tmp/boxmot-gpu.yaml /tmp/boxmot-upgrade.yaml
```

The deployment renderer itself was exercised by the Kubernetes contract tests.
These checks do not establish that any placeholder image exists in a registry.

Also passed:

```bash
uv run --no-sync python -m boxmot.engine.cli stream --help
uv run --no-sync python -m boxmot.engine.cli stream-db --help
uv run --no-sync python -m boxmot.engine.cli track --help
uv run --no-sync mkdocs build --strict
git diff --check
```

Ruff checks passed on all changed runtime modules and new tests. New streaming
modules passed Python 3.10 syntax parsing and Python 3.11 byte compilation.

### Checks requiring deployment infrastructure

- No kubeconfig was available: cluster server dry-run, real CPU pod soak,
  scheduling, PVC behavior, failure injection and live camera connectivity were
  not exercised.
- Docker daemon access was denied to the current user, including an elevated
  read-only check; passwordless sudo was unavailable. Affected images were not
  built or started locally.
- `nvidia-smi` could not communicate with the NVIDIA driver. GPU model startup,
  CUDA compatibility and separate physical GPU allocation were not exercised.
- The full Python 3.10–3.13 execution matrix remains a CI check; local execution
  used Python 3.11.

Follow the [deployment guide](../../docs/integrations/kubernetes.md) to prepare
real image digests, Secrets, storage and models, then run the documented cluster
and hardware acceptance checks before production.
