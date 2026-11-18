# Local validation

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
