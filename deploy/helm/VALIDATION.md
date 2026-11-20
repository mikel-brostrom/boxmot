# Helm validation — 2026-11-20

Chart `0.1.0` (`appVersion: 26.0.0`) passed local rendering and isolated
Minikube installation, configuration upgrade, and recovery tests with both
shared and chart-managed CPU vLLM. No chart or image was published, and no cloud
deployment was performed.

## Bundled CPU test

A second test used the chart's own CPU vLLM and PostgreSQL Deployments in
namespace `boxmot-helm-e2e-5bd729e30c` on the existing `boxmot-local` cluster.
It used the bundled MOT17 pedestrian fixture, a separate database and persistent
claims, and the same locally imported BoxMOT image as the earlier test below.
The existing `boxmot-streams` camera and vLLM Deployments were temporarily scaled
to zero to keep both inference stacks within the host's memory capacity. Their
original replica counts and restoration results are recorded separately in
`runs/helm-bundled-20261120-orchestration.json`. Both were restored to one
replica and became ready after the test exited zero.

The harness was run from the repository root without an external vLLM endpoint:

```bash
PATH=/tmp/boxmot-helm-3.19.2/linux-amd64:$PATH \
/home/mikel/boxmot/.venv/bin/python3 -u -m tests.e2e.helm_streaming \
  --context boxmot-local \
  --image boxmot/boxmot@sha256:66df3468dfa1b707f442491de7250893b2eca929889bcfb5d5764ebacd82e268 \
  --output runs/helm-bundled-20261120 \
  --detector-weights models/yolo26n.pt \
  --storage-class standard
```

The chart used CPU vLLM image digest
`4154467a886f8b7ffa9dbab32f873f5ca4528409d042a6ec866e8888223263d3` and
`Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`.
The initial Hugging Face Xet transfer returned successful partial responses but
was slow, approximately 0.48 MB/s in the observed interval. To bound the local
test duration, the already-cached, pinned model weights were copied into the
test's vLLM claim and only the test vLLM pod was replaced. The 1,746,942,600-byte
file was SHA256-verified inside the container as
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
The container ran as UID/GID 10001 and its Hugging Face and vLLM cache
subdirectories were owned by that user. This checks serving from a seeded
persistent cache; a full cold network model download did not complete.
The intervention is recorded in `cache-seeding.json` and
`xet-download-summary.json` under `runs/helm-bundled-20261120/`.

| Check | Result |
| --- | --- |
| Bundled vLLM startup and real detection/tracking/attributes | Passed with the seeded model cache |
| Installed BoxMOT version | `26.0.0`, matching chart `appVersion` |
| Config upgrade and retained PVC UIDs/event history | Passed |
| Source disconnect/reconnect and camera pod replacement | Passed, with fresh sessions |
| PostgreSQL restart and resumed attribute generation | Passed |
| Database and output maintenance Jobs | Passed |
| Export/database correspondence | 10/10 events matched exactly |
| Successful attribute results | 3, spanning install, upgrade and database recovery |
| Track exports | 1,737 frame records across 5 sessions |
| Uninstall retention | All 4 chart-created PVCs remained |
| Isolated namespace cleanup | Completed; no test namespace remains |

The other seven events were cancellations during deliberately interrupted
sessions. Every session with a completed attribute result exported nonempty
tracks. All eight harness phases passed, with no test, cleanup or diagnostic
errors in `result.json`. Counts and image/version evidence are in
`verification.json` and `deployment-images.json` in the same evidence directory.
Deleting the isolated namespace after the retention check also removed its test
data and retained claims.

Thirty read-only runtime samples were taken approximately every 35 seconds.
The camera had limits of 2 CPUs/2 GiB and vLLM 4 CPUs/8 GiB. The fixture supplied
5 FPS.

| Observed metric | Result |
| --- | --- |
| Seeded vLLM pod creation to first Ready condition | 81 seconds |
| Initial camera pod creation to first Ready condition | 11 seconds |
| First vLLM request latency, including first-use compilation | 250.31 seconds |
| Two subsequent vLLM request latencies | 59.05 and 44.91 seconds; mean 51.98 seconds |
| Tracking over an initial 210-second ready interval | 4.87 FPS while vLLM was largely compiling |
| Tracking over a later 35-second ready interval | 2.71 FPS during warm inference |
| vLLM cgroup memory peak | 6.82 GiB, including cache |

These short windows describe a functional disruption test, not sustained
throughput or attribute accuracy. The vLLM health endpoint became ready before
first-use Triton CPU kernel compilation completed, so a healthy server did not
yet imply warm inference latency. NUMA memory-binding warnings under the
container's `RuntimeDefault` seccomp profile did not prevent inference.
No vLLM container restarts or OOM kills appeared in the runtime samples. The
camera restarted three times during the induced database outage, then recovered
and completed another attribute result. `runtime-samples.jsonl` and
`runtime-summary.json` separate pod identities and container restart counters
when calculating rates.

## Earlier CPU test with shared vLLM

The test used the existing `boxmot-local` Kubernetes 1.37.2 cluster and a new
namespace/database, `boxmot-helm-e2e-27312475f8`. It served the bundled MOT17
pedestrian fixture over HTTP, seeded YOLO26n into its own model claim, and
connected to the already-running vLLM server in `boxmot-streams`. The original
camera, database and serving Deployments were not replaced or reconfigured.

```bash
PATH=/tmp/boxmot-helm-3.19.2/linux-amd64:$PATH \
UV_CACHE_DIR=/tmp/boxmot-review-uv uv run --no-sync python -m tests.e2e.helm_streaming \
  --context boxmot-local \
  --image boxmot/boxmot@sha256:66df3468dfa1b707f442491de7250893b2eca929889bcfb5d5764ebacd82e268 \
  --output runs/helm-live-20261120 \
  --detector-weights models/yolo26n.pt \
  --external-vllm-url http://vllm.boxmot-streams.svc.cluster.local:8000 \
  --storage-class standard
```

The camera image is a locally imported test build, not a published registry
artifact. Its installed BoxMOT version was checked separately in the running
pod and returned `26.0.0`. The existing server used CPU vLLM image digest
`4154467a886f8b7ffa9dbab32f873f5ca4528409d042a6ec866e8888223263d3` and
`Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`.

| Check | Result |
| --- | --- |
| Install and real detection/tracking/attributes | Passed |
| Config upgrade restarts camera, preserving PVC UIDs and event history | Passed |
| Source disconnect/reconnect and fresh session | Passed |
| Camera pod replacement and ownership recovery | Passed |
| PostgreSQL restart and resumed attribute generation | Passed |
| Database and output maintenance Jobs | Passed |
| Export/database correspondence | 9/9 events matched exactly |
| Successful attribute results | 3, spanning install, upgrade and database recovery |
| Track exports | 656 frame records across 5 sessions |
| Uninstall retention | All 3 chart-created PVCs remained |
| Isolated namespace cleanup | Completed; no test namespace remains |

The other six events record cancellations during the deliberately interrupted
sessions. Every session with a completed attribute result also exported nonempty
tracks. Counts describe functional validation with a shared CPU model server,
not a throughput benchmark or attribute-accuracy evaluation. Ignored evidence
is in `runs/helm-live-20261120/result.json`, `verification.json`, and numbered
command logs. The process exited zero with no test or cleanup error. The
original four Deployments were confirmed available after cleanup.

## Configuration and release checks

The following checks passed (Helm 3.19.2 was checksum-verified):

```bash
PATH=/tmp/boxmot-helm-3.19.2/linux-amd64:$PATH \
UV_CACHE_DIR=/tmp/boxmot-review-uv uv run --no-sync python -m tests.ci.helm_contract

PATH=/tmp/boxmot-helm-3.19.2/linux-amd64:$PATH \
UV_CACHE_DIR=/tmp/boxmot-review-uv uv run --no-sync python -m pytest -q \
  tests/unit/test_deployment_release.py \
  tests/unit/test_helm_streaming_harness.py \
  tests/unit/test_docker_workflow_v24.py \
  tests/unit/engine/streaming/test_kubernetes.py

UV_CACHE_DIR=/tmp/boxmot-review-uv uv run --no-sync python -m ruff check \
  tests/ci/helm_contract.py tests/ci/deployment_release.py \
  tests/e2e/helm_streaming.py tests/unit/test_deployment_release.py \
  tests/unit/test_helm_streaming_harness.py

/tmp/boxmot-actionlint/actionlint .github/workflows/docker.yml \
  .github/workflows/helm.yml .github/workflows/helm-release.yml \
  .github/workflows/promote-images.yml

UV_CACHE_DIR=/tmp/boxmot-review-uv uv run --no-sync mkdocs build --strict \
  --site-dir /tmp/boxmot-helm-docs-site-final
git diff --check
```

The focused pytest selection passed **58 tests**. The chart contract runs strict
lint plus real Helm rendering for multiple cameras, CPU/GPU combinations,
external dependencies, all four environment examples, authentication, ingress
policies, monitoring, upgrades, and invalid configuration. It checks naming
boundaries, model/profile matching, immutable references and Secret isolation.
Synthetic image digests are used only for these structural checks.

CPU, GPU and migration manifests also passed Kubernetes server dry-run,
including the optional NetworkPolicies:

```bash
kubectl --context boxmot-local -n boxmot-streams apply --dry-run=server \
  -f /tmp/boxmot-helm-cpu.yaml -f /tmp/boxmot-helm-gpu.yaml \
  -f /tmp/boxmot-helm-migration.yaml
```

These temporary manifests were produced by `tests.ci.helm_contract.render`.
The final documentation authentication example was separately resolved with
the actual attribute configuration loader and rendered through Helm.

## Remaining runtime validation

- Bundled CPU serving under UID 10001 passed with a seeded, verified model cache.
  Full cold network download and representative cold/warm performance remain
  unverified. The scheduled/manual CI job still requires a published
  `BOXMOT_HELM_IMAGE` and a suitable runner.
- GPU hardware, AKS/EKS storage and workload identity, NetworkPolicy enforcement
  and actual Prometheus scraping were not exercised. ServiceMonitor rendering
  was checked; this cluster does not have the required Operator CRD.
- Migration Job rendering and API validation passed; the live test used a new
  schema version 2 database, not a version 1 database upgrade.
- The bundled run exercised the harness's separate dependency waits, runtime
  version check, per-session track assertions and cleanup checks. Minikube's
  `standard` StorageClass uses `Immediate` binding; actual
  `WaitForFirstConsumer` provisioning remains a kind/cloud CI check.
- OCI publication and promotion were validated through helper tests and workflow
  checks, without registry writes. External-server evidence is explicitly
  ineligible for image promotion. A 24-hour representative-camera soak remains
  a production prerequisite.
