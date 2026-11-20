# Real deployment integration tests

## Compose smoke and recovery

Run from the repository root with Docker daemon access and host `ffmpeg`:

```bash
uv run --no-sync python -m tests.e2e.compose_attributes --profile cpu
```

The test builds the current BoxMOT CPU image and starts PostgreSQL and vLLM.
It uses actual YOLO detection and attribute inference on a small crop from the
bundled MOT17 fixture. A successful tracking exit alone is insufficient: the
test requires completed attributes in both exported JSONL and PostgreSQL.

It then serves a repeating JPEG as a real HTTP camera stream and exercises:

- continuous capture and attribute export;
- competing camera ownership, requiring an ownership rejection and unchanged lease/session identities;
- camera disconnect/reconnect with fresh session IDs;
- SIGTERM during an outstanding inference request, requiring an explicit cancellation for the observed batch and a clean exit;
- database loss, worker termination, preserved event IDs/payloads, and a newly completed result after recovery;
- correspondence between synchronized JSONL payloads and database acknowledgments, including consistency of any duplicate event IDs.

The shutdown check briefly pauses this test's vLLM container and starts a fresh
worker session, so inference cannot finish before SIGTERM reaches the worker.
The server is unpaused in cleanup. No model responses are fabricated.

Each invocation uses a unique Compose project and database volume. Existing
application databases and running Compose projects are not selected. vLLM uses
the dedicated external volume `boxmot-e2e-vllm-models`; it never writes to the
host Hugging Face cache. Override its name with `BOXMOT_TEST_VLLM_CACHE_VOLUME`.
The volume is created when needed and retained across tests. To prefill it, use
the Docker guide's read-only host cache copy with this test volume as the target.
Exact commands, logs, JSONL, and `result.json` are retained
under the printed `runs/boxmot-e2e-.../` directory. The test stops its containers
on completion or failure and preserves its database volume for inspection.
`BOXMOT_TEST_PROJECT` may select a unique `boxmot-e2e-...` project name for CI
cleanup; an existing artifact directory is rejected to prevent accidental reuse.

For the GPU Compose variant, use `--profile gpu`. Docker must support NVIDIA
GPU allocation, and the GPU must support the selected model/image. The existing
Compose defaults share GPU 0; set `BOXMOT_GPU_ID=0 VLLM_GPU_ID=1` to test distinct
devices on a two-GPU host. This does not test Kubernetes GPU scheduling.

To reuse an already verified BoxMOT image, pass `--image IMAGE_REFERENCE`. This
skips the build; the image must contain the code you intend to test. The default
always builds the current checkout.

`--prepare-only` generates the fixture and checks Compose configuration without
starting containers. It does **not** establish that model inference works.

This bounded test does not replace a 24-hour deployment soak, physical-camera
testing, Kubernetes/PVC validation, or the dedicated retention and replay tests.
It does not inject output failures, a crash between synchronization and acknowledgment,
or a blocked capture backend. Queue capacity, rotation, retention, stale inference
responses after takeover, and bounded memory need their dedicated tests.
Cold image builds, downloads, and CPU inference may take several minutes. Each
stage has a timeout and retains its log on failure.

### Compose CI

The `Compose integration` workflow checks all six assembled CPU, GPU and Ollama
configurations, with and without the webcam overlay, on relevant PRs. Run this
check locally without a Docker daemon:

```bash
uv run --no-sync python -m tests.ci.compose_contract
```

A weekly or manually dispatched job builds the checkout and runs the real CPU
test. It requires at least 12 GiB RAM and 25 GiB free disk, and retains logs,
result JSON and attribute exports for 14 days. The default runner is
`ubuntu-24.04`; set the repository variable `BOXMOT_COMPOSE_RUNNER` to a suitable
larger runner label if needed. Each CI invocation uses its own model cache and
removes its test volumes after collecting the evidence. Persistent runners
should also reclaim resources left by forced job termination.
GPU execution remains a manual check on a host with the required hardware.

## Helm install, upgrade and recovery

Use an existing single-node Kubernetes test cluster with Helm, `kubectl`,
`ffmpeg`, a default dynamic storage class and immutable images accessible to
its nodes. The BoxMOT CPU image must contain the code you intend to test; this
harness does not build or publish it. Run from the repository root:

```bash
uv run --no-sync python -m tests.ci.helm_contract
uv run --no-sync python -m tests.e2e.helm_streaming \
  --context YOUR_TEST_CONTEXT \
  --image 'your-registry/boxmot@sha256:YOUR_64_HEX_DIGEST' \
  --output runs/helm-validation-001
```

The contract checks lint/render the chart, validate generated stream settings
and exercise invalid configurations without connecting to Kubernetes. The real
test creates an isolated `boxmot-helm-e2e-*` namespace, fresh PostgreSQL and a
bundled fixture camera. It requires actual detector/tracker and VLM results,
then checks:

- tracking and completed attributes in PostgreSQL and synchronized JSONL;
- configuration upgrade restarting the camera while retaining PVCs and history;
- camera disconnect/reconnect, camera pod replacement and database restart;
- fresh sessions and completed attributes after recovery;
- maintenance Job completion and database/export correspondence;
- PVC retention after Helm uninstall.

Existing namespaces are refused. Commands always use the supplied Kubernetes
context. Cleanup deletes only the namespace created and owned by that run,
including its test data, after collecting evidence. PVC retention is checked
before namespace deletion. Use a new output directory for each invocation.
Logs, `result.json` and `deployment-images.json` identify completed checks and
the tested artifacts. Test-created secrets are sent through stdin and excluded
from command logs; treat collected application data as private.

`--storage-class` selects the cluster's storage class. `--vllm-image` overrides
the chart's pinned CPU image. `--detector-weights /path/to/yolo26n.pt` seeds the
isolated model claim to avoid downloading an already available checkpoint.
For a local chart check, `--external-vllm-url` can reuse a reachable serving
endpoint; this does not validate bundled vLLM installation and cannot produce
image-promotion evidence. `--prepare-only` generates the fixture and renders
values without contacting Kubernetes; it does not establish inference works.

The `Helm integration` workflow runs render checks on relevant PRs. Weekly CPU
inference requires repository variable `BOXMOT_HELM_IMAGE`; use
`BOXMOT_HELM_RUNNER` for a runner with 8 CPUs, at least 16 GiB RAM and 25 GiB
free disk, Docker and sudo/apt access. A manual run accepts `image` and optional
`vllm_image` inputs. Each CI run creates and removes its own kind cluster and
retains evidence for 30 days. The promotion workflow only accepts successful
scheduled/manual runs with bundled vLLM and matching immutable identities.

This bounded fixture test does not validate GPU execution, AKS/EKS CSI storage,
managed database failover, physical camera accuracy or 24-hour resource bounds.
Test those on the target deployment before production use.
