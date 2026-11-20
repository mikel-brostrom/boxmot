# Real deployment integration tests

Shared Kubernetes lifecycle and evidence collection live in `support.py`; the
network camera lives in `fixtures/camera.py`. Recovery and capacity are separate
scenarios. Helm is the supported deployment path.

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
free disk, Docker and sudo/apt access. A manual run accepts `image`, optional `vllm_image`, and
`scenario: recovery` (default) or `scenario: capacity` with a `cameras` count. Each CI run creates and removes its own kind cluster and
retains evidence for 30 days. The promotion workflow only accepts successful
scheduled/manual runs with bundled vLLM and matching immutable identities.

This bounded fixture test does not validate GPU execution, AKS/EKS CSI storage,
managed database failover, physical camera accuracy or 24-hour resource bounds.
Test those on the target deployment before production use.

## Moving-video capacity

Run this separately from recovery, with enough resources for every camera and
the shared model server:

```bash
uv run --no-sync python -m tests.e2e.helm_capacity \
  --context YOUR_TEST_CONTEXT \
  --image 'your-registry/boxmot@sha256:YOUR_64_HEX_DIGEST' \
  --cameras 2 --duration 600 --interval 30 \
  --min-fps 1 --max-queue-age 300 \
  --output runs/helm-capacity-two-cameras
```

The default fixture loops the bundled MOT17-04 JPEG sequence at 5 FPS, resizing
frames to width 640. Each worker has a separate camera ID, session and PVCs.
The fixture uses different deterministic frame offsets for camera URLs. Change
`--cameras` to 4 in a separate run to compare load; resource requests must fit
the cluster. `--fixture-frames` selects an existing JPEG sequence. Its resized
payload must fit the bounded 650 KB fixture ConfigMap; use a short clip. No
model weights or generated frames are committed.

Warmup waits up to `--warmup-timeout` seconds (default 1800) for every camera to
complete real attributes. Measurement then records health samples and database
outcomes. It fails on worker replacement/restarts, missing frame/attribute
progress, or configured FPS/backlog thresholds. It never injects recovery
failures during the measurement interval. Completed attribute events per second
are independent of tracking FPS; they are not per-request latency.

`capacity.json` contains per-camera rates and queue peaks; `samples.jsonl`
retains observations, `fixture.json` identifies input frames by SHA-256, and
`values.json` records the exact configuration. Image/model identities and
command logs are retained beside them. `kubectl top pods` is captured when the
cluster has metrics-server. This scenario uses the same namespace ownership
and cleanup protections as recovery and never emits promotion evidence.

Use `--prepare-only` to generate moving fixtures and validate the assembled
multi-camera chart without deploying. Short looping fixtures demonstrate
repeatable load, not tracking/attribute accuracy or a representative production
workload. Follow with real-camera and 24-hour soak validation on target hardware.
