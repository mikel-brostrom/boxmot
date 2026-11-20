# Kustomize camera deployment

For new deployments, use the [Helm chart](../helm/README.md) and
[Kubernetes guide](../../docs/integrations/kubernetes.md). These existing CPU/GPU
overlays remain available during the transition. They do not provision
PostgreSQL, and raw bases contain image placeholders.

The runtime, camera ownership, output and retention behavior is shared with
Helm. This directory's resource names and installation commands differ.

## Configure and render

Edit `base/stream.yaml` for CPU or `overlays/gpu/stream.yaml` for GPU. Configure
storage classes, capacity, node placement and resources for your cluster.
Outputs require RWX across camera and maintenance nodes; model caches use RWO.
The GPU overlay requests one device for the camera and one for vLLM. Pin the
model revision in the serving arguments when preparing a release.

Create the pre-existing database and camera Secrets from protected files:

```bash
kubectl apply -f deploy/kubernetes/base/namespace.yaml
kubectl -n boxmot-streams create secret generic attributes-database \
  --from-file=url=/secure/postgres-url
kubectl -n boxmot-streams create secret generic camera-01 \
  --from-file=url=/secure/camera-url
export CAMERA_IMAGE='your-registry/boxmot@sha256:YOUR_64_HEX_DIGEST'
export VLM_IMAGE='your-registry/vllm@sha256:YOUR_64_HEX_DIGEST'
uv run --no-sync python -m deploy.kubernetes.render --profile cpu \
  --camera-image "$CAMERA_IMAGE" --vlm-image "$VLM_IMAGE" > /tmp/boxmot.yaml
kubectl apply --dry-run=server -f /tmp/boxmot.yaml
```

Use `--profile gpu` and compatible CUDA image digests for the GPU overlay.
Rendering rejects mutable image tags and applies no live resources. A new
database initializes automatically when clients start; existing schema version
1 databases require the explicit upgrade below.

## Upgrade and install

Stop every attribute client sharing the database, including finite `track`
jobs, and suspend existing maintenance CronJobs. Wait for running Jobs and
camera pods to finish, then make and verify a restorable PostgreSQL backup.
Render the separate migration Job:

```bash
uv run --no-sync python -m deploy.kubernetes.render --profile cpu \
  --camera-image "$CAMERA_IMAGE" --vlm-image "$VLM_IMAGE" --upgrade \
  > /tmp/boxmot-upgrade.yaml
kubectl apply -f /tmp/boxmot-upgrade.yaml
kubectl -n boxmot-streams wait --for=condition=complete \
  job/attribute-schema-upgrade --timeout=900s
kubectl -n boxmot-streams logs job/attribute-schema-upgrade > schema-upgrade.log
kubectl apply -f /tmp/boxmot.yaml
kubectl -n boxmot-streams rollout status deployment/vllm --timeout=1800s
kubectl -n boxmot-streams rollout status deployment/camera-01 --timeout=1800s
```

Before a subsequent upgrade, archive logs and delete the previous completed
`attribute-schema-upgrade` Job. Resume any CronJobs you explicitly suspended.
The upgrade is idempotent; rolling back to a version 1 client requires restoring
its matching database backup. See the main guide for shutdown, schema and
rollback constraints.

Each extra camera requires its own Deployment, ConfigMap, Secret, output/model
claims and output-maintenance CronJob, with consistent names, selectors and
camera ID. Keep one replica with `Recreate`. Helm generates these resources
from a camera map, but it cannot automatically adopt this installation. Stop
old owners before a cutover, retain the database/output data, and deliberately
configure the existing claim and Secret references.
