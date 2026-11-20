# BoxMOT Helm deployment

The chart in `boxmot/` deploys continuous camera workers, their maintenance
Jobs and storage, with optional vLLM and PostgreSQL. Follow the
[Kubernetes guide](../../docs/integrations/kubernetes.md) for installation,
schema upgrades, camera ownership, outputs, and production preparation.

The chart requires an explicit BoxMOT image digest. It does not contain camera
URLs or database passwords; create the referenced Secrets separately.
See the [validation record](VALIDATION.md) for the local install/upgrade/recovery
results and remaining cloud/GPU checks.
Model API keys can use `cameraDefaults.secretEnv` and `vllm.secretEnv`; configure
the matching `api_key_env` in the client's full attribute configuration. See the
guide's complete example. Changing `vllm.model` also requires a matching client
profile and model revision.

```bash
export CAMERA_IMAGE='your-registry/boxmot@sha256:YOUR_64_HEX_DIGEST'
helm lint deploy/helm/boxmot -f deploy/helm/examples/minikube.yaml \
  --set-string image="$CAMERA_IMAGE"
helm template boxmot deploy/helm/boxmot --namespace boxmot-helm \
  -f deploy/helm/examples/minikube.yaml --set-string image="$CAMERA_IMAGE"
```

Examples are overlays on the chart defaults:

| File | Purpose |
| --- | --- |
| `examples/minikube.yaml` | One CPU camera, local PostgreSQL, CPU vLLM and single-node storage. |
| `examples/aks.yaml` | Existing PostgreSQL and RWX output claim, with AKS model-cache storage. |
| `examples/eks.yaml` | Existing PostgreSQL and RWX output claim, with a provisioned EBS storage class. |
| `examples/gpu-vllm.yaml` | GPU model serving while camera tracking remains on CPU. |

Cloud examples require operator-provisioned Secrets, storage, node pools,
networking and images. They do not create an AKS/EKS cluster or managed database.
Review the comments before using them. Keep the selected chart version, image
digests, model revision and environment values together in version control.

`values.schema.json` validates the values structure; template checks reject
incompatible dependency, migration and camera settings. Tests additionally
render multiple cameras, CPU/GPU combinations, external services and upgrades.

Optional ingress NetworkPolicies restrict the bundled database and model server.
The optional vLLM ServiceMonitor requires an existing Prometheus Operator CRD
and appropriate scraper access. Neither feature provisions the cluster's
networking or monitoring stack.
