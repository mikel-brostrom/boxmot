# Historical Kubernetes validation

Helm is the supported deployment path. The former Kustomize manifests and
renderer have been removed; no existing cluster resources are changed by that
source cleanup.

See the [Helm chart](../helm/README.md) and the
[migration procedure](../../docs/integrations/kubernetes.md#migrate-an-existing-deployment).
Stop old workers and maintenance before cutover, back up PostgreSQL and outputs,
and explicitly reuse or restore claims and Secrets. Do not delete data volumes
as part of changing deployment tooling.

[VALIDATION.md](VALIDATION.md) records historical test results for their stated
revision. It does not validate the current chart or application image.
