# Docker images

Use the [Docker integration guide](../docs/integrations/docker.md) for image
builds, CPU/GPU video jobs, persistent caches, and batch workflows. Continuous
camera deployments use the [Helm chart](../deploy/helm/README.md) on Minikube,
AKS, or EKS. Compose and the former Kustomize deployment are retired.

## Targets

| Target | Contents | Published tags |
| --- | --- | --- |
| `cli-cpu` | Detection, tracking, attributes and evaluation with CPU PyTorch | `boxmot/boxmot:<version>-cpu`, `sha-<commit>-cpu`, `latest-cpu` |
| `cli-gpu` | Full CLI with CUDA 13.0 PyTorch | Build locally |
| `service-cpu` | Non-root, headless detection-to-track HTTP service | `boxmot/boxmot-service:<version>`, `sha-<commit>`, `latest` |
| `service-gpu` | Non-root HTTP service with CUDA ReID | Build locally |

`<version>` is the root `pyproject.toml` release version. Use immutable image
digests for deployments. A build without `--target` selects `cli-gpu`.

```bash
docker build --target cli-cpu -f docker/Dockerfile -t boxmot/boxmot:local-cpu .
docker build --target cli-gpu -f docker/Dockerfile -t boxmot/boxmot:local-gpu .
```

The Dockerfile uses the root manifest, lockfile and pinned uv version. Separate
dependency, native-build and application layers preserve build cache reuse.
CLI wheels include native tracker libraries; compilers stay in build stages.
`.dockerignore` excludes host native binaries, model weights and local data.
CPU release images are smoke-tested and retained as candidate artifacts. Release
publication promotes those exact images without rebuilding; see the
[release workflow](../docs/contributing/ci.md#release-gates) for artifact retention
and recovery. GPU inference requires acceptance testing on a compatible NVIDIA
host.

- [HTTP tracking service deployment](../docs/guides/deployment.md)
- [Helm end-to-end verification](../tests/e2e/README.md)
- [Database upgrades and migration](../docs/integrations/kubernetes.md)

Removing deployment files does not delete existing Docker containers or volumes.
Back up and migrate existing databases before stopping an old installation.
