# Continuous camera deployment

See the [Kubernetes guide](../../docs/integrations/kubernetes.md) for image
publication, Secrets, model caches, storage, schema upgrades and installation.

Render `overlays/cpu` or `overlays/gpu` through `python -m deploy.kubernetes.render`
with explicit image digests. The raw bases contain image placeholders and do
not provision PostgreSQL. No live cluster resources are applied by the renderer.
