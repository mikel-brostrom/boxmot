# Contributing Guide

The repo has six main extension surfaces:

- representation-first tracker packages under `boxmot/trackers/box`,
  `boxmot/trackers/mask`, and `boxmot/trackers/multimodal`
- native C++ sources and low-level ABI bindings under `boxmot/native`, with
  Python-facing adapters beside their tracker or ReID domain implementation
- canonical packaged configuration under `boxmot/configs`
- application orchestration under the owning `boxmot/engine` subsystem
  (`tracking`, `materialization`, `eval`, `tuning`, `service`, or
  `commands`), with command presentation and Rich progress isolated in
  `boxmot/engine/ui`; there is no generic workflow or `boxmot.api` layer
- reusable ReID backbones, backends, datasets, training, and exporters under
  `boxmot/reid`, with CLI command orchestration under `boxmot/engine/commands/reid`
- dependency-light downloads and local model-path resolution under
  `boxmot/resources`; renderers integrate through callbacks, so resources never
  import the engine UI

`boxmot/utils` is deliberately limited to genuinely cross-domain checks,
configuration primitives, logging/constants, and Torch device helpers. Put new
helpers beside the subsystem that owns their behavior instead of rebuilding a
generic `misc` module.

## Where to start

- [Add a Tracker](add-tracker.md)
- [Add OBB Support](obb-support.md)
- [Add Catalog Entries and Experiments](configs.md)
- [Build the Documentation](documentation.md)
- [Testing](testing.md)
- [CI and Benchmarks](ci.md)
