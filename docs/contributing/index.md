# Contributing Guide

The repo has five main extension surfaces:

- trackers under `boxmot/trackers`
- native C++ sources under `boxmot/native/cpp/trackers` and Python bindings under `boxmot/native/trackers`
- configs under `boxmot/configs`
- generic workflow internals under `boxmot/engine` and public Python API entrypoints under `boxmot/api`
- reusable ReID backbones, backends, datasets, training, and exporters under
  `boxmot/reid`, with CLI/workflow orchestration under `boxmot/engine/reid`

## Where to start

- [Add a Tracker](add-tracker.md)
- [Add OBB Support](obb-support.md)
- [Add Catalog Entries and Experiments](configs.md)
- [Testing](testing.md)
- [CI and Benchmarks](ci.md)
