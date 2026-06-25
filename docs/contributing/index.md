# Contributing Guide

The repo has four main extension surfaces:

- trackers under `boxmot/trackers`
- native C++ sources under `boxmot/native/cpp/trackers` and Python bindings under `boxmot/native/trackers`
- configs under `boxmot/configs`
- generic workflow internals under `boxmot/engine`, public Python API entrypoints under `boxmot/api`, and ReID lifecycle workflows under `boxmot/reid/workflows`

## Where to start

- [Add a Tracker](add-tracker.md)
- [Add OBB Support](obb-support.md)
- [Add Configs and Benchmarks](configs.md)
- [Testing](testing.md)
- [CI and Benchmarks](ci.md)
