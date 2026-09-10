# Contributing Guide

The repo has six main extension surfaces:

- algorithm packages under `boxmot/trackers/<name>`, with shared support code
  under `boxmot/trackers/common` and representation families declared in
  capability metadata
- native C++ sources and low-level ABI bindings under `boxmot/native`, with
  Python-facing adapters beside their tracker or ReID domain implementation
- canonical packaged configuration under `boxmot/configs`
- application orchestration under the owning `boxmot/engine` subsystem
  (`tracking`, `materialization`, `eval`, `calibration`, `tuning`, `service`, or
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

The engine root contains `cli.py` and its package initializer. Shared workflow
configuration lives in `boxmot/engine/config/`: `runtime.py` owns mode defaults
and argument normalization, `experiments.py` resolves authored experiments,
and `trackers.py` applies workflow-specific tracker overrides. Import directly
from these modules; the configuration namespace stays lightweight.

Dataset acquisition belongs in `engine/materialization/resources.py` beside
catalog and build orchestration. Capture timestamps belong in
`engine/tracking/timestamps.py`; `tracking/timing.py` measures execution time.
Console logging belongs in `engine/ui/logging.py`.

Shared tracker motion code lives under `boxmot/trackers/common/motion/`:
`models.py` provides motion adapters, `tracker.py` integrates them with
trackers, and `kalman_filters/` contains filters, noise configuration, and
numerical fitting. Camera-motion estimators live in `cmc/`, with construction
in `cmc/registry.py` and track updates in `cmc/integration.py`. Keep these
package initializers lightweight and import implementations from their owning
modules. Dataset loading and Kalman calibration workflows belong in
`boxmot/engine/calibration/`: `kalman.py` fits covariance scales from cached
detections and ground truth, while `ground_truth_noise.py` owns
annotation-based noise estimation.

Hyperparameter search belongs in `boxmot/engine/tuning/`, including search
backends and `eagermot_kitti.py`'s Optuna trials. Tuning may run calibration
before a search and freeze its results through `calibration_profile.py`.
Calibration must not import tuning or require search backends.

Dependency validation lives in `boxmot/utils/dependencies.py` and reads package
metadata. Keep package installation in `boxmot/engine/commands/install.py`,
which owns the explicit `boxmot install` command. Reusable components and
normal workflows should validate their requirements and report missing
dependencies.

## Where to start

- [Add a Tracker](add-tracker.md)
- [Add OBB Support](obb-support.md)
- [Add Catalog Entries and Experiments](configs.md)
- [Build the Documentation](documentation.md)
- [Testing](testing.md)
- [CI and Benchmarks](ci.md)
