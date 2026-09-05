# Config System Overview

BoxMOT keeps packaged tracking configuration in the central `boxmot/configs`
catalog. Experiments describe reproducible dataset/model compositions and are
the only materialization inputs. Tracker selection remains a replay-time
choice.

## Config families

- `experiments/` selects a dataset split, detection source, optional ReID profile, and class map.
- `datasets/` describes dataset facts and download locations.
- `detectors/` describes detector models and named checkpoints.
- `reid/` describes reusable runtime ReID models.
- `trackers/<tracker>.yaml` contains tracker runtime defaults and tuning search spaces.
- `trackers/presets/` contains reusable tracker overrides.
- `runtime.yaml` contains shared tracking-workflow defaults.

Use an experiment for every materialization run:

```bash
boxmot materialize --experiment mot17-ablation-yolox-lmbn
boxmot eval --experiment mot17-ablation-yolox-lmbn --build BUILD_ID --tracker boosttrack
```

The `--experiment` option accepts a unique experiment ID,
a filename, or an explicit YAML path. Built-in IDs use kebab-case and catalog
assets use repository-relative paths rather than workstation-specific absolute
paths.

When perception choices should change, add a new experiment YAML that
references the desired dataset, split, detector checkpoint, and optional ReID
or segmentor config. Materialization has no dataset-only, direct-source, or
component-override mode.

Before setup begins, BoxMOT expands experiment references, resolves numeric
class IDs, and validates the combination. Materialized manifests record the
resolved semantic component fingerprints; eval, tune, and research validate
them without performing perception inference.

## Related pages

- [Mode Defaults](modes.md)
- [Experiments](experiments.md)
- [Datasets](datasets.md)
- [Detectors](detectors.md)
- [ReID Profiles](reid.md)
- [Tracker YAMLs](trackers.md)
