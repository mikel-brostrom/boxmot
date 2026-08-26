# Config System Overview

BoxMOT keeps packaged tracking configuration in the central `boxmot/configs`
catalog. Experiments describe reproducible dataset/model compositions; model-free
dataset configs leave detector, ReID, tracker, and runtime choices to the
caller.

## Config families

- `experiments/` selects a dataset split, detection source, optional ReID profile, and class map.
- `datasets/` describes dataset facts and download locations.
- `artifacts/` describes public detections and precomputed detection/embedding artifacts.
- `detectors/` describes detector models and named checkpoints.
- `reid/` describes reusable runtime ReID models.
- `trackers/<tracker>.yaml` contains tracker runtime defaults and tuning search spaces.
- `trackers/presets/` contains reusable tracker overrides.
- `runtime.yaml` contains shared tracking-workflow defaults.

Use an experiment for a composed catalog run:

```bash
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack
```

The `--experiment` option accepts a unique experiment ID,
a filename, or an explicit YAML path. Built-in IDs use kebab-case and catalog
assets use repository-relative paths rather than workstation-specific absolute
paths.

Use `--dataset` when model choices should remain selected by the CLI or API:

```bash
boxmot eval --dataset mot17 --split ablation --tracker boosttrack
```

Before setup begins, BoxMOT expands experiment references, resolves numeric
class IDs, and validates the combination. Experiment-driven cached tracking
result directories later record both `config.source.yaml` and
`config.resolved.yaml`, including the effective tracker and runtime overrides.

## Related pages

- [Mode Defaults](modes.md)
- [Experiments](experiments.md)
- [Datasets](datasets.md)
- [Detectors](detectors.md)
- [ReID Profiles](reid.md)
- [Tracker YAMLs](trackers.md)
