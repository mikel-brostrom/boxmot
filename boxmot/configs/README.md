# BoxMOT configuration assets

This directory is the single source of truth for the version-controlled YAML
assets used by BoxMOT's tracking-by-detection workflows.

## Layout

- `runtime.yaml` contains shared CLI defaults plus mode-specific defaults for
  `track`, `materialize`, `eval`, `tune`, and `research`.
- `datasets/` describes dataset format, storage, splits, ground-truth
  availability, classes, and dataset download resources.
- `detectors/` describes detector classes, box type, inference defaults, and
  checkpoints.
- `reid/` describes ReID weights, runtime defaults, and preprocessing.
- `trackers/<tracker>.yaml` contains each tracker's runtime defaults and tuning
  search metadata.
- `trackers/presets/` contains named runtime parameter profiles produced for a
  particular dataset or split.
- `experiments/` contains the user-facing compositions that select a dataset
  split, detector profile, optional ReID profile, and evaluation class map.

## Ownership rules

Each fact belongs to exactly one asset. Experiments reference reusable assets
by identifier; they do not copy dataset, detector, ReID, or tracker
definitions. Tracker selection remains an independent runtime choice rather
than being embedded in a dataset or experiment.

For example, an experiment may compose:

```yaml
dataset:
  ref: mot17
  split: ablation
detector:
  ref: yolox-x-mot17
  checkpoint: ablation
reid:
  ref: lmbn-n-duke
```

ReID crop extraction follows the detection geometry automatically. AABB
detections use clipped axis-aligned crops, while OBB detections use the
canonical rectified OBB transform. Built-in ReID profiles do not consume
detection masks; a custom mask-dependent encoder declares that requirement
through its encoder contract.

Configuration loading and validation live with the owning Python domain; this
directory contains declarative assets only. ReID training recipes and export
defaults intentionally remain under `boxmot/reid/` because they are not
tracking runtime profiles.

## References

Dataset, detector, and ReID references resolve by unique ID, filename, or
explicit YAML path. Experiment selectors resolve by YAML filename or path and
never by a declared `id`. Built-in asset paths must be portable
repository-relative paths rather than workstation-specific absolute paths.

Materialization accepts only `--experiment`, which selects a complete catalog
composition by its catalog-relative YAML filename or an explicit YAML path. To
change the dataset split or perception components, create a distinct experiment
that references the corresponding reusable assets:

```bash
boxmot materialize --experiment mot17/ablation-yolox-lmbn.yaml
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --tracker boosttrack
```
