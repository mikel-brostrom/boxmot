# BoxMOT configuration assets

This directory is the single source of truth for the version-controlled YAML
assets used by BoxMOT's tracking-by-detection workflows.

## Layout

- `runtime.yaml` contains shared CLI/API defaults plus mode-specific defaults
  for `track`, `generate`, `eval`, `tune`, and `research`.
- `datasets/` describes dataset format, storage, splits, ground-truth
  availability, classes, and dataset download resources.
- `artifacts/` describes public detections and precomputed
  detection/embedding runs, including their producer lineage.
- `detectors/` describes detector classes, box type, inference defaults, and
  checkpoints.
- `reid/` describes ReID weights, runtime defaults, and preprocessing.
- `trackers/<tracker>.yaml` contains each tracker's runtime defaults and tuning
  search metadata.
- `trackers/presets/` contains named runtime parameter profiles produced for a
  particular dataset or split.
- `experiments/` contains the user-facing compositions that select a dataset
  split, detection source, optional ReID profile, and evaluation class map.

## Ownership rules

Each fact belongs to exactly one asset. Experiments reference reusable assets
by identifier; they do not copy dataset, detector, ReID, artifact, or tracker
definitions. Tracker selection remains an independent runtime choice rather
than being embedded in a dataset or experiment.

For example, an experiment may compose:

```yaml
dataset:
  ref: mot17
  split: ablation
detections:
  source: model
  model:
    ref: yolox-x-mot17
    checkpoint: ablation
reid:
  ref: lmbn-n-duke
```

Configuration loading and validation live with the owning Python domain; this
directory contains declarative assets only. ReID training recipes and export
defaults intentionally remain under `boxmot/reid/` because they are not
tracking runtime profiles.

## References

Catalog references resolve by unique ID, filename, or explicit YAML path.
Built-in IDs use kebab-case, and built-in asset paths must be portable
repository-relative paths rather than workstation-specific absolute paths.

Use `--dataset` when the detector, ReID model, and other runtime choices stay
caller-controlled:

```bash
boxmot eval --dataset mot17 --split ablation --tracker boosttrack
```

Use `--experiment` to select a complete catalog composition by ID or YAML:

```bash
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack
```
