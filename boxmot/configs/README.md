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
  split, detection source, optional ReID profile, and evaluation class map.

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
detections:
  source: model
  model:
    ref: yolox-x-mot17
    checkpoint: ablation
reid:
  ref: lmbn-n-duke
  crop_strategy: aabb
```

The crop strategy belongs to the experiment rather than the reusable ReID
profile because it describes how that encoder consumes the selected detection
geometry. OBB experiments can select `perspective` or `rotated`;
mask-producing experiments can select `mask_aware`.

Configuration loading and validation live with the owning Python domain; this
directory contains declarative assets only. ReID training recipes and export
defaults intentionally remain under `boxmot/reid/` because they are not
tracking runtime profiles.

## References

Catalog references resolve by unique ID, filename, or explicit YAML path.
Built-in IDs use kebab-case, and built-in asset paths must be portable
repository-relative paths rather than workstation-specific absolute paths.

Materialization accepts only `--experiment`, which selects a complete catalog
composition by ID or YAML. To change the dataset split or perception
components, create a distinct experiment that references the corresponding
reusable assets:

```bash
boxmot materialize --experiment mot17-ablation-yolox-lmbn
boxmot eval --experiment mot17-ablation-yolox-lmbn --build BUILD_ID --tracker boosttrack
```
