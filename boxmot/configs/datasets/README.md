# Dataset Configs

This directory contains dataset definitions for BoxMOT's config-driven
`generate`, `eval`, and `tune` commands.

Each dataset config is a self-contained benchmark definition:

- `id`: dataset/benchmark identifier
- `root`: local directory where the dataset is stored (download destination)
- `layout`: dataset storage layout, for example `mot` or `visdrone`
- `box_type`: `aabb` or `obb`
- `splits`: dict of available splits (e.g. `train`, `val`, `test`, `ablation`)
- `default_split`: which split to use when `--split` is not specified
- `classes`: benchmark ground-truth class labels (id → name)
- `distractors`: classes to ignore during evaluation (optional)
- `class_map`: GT class name to detector class name mapping (optional)
- `defaults.detector`: default detector profile (references `configs/detectors/`)
- `defaults.reid`: default ReID profile (references `configs/reid/`)
- `download.dataset`: optional dataset download URL
- `download.runs`: optional detections/embeddings cache download URL

TrackEval selection is derived from `box_type`:

- `aabb` -> MOTChallenge runner
- `obb` -> OBB runner for RGB image sequences

OBB ground-truth data should be stored in 13-column corner format on disk.
