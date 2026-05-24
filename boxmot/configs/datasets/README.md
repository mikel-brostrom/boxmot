# Dataset Configs

This directory contains dataset definitions for BoxMOT's config-driven
`generate`, `eval`, and `tune` commands.

Each dataset config is a self-contained benchmark definition:

- `id`: dataset/benchmark identifier
- `path`: dataset root
- `split`: default split name (overridable via `--split`)
- `train`, `val`, `test`: split-relative paths when needed
- `layout`: dataset storage layout, for example `mot` or `visdrone`
- `box_type`: `aabb` or `obb`
- `detector`: default detector profile (references `configs/detectors/`)
- `reid`: default ReID profile (references `configs/reid/`)
- `names`: benchmark ground-truth classes
- `distractors`: classes to ignore during evaluation
- `class_map`: benchmark class name to detector class name mapping
- `download.dataset`: optional dataset download URL
- `download.runs`: optional detections/embeddings cache download URL

TrackEval selection is derived from `box_type`:

- `aabb` -> MOTChallenge runner
- `obb` -> OBB runner for RGB image sequences

OBB ground-truth data should be stored in 13-column corner format on disk.
