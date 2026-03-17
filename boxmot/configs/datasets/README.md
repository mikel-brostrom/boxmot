# Dataset Configs

This directory contains dataset definitions for BoxMOT's config-driven
`generate`, `eval`, and `tune` commands.

A dataset config should describe dataset facts only:

- `id`: dataset config identifier
- `path`: dataset root
- `split`: active split name
- `train`, `val`, `test`: split-relative paths when needed
- `layout`: dataset storage layout, for example `mot` or `visdrone`
- `box_type`: `aabb` or `obb`
- `names`: benchmark ground-truth classes
- `distractors`: classes to ignore during evaluation
- `class_map`: benchmark class name to detector class name mapping
- `download.dataset`: optional dataset download URL

TrackEval selection is derived from `box_type`:

- `aabb` -> MOTChallenge runner
- `obb` -> OBB runner for RGB image sequences

OBB ground-truth data should be stored in 13-column corner format on disk.

Detector and ReID defaults belong in benchmark bundles under
`boxmot/configs/benchmarks/`.
