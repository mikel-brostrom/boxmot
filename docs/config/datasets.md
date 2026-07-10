# Datasets

Dataset settings are defined inline in each benchmark YAML under `boxmot/configs/benchmarks`.

## Role

Dataset blocks describe dataset facts only:

- `id`
- dataset `path`
- active `split`
- layout such as `mot` or `visdrone`
- `box_type` as `aabb` or `obb`
- class names, distractors, and class mappings
- optional dataset and split-aware cache download URLs

Detector and ReID defaults are defined alongside the dataset in the same benchmark file.

## Geometry and evaluation

Metric geometry is derived from `box_type`:

- `aabb` uses MOTChallenge-style axis-aligned IoU
- `obb` uses rotated IoU for RGB image sequences

OBB ground truth is expected in 13-column corner format on disk.
