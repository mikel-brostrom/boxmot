# Datasets

Dataset configs live under `boxmot/configs/datasets`.

## Role

Dataset configs describe dataset facts only:

- `id`
- dataset `path`
- active `split`
- layout such as `mot` or `visdrone`
- `box_type` as `aabb` or `obb`
- class names, distractors, and class mappings
- optional dataset and cache download URLs

Detector and ReID defaults do not belong here. They are selected by benchmark bundles.

## Geometry and evaluation

TrackEval selection is derived from `box_type`:

- `aabb` uses the MOTChallenge-style runner
- `obb` uses the OBB runner for RGB image sequences

OBB ground truth is expected in 13-column corner format on disk.
