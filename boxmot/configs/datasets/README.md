# Dataset Configs

This directory contains dataset definitions for BoxMOT's config-driven
`generate`, `eval`, and `tune` commands.

A dataset config should describe benchmark facts only:

- `id`: dataset config identifier
- `path`: dataset root
- `split`: active split name
- `train`, `val`, `test`: split-relative paths when needed
- `layout`: dataset layout, for example `mot`
- `box_type`: `aabb` or `obb`
- `trackeval`: TrackEval adapter name
- `names`: benchmark ground-truth classes
- `distractors`: classes to ignore during evaluation
- `class_map`: benchmark class name to detector class name mapping
- `models`: default detector+ReID config to use
- `download`: optional dataset or cached-runs URLs

Example:

```yaml
id: mot17-ablation

path: boxmot/engine/trackeval/data/MOT17-ablation
split: train
train: train
val:
test:

layout: mot
box_type: aabb
trackeval: mot_challenge

names:
  1: pedestrian

distractors:
  2: person_on_vehicle
  7: static_person
  8: distractor
  12: reflection

class_map: {}
models: mot17-ablation-models
```

Use a dataset config by name:

```bash
boxmot eval --data mot17-ablation --tracker boosttrack
```

Or by path:

```bash
boxmot eval --data boxmot/configs/datasets/mot17-ablation.yaml --tracker boosttrack
```

Detector and ReID defaults live separately under `boxmot/configs/models/`.
