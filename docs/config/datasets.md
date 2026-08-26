# Datasets

Dataset configs live under `boxmot/configs/datasets`. They contain
dataset facts and their download locations, with no detector, ReID, or
experiment selection.

```yaml
id: mot17

format:
  layout: mot
  box_type: aabb

storage:
  root: boxmot/datasets/mot/MOT17

default_split: ablation

splits:
  train:
    path: train
    has_ground_truth: true
  ablation:
    path: ablation
    has_ground_truth: true
  test:
    path: test
    has_ground_truth: false

classes:
  target:
    pedestrian: 1
  ignore:
    distractor: 8

resources:
  dataset:
    type: per_split
    uris:
      train: hf://example/mot17/train
      test: hf://example/mot17/test
      ablation: hf://example/mot17/train
```

The class groups make evaluation roles explicit without repeating an
`evaluation` field for every entry. Split properties stay together, so
evaluation can reject a split without ground truth before looking for annotation
files. A dataset's `resources` mapping may contain only its own `dataset`
download.

## Artifact resources

Dataset download URIs stay in the dataset config. Detector checkpoint URIs live
in detector configs, and ReID weight URIs live in ReID configs.

Only shared evaluation artifacts—public detections and precomputed
detections/embeddings—live in `boxmot/configs/artifacts`. An artifact
profile has the same `id` as its dataset:

```yaml
id: mot17

artifacts:
  precomputed:
    ablation:
      uri: hf://example/runs/mot17/ablation
      contains:
        - detections
        - embeddings
      produced_by:
        detector: yolox-x-mot17/ablation
        reid: lmbn-n-duke
```

The resolver attaches the matching artifact profile to the dataset at runtime.
Datasets without public or precomputed artifacts do not need a profile.
The experiment resolver always requires `produced_by.detector`. When an
experiment declares ReID, the artifact must contain embeddings; when the
experiment omits ReID, `produced_by.reid` can select the profile represented by
the cached embeddings.

`box_type: aabb` selects axis-aligned MOT metrics. `box_type: obb` selects rotated
IoU; OBB ground truth is expected in 13-column corner format on disk.
