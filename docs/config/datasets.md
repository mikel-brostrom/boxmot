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
  root: MOT17

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
`evaluation` field for every entry. Their numeric IDs must exactly match the
IDs stored in that dataset's ground-truth annotations; they are not implicitly
one-based. Experiment class maps translate detector IDs into this native
dataset domain before materialization. Split properties stay together, so
evaluation can reject a split without ground truth before looking for annotation
files. A dataset's `resources` mapping may contain only its own `dataset`
download.

`storage.root` is a safe POSIX-style path relative to the selected raw-data
root. That root is `--data-root`, then `BOXMOT_DATASETS_DIR`, then
`platformdirs.user_cache_path("boxmot") / "datasets"`. Repository-local
dataset folders are not auto-discovered, moved, or deleted.

Dataset download URIs stay in the dataset config. Detector checkpoint URIs live
in detector configs, and ReID weight URIs live in ReID configs. Perception
artifacts are published only as canonical keyed builds. Legacy `.npy`, `.npz`,
and text-only perception roots remain unsupported.

A split may declare an `annotations` directory relative to `storage.root` when
ground truth is stored as flat `<sequence>.txt` files beside, rather than
inside, its frame sequences. That path is authoritative for catalog identity
and evaluation. The MMOT profile uses `test/npy` for raw multispectral frames
and `test/mot` for the corresponding OBB annotations.

`box_type: aabb` selects axis-aligned MOT metrics. `box_type: obb` selects rotated
IoU; OBB ground truth is expected in 13-column corner format on disk.
