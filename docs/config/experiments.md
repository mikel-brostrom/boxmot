# Experiments

Experiment configs live under `boxmot/configs/experiments`. They are the small,
user-facing entry points for config-driven runs.

```yaml
id: mot17-ablation-yolox-lmbn

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

evaluation:
  class_map:
    pedestrian: person
```

Tracker selection remains an engine runtime choice and is not embedded in
the experiment.

Class maps use semantic names. The resolver looks up numeric IDs in the dataset
and detector registries and builds the numeric bridge consumed by evaluation.
Use `class_map: auto` when target class names match detector class names.

`reid.crop_strategy` is part of the experiment because the same appearance
model can consume different detection geometry. Use `aabb` for axis-aligned
boxes, `perspective` for a perspective-rectified OBB, `rotated` for the
canonical affine OBB transform, or `mask_aware` when the experiment also
provides masks. The built-in MMOT OBB experiments use `perspective`, matching
the crop transform that produced the published benchmark features.

## Detection sources

`detections.source` is required and must select a model-backed detector:

```yaml
# Run a model
detections:
  source: model
  model:
    ref: yolox-x-mot17
    checkpoint: ablation

```

Legacy public, positional NPY/NPZ, and text-only perception caches are not
experiment sources and cannot be passed directly to materialization. Define a
supported dataset plus model component configs, compose them in an experiment,
then materialize that experiment into a keyed build.

## Built-in examples

Materialize any built-in experiment by ID, then evaluate its exact build:

```bash
boxmot materialize --experiment mot17-ablation-yolox-lmbn
boxmot eval --experiment mot17-ablation-yolox-lmbn --build BUILD_ID --tracker boosttrack
```

Other built-in IDs include `sportsmot-val-yolox-lmbn`,
`mmot-obb-test-yolo11l-lmbn`, and
`mmot-obb-mini-train-yolo11l-lmbn`. Each semantic configuration produces its
own build ID.

## Validation and reproducibility

Resolution fails before downloads or inference when a split, class, checkpoint,
or model reference is missing; box types are incompatible; inference
values are invalid; or evaluation targets a split without ground truth.

The materialized manifest contains the authored and resolved semantic identity,
artifact hashes, source/taxonomy digests, stage fingerprints, and publish
flags. Evaluation outputs record the effective tracker configuration separately.
