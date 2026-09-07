# Experiments

Experiment configs live under `boxmot/configs/experiments`. They are the small,
user-facing entry points for config-driven runs.

```yaml
dataset:
  ref: mot17
  split: ablation

detector:
  ref: yolox-x-mot17
  checkpoint: ablation

reid:
  ref: lmbn-n-duke

evaluation:
  class_map:
    pedestrian: person
```

The YAML filename is the experiment selector. Built-in experiments use their
catalog-relative filename, such as `mot17/ablation-yolox-lmbn.yaml`, while an
explicit external YAML uses its own path. Experiment files do not define an
`id`; the stable manifest identity is derived from the filename.

Tracker selection remains an engine runtime choice and is not embedded in
the experiment.

Class maps use semantic names. The resolver looks up numeric IDs in the dataset
and detector registries and builds the numeric bridge consumed by evaluation.
Use `class_map: auto` when target class names match detector class names.

ReID crop extraction follows the resolved detection geometry automatically.
AABB detections use clipped axis-aligned crops, while OBB detections use the
canonical rectified OBB transform. Experiments do not select a crop strategy.
Built-in ReID profiles do not consume detection masks; a custom mask-dependent
encoder declares that requirement through its encoder contract.

## Detector profiles

The top-level `detector` block selects a reusable detector profile and one of
its named checkpoints:

```yaml
detector:
  ref: yolox-x-mot17
  checkpoint: ablation

```

Legacy public, positional NPY/NPZ, and text-only perception caches are not
experiment sources and cannot be passed directly to materialization. Define a
supported dataset plus detector component configs, compose them in an
experiment, then materialize that experiment into a keyed build.

## Built-in examples

Materialize any built-in experiment by its catalog-relative YAML filename,
then evaluate its exact build:

```bash
boxmot materialize --experiment mot17/ablation-yolox-lmbn.yaml
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID --tracker boosttrack
```

Other built-in files include `sportsmot/val-yolox-lmbn.yaml`,
`mmot-obb/test-yolo11l-lmbn.yaml`, and
`mmot-obb-mini/train-yolo11l-lmbn.yaml`. Each semantic configuration produces
its own build ID.

## Validation and reproducibility

Resolution fails before downloads or inference when a split, class, checkpoint,
or model reference is missing; box types are incompatible; inference
values are invalid; or evaluation targets a split without ground truth.

The materialized manifest contains the authored and resolved semantic identity,
artifact hashes, source/taxonomy digests, stage fingerprints, and publish
flags. Evaluation outputs record the effective tracker configuration separately.
