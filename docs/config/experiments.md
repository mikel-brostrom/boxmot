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

evaluation:
  class_map:
    pedestrian: person
```

Tracker selection remains a CLI or API runtime choice and is not embedded in
the experiment.

Class maps use semantic names. The resolver looks up numeric IDs in the dataset
and detector registries and builds the numeric bridge consumed by evaluation.
Use `class_map: auto` when target class names match detector class names.

## Detection sources

`detections.source` is required and selects one validated shape:

```yaml
# Run a model
detections:
  source: model
  model:
    ref: yolox-x-mot17
    checkpoint: ablation

# Use public detections from the dataset artifact profile
detections:
  source: public
  name: frcnn

# Use a precomputed artifact from the dataset artifact profile
detections:
  source: precomputed
  artifact: ablation
```

Public and precomputed names must exist in the selected dataset's matching
artifact profile under `boxmot/configs/artifacts`.
Every precomputed artifact must contain detections and declare
`produced_by.detector`. Embeddings are required when the experiment declares a
ReID profile or the artifact declares `produced_by.reid`; in the latter case,
the resolver can infer the ReID profile from that lineage.

## Built-in examples

```bash
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack
boxmot eval --experiment mot17-ablation-frcnn-lmbn --tracker boosttrack
boxmot eval --experiment mot17-ablation-precomputed --tracker boosttrack
boxmot eval --experiment sportsmot-val-yolox-lmbn --tracker boosttrack
boxmot eval --experiment mmot-obb-test-yolo11l-lmbn --tracker botsort
boxmot eval --experiment mmot-obb-mini-train-yolo11l-lmbn --tracker botsort
```

## Validation and reproducibility

Resolution fails before downloads or inference when a split, class, checkpoint,
public source, or artifact is missing; box types are incompatible; inference
values are invalid; or evaluation targets a split without ground truth.

Each experiment-driven evaluation result directory contains:

- `config.source.yaml`: the authored experiment
- `config.resolved.yaml`: expanded dataset/model/class information plus the effective tracker, backend, tracker parameters, and runtime overrides
