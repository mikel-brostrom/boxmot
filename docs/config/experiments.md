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

## Dataset input selection

A dataset YAML describes available data: paths, encodings, classes, and splits.
`dataset.modalities` in an experiment is an allowlist of the inputs for that run.
Omitted modalities are neither validated on disk nor loaded. An empty mapping
for a role, such as `images: {}`, uses its dataset declaration unchanged.
`source` selects another declared dataset modality for that role; `options`
replaces its reader options. Experiments cannot override paths or formats.
Split-specific paths are applied before this selection, so KITTI validation
still uses its T2 predictions. An omitted `dataset.modalities` uses all declared
inputs, as for datasets that expose only images and annotations.

KITTI has three presets under `experiments/kitti-mots/`, all referencing the
same `kitti-mots.yaml` dataset:

- `full`: all declared inputs, including masks, 3D boxes, calibration, and poses.
- `2d`: saved TrackR-CNN boxes, images, and image-box ground truth.
- `2d-lmbn-n-duke`: the same 2D inputs with LMBN appearance features.

They use the dataset's default validation split. Pass `--split train` to select
training sequences; separate split-specific experiment files are unnecessary.

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

## Saved 2D detections

An evaluation experiment can omit `detector` when its dataset declares saved
2D detections. Those predictions already use the dataset class IDs, so no
detector class map is needed:

```yaml
dataset:
  ref: kitti-mots
  modalities:
    images: {}
    detections_2d:
      options:
        load_masks: false
    ground_truth:
      source: ground_truth_3d
      options: {}
reid:
  ref: lmbn-n-duke
```

Run the shipped preset against an existing KITTI multimodal folder:

```bash
boxmot eval --experiment kitti-mots/2d-lmbn-n-duke \
  --data-root ./kitti-mots --tracker occluboost --cache-inputs \
  --project runs/kitti-2d
```

The `kitti-mots/2d` preset omits ReID; `kitti-mots/2d-lmbn-n-duke`
generates appearance features from the selected images. These experiments run
saved-box `eval` directly and do not support materialization, tuning, or KF
calibration. See the [saved 2D dataset layout](datasets.md#existing-2d-detections)
for required files.

## Saved multimodal KITTI inputs

Choose an experiment according to the inputs your tracker consumes. Both presets
read the same local KITTI folder and reuse saved predictions:

```bash
# 2D boxes: OC-SORT, ByteTrack, and other image-box trackers
boxmot eval --experiment kitti-mots/2d \
  --data-root ./kitti-mots --tracker ocsort --cache-inputs --project runs/kitti-2d

# Images, masks, 2D/3D boxes, calibration, and camera poses: EagerMOT
boxmot eval --experiment kitti-mots/full \
  --data-root ./kitti-mots --tracker eagermot --cache-inputs --project runs/kitti-multimodal
```

The multimodal experiment selects `kitti-mots`. Its dataset config
declares Track R-CNN predictions under `predictions/trackrcnn`, PointGNN boxes
under `predictions/pointgnn-car-t2` (validation), `predictions/pointgnn-car-t3`
(training), and `predictions/pointgnn-pedestrian`, and images,
mask ground truth, calibration, and poses under `sequences/{partition}/{sequence}`.
KITTI tracking labels under `{partition}/label_02` supply 3D ground truth when
`--eval-3d` or `--calibrate-kf` is selected. Default evaluation scores masks.

Use `--split train` for training sequences. The `full` experiment also supports
EagerMOT `tune`. It omits detector and ReID components and cannot be materialized.
Incompatible trackers are rejected before inputs are loaded; select the 2D
experiment for box trackers, using `2d-lmbn-n-duke` when appearance is needed.

## Built-in examples

Materialize a detector-based experiment by its catalog-relative YAML filename,
then evaluate its exact build:

```bash
boxmot materialize --experiment mot17/ablation-yolox-lmbn.yaml
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID --tracker boosttrack
```

For detections, segmentation, and embeddings together, select
`mot17/ablation-yolox-edgetam-lmbn.yaml` and materialize with
`--publish-masks --publish-embeddings`. This experiment uses YOLOX, EdgeTAM, and
LMBN; see [materializing detection masks](../tasks/masks.md#materialize-detection-masks).
Experiments with a standalone segmentor require explicit `--experiment`
selection; direct dataset/detector/ReID selectors select experiments without one.

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
