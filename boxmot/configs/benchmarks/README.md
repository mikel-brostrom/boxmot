# Benchmark Configs

This directory contains the benchmark configuration files used by the `track`,
`eval`, and `tune` commands when `--source` points to a named benchmark such as
`mot17-ablation` or `mmot-obb`.

## Naming

- Filenames are lowercase and use kebab-case.
- `id` should match the filename stem.
- `id` is the canonical benchmark identifier used for config lookup.

Example:

```yaml
id: mot17-ablation
```

## Top-Level Blocks

Each benchmark config can define up to four top-level blocks:

### `download`

Where BoxMOT should fetch the benchmark assets from.

```yaml
download:
  runs_url: "https://..."
  dataset_url: "https://..."
```

- `dataset_url`: archive or remote dataset location for the benchmark itself.
- `runs_url`: optional archive with precomputed detections, embeddings, or runs.

Leave a URL empty or `null` when there is nothing to download for that field.

### `storage`

Where the dataset should live locally after download/extraction.

```yaml
storage:
  root: "boxmot/engine/trackeval/data/MOT17-ablation"
  split: "train"
```

- `root`: local dataset root.
- `split`: subdirectory under `root` used at runtime.

BoxMOT resolves the active source path as:

```text
<storage.root>/<storage.split>
```

### `evaluation`

How the benchmark should be interpreted by BoxMOT and TrackEval.

```yaml
evaluation:
  box_type: aabb
  layout: mot
  tracker_eval: mot_challenge
  classes:
    eval:
      1: pedestrian
    distractor:
      2: person_on_vehicle
    mapping: {}
```

- `box_type`: geometry used by the benchmark.
  - `aabb`: axis-aligned boxes
  - `obb`: oriented boxes
- `layout`: on-disk dataset layout that BoxMOT expects.
  - examples: `mot`, `visdrone`, `mmot_rgb`
- `tracker_eval`: TrackEval adapter/runner family.
  - examples: `mot_challenge`, `mmot_rgb`

### `evaluation.classes.eval`

Benchmark class ids and names as they appear in GT and evaluation summaries.

These ids are benchmark ids, not detector ids.

### `evaluation.classes.distractor`

GT-only classes that exist in the annotations but should be ignored during
scoring.

### `evaluation.classes.mapping`

Optional class-name translation from benchmark classes to detector classes.

This is only needed when the benchmark class names differ from the detector
class names. The mapping is name-based, not id-based.

Example:

```yaml
evaluation:
  classes:
    eval:
      1: pedestrian
    mapping:
      pedestrian: person
```

That means:

- benchmark GT class: `pedestrian`
- detector class: `person`

If `mapping` is empty, BoxMOT falls back to positional mapping by class order.
That is fine when the benchmark-default detector already uses the same taxonomy
and ordering as the benchmark.

### `detector`

Default detector settings for the benchmark.

```yaml
detector:
  default_model: "models/yolox_x_MOT17_ablation.pt"
  model_url: "https://..."
  imgsz: [800, 1440]
  conf: 0.01
  classes:
    0: person
```

- `default_model`: detector weight path used for this benchmark by default.
- `model_url`: optional download URL for the default model.
- `imgsz`: default image size for the benchmark detector.
- `conf`: default confidence threshold for the benchmark detector.
- `classes`: detector class ids and names used for class filtering/remapping.

Detector class ids are usually zero-based, unlike many benchmark GT ids.

## How Detector Defaults Are Applied

If the user:

- does not explicitly override `--yolo-model`, or
- explicitly passes the same model as `detector.default_model`

then BoxMOT uses the detector settings from the benchmark config.

If the user passes a different detector model explicitly, that explicit model
wins.

## AABB vs OBB

Use:

- `box_type: aabb` for standard MOT-style axis-aligned benchmarks
- `box_type: obb` for oriented-box benchmarks such as `mmot-obb`

For OBB benchmarks:

- tracker-facing detections are `xc, yc, w, h, angle`
- TrackEval-facing files are written as corner points:
  `x1, y1, x2, y2, x3, y3, x4, y4`

## Minimal Examples

### AABB benchmark

```yaml
id: mot17-ablation

download:
  runs_url: "https://..."
  dataset_url: "https://..."

storage:
  root: "boxmot/engine/trackeval/data/MOT17-ablation"
  split: "train"

evaluation:
  box_type: aabb
  layout: mot
  tracker_eval: mot_challenge
  classes:
    eval:
      1: pedestrian
    distractor:
      2: person_on_vehicle
    mapping: {}

detector:
  default_model: "models/yolox_x_MOT17_ablation.pt"
  model_url: "https://..."
  imgsz: [800, 1440]
  conf: 0.01
  classes:
    0: person
```

### OBB benchmark

```yaml
id: mmot-obb

storage:
  root: "boxmot/engine/trackeval/data/MMOT-OBB"
  split: "train"

evaluation:
  box_type: obb
  layout: mmot_rgb
  tracker_eval: mmot_rgb
  classes:
    eval:
      1: car
      2: bike
      3: pedestrian
    distractor: {}
    mapping: {}

detector:
  default_model: "models/yolo11l-3ch.pt"
  model_url: "https://..."
  imgsz: [1024, 1024]
  conf: 0.2
  classes:
    0: car
    1: bike
    2: pedestrian
```
