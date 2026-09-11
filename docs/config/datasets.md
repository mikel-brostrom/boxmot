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
  val:
    path: val
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
      train: hf://Lekim89/MOT17/train
      val: hf://Lekim89/MOT17/val
      test: hf://Lekim89/MOT17/test
      ablation: hf://Lekim89/MOT17/ablation
```

The class groups make evaluation roles explicit without repeating an
`evaluation` field for every entry. Their numeric IDs must exactly match the
IDs stored in that dataset's ground-truth annotations; they are not implicitly
one-based. Experiment class maps translate detector IDs into this native
dataset domain before materialization. Split properties stay together, so
evaluation can reject a split without ground truth before looking for annotation
files. A dataset's `resources` mapping may contain only its own `dataset`
download.

When a selected split is not already populated locally, materialization and
evaluation download its configured Hugging Face `per_split` resource before
cataloging. Only the active split URI is fetched. A populated local split
remains authoritative and is never replaced automatically. Archive resources
remain explicit downloads and are not materialized implicitly.

`storage.root` is a safe POSIX-style path. Built-in profiles resolve it beneath
`./datasets/mot`; local configs resolve it beside their YAML file. Pass
`--data-root` explicitly to override that base. This keeps downloaded tracking
data outside the importable `boxmot.datasets` Python package and local dataset
folders portable.

Dataset download URIs stay in the dataset config. Detector checkpoint URIs live
in detector configs, and ReID weight URIs live in ReID configs. Perception
artifacts for image tracker replay are published as canonical keyed builds.
Legacy `.npy`, `.npz`, and text-only perception roots remain unsupported in
that workflow. [Multimodal sequence datasets](#multimodal-sequence-datasets) provide the
independent image and spatial observations required by EagerMOT tuning.

A split may declare an `annotations` directory relative to `storage.root` when
ground truth is stored as flat `<sequence>.txt` files beside, rather than
inside, its frame sequences. That path is authoritative for catalog identity
and evaluation. The MMOT profile uses `test/npy` for raw multispectral frames
and `test/mot` for the corresponding OBB annotations.

A split may also declare `sequences`, a non-empty list of exact sequence
directory names. Only those sequences are cataloged; missing names are errors.
This supports train/validation partitions that share a directory of frames.

`box_type: aabb` selects axis-aligned MOT metrics. `box_type: obb` selects rotated
IoU; OBB ground truth is expected in 13-column corner format on disk.

## KITTI MOTS instance masks

The `kitti-mots` profile reads the original KITTI tracking images and MOTS
instance PNGs.

### Download KITTI MOTS data

Download these two archives:

1. **Color images:** On the [KITTI tracking download page](https://www.cvlibs.net/datasets/kitti/eval_tracking.php),
   select **Download left color images of tracking data set (15 GB)**.
   The archive is **`data_tracking_image_2.zip`**. KITTI requires
   [registration and a stated usage purpose](https://www.cvlibs.net/datasets/kitti/user_login.php)
   before downloading.
2. **Segmentation masks:** Under **KITTI MOTS** on the
   [MOTS download page](https://www.vision.rwth-aachen.de/page/mots), select
   **Annotations in png format (train+val)** to download **`instances.zip`**.
   These are the ground-truth instance masks used by BoxMOT for both box and
   segmentation evaluation.

The [MOTS annotations](https://www.vision.rwth-aachen.de/page/mots) are licensed
under CC BY-NC-SA 3.0 (attribution, noncommercial use, and share-alike).

From the repository root, extract archives downloaded to `~/Downloads`:

```bash
mkdir -p datasets/KITTI-MOTS/data_tracking_image_2
unzip ~/Downloads/data_tracking_image_2.zip -d datasets/KITTI-MOTS/data_tracking_image_2
unzip ~/Downloads/instances.zip -d datasets/KITTI-MOTS
```

The image archive contains `training/` and `testing/`; the annotation archive
already contains `instances/`. The resulting layout beneath `--data-root datasets`
is:

```text
KITTI-MOTS/
├── data_tracking_image_2/
│   ├── training/image_02/0000/000000.png
│   └── testing/image_02/0000/000000.png
└── instances/0000/000000.png
```

For example, use `--data-root datasets` when the dataset is at
`datasets/KITTI-MOTS`. If both extracted directories already share a directory such as
`~/Downloads`, copy the dataset YAML, set `storage.root: .`, and use that
directory as `--data-root`. No conversion of images or labels is required.

### Load and evaluate

The profile provides the official 12-sequence `train` and 9-sequence `val`
partitions, `fulltrain` for all 21 annotated sequences, and the unannotated
`test` partition. The sequence selections follow the reference
[train](https://github.com/VisualComputingInstitute/mots_tools/blob/master/mots_eval/train.seqmap)
and [validation](https://github.com/VisualComputingInstitute/mots_tools/blob/master/mots_eval/val.seqmap)
lists. The built-in profile uses `layout: sequence`, with `images` and
`ground_truth` modalities pointing to the original image and instance
directories. Every selected annotated image must have a matching
`<sequence>/<frame>.png` mask; the test split removes the ground-truth modality.
Annotation content and dimensions participate in catalog validation and build
identity. Native frame numbers start at zero, with timestamps at 10 Hz;
`--fps` uses the existing frame sampling behavior.

Load the raw annotated frames directly in Python:

```python
from pathlib import Path

from torch.utils.data import DataLoader

from boxmot.datasets import ImageDataset
from boxmot.datasets.inputs import load_dataset_inputs

inputs = load_dataset_inputs(
    "kitti-mots",
    split="train",
    data_root=Path("datasets"),
    roles=("images", "ground_truth"),
)
dataset = ImageDataset.from_inputs(inputs)
loader = DataLoader(dataset, batch_size=2, collate_fn=list)
sample = next(iter(loader))[0]
frame = sample.frame                 # RGB uint8 [3, H, W]
truth = sample.ground_truth          # canonical Tracks
masks = truth.masks.values           # bool [N, H, W]
track_ids = truth.track_ids          # original encoded instance IDs
ignore = sample.ignore_mask          # bool [H, W]
```

Images and masks are decoded per item, so worker processes do not preload the
dataset. The list collation keeps different image sizes and object counts.
`load_dataset_inputs` applies the selected split's paths and sequence names.
Use `sequence_names=("0002",)` with `split="val"` to select one validation
sequence. Selecting `split="test"` uses its images without ground truth;
`ground_truth` and `ignore_mask` are then `None`.

The loader preserves native classes (`1` car, `2` pedestrian) and full encoded
track IDs, including instances numbered zero. Background (`0`) is excluded,
and ignore pixels (`10000`, class `10`) are returned separately. Boxes tightly
enclose each mask with exclusive upper coordinates. Empty annotated frames
return zero-row `Tracks` and masks shaped `[0, H, W]`.

Materialization supports images and detections, with optional prediction masks.
To publish masks, use `--publish-masks` with a detector that provides masks or
an experiment that defines a segmentor. Published masks come from these
perception stages; annotation masks remain ground truth. Map detector class
names to KITTI names in the experiment, for example
`class_map: {car: car, pedestrian: person}` for a COCO detector.

The native `eval` and `tune` workflows compute HOTA, CLEAR, Identity, and Count
using box IoU by default, deriving ground-truth boxes from the instance PNGs.
Add `--eval-masks` to compute segmentation metrics from predicted masks.
See [MOTS evaluation](../guides/evaluation.md#kitti-mots-evaluation) for commands,
dependencies, and output formats.

## Multimodal sequence datasets

A local `dataset.yaml` uses the same `id`, `format`, `storage`, `classes`,
and `splits` schema as the built-in dataset configs. Set
`format.layout: sequence` and declare the `modalities` to use: images,
ground truth, calibration, ego motion, and saved 2D/3D detections. Each modality
selects its encoding and relative paths. Dataset identity and layout are
independent of the tracker and detector models.

`eval` and `tune` read these inputs through `--dataset ./my-sensor-dataset`.
A folder resolves to its `dataset.yaml`; an explicit YAML path also works.
The bare name `--dataset kitti-mots` selects the built-in image dataset profile
above. Local sequence paths resolve beneath `storage.root`, relative to the
containing YAML. This template uses `root: .` so the folder can move as a unit.

### Bring your own sensor dataset

Copy the [sensor dataset config](https://github.com/mikel-brostrom/boxmot/blob/master/boxmot/configs/datasets/sensor-fusion.yaml)
from `boxmot/configs/datasets` into your dataset folder. From the repository root:

```bash
mkdir -p ./my-sensor-dataset
cp boxmot/configs/datasets/sensor-fusion.yaml ./my-sensor-dataset/dataset.yaml
```

Edit the copied `dataset.yaml` to set your dataset ID and input paths.
Selecting the built-in `--dataset sensor-fusion` directly resolves its
`root: .` beneath `datasets/mot` in the working directory. For another payload
folder, use a local copy: its paths resolve relative to the containing folder.
Saved sensor `eval` and `tune` do not accept `--data-root`.
Populate these payloads for both `drive-001` and `drive-002`:

```text
my-sensor-dataset/
  dataset.yaml
  sequences/recordings/drive-001/
    images/000000.png
    ground_truth/000000.png
    calibration.txt
    poses.npy
  sequences/recordings/drive-002/...
  predictions/
    image/recordings/drive-001.txt
    image/recordings/drive-002.txt
    car/recordings/drive-001/000000.txt
    car/recordings/drive-002/000000.txt
    pedestrian/recordings/drive-001/000000.txt
    pedestrian/recordings/drive-002/000000.txt
```

The complete dataset configuration is:

```yaml
id: my-sensor-dataset
format:
  layout: sequence
  box_type: aabb
storage:
  root: .
classes:
  target:
    car: 1
    pedestrian: 2
  ignore:
    ignore: 10
fps: 10
default_split: val
splits:
  train:
    partition: recordings
    sequences: [drive-001]
    has_ground_truth: true
  val:
    partition: recordings
    sequences: [drive-002]
    has_ground_truth: true
modalities:
  images:
    format: image-directory
    path: sequences/{partition}/{sequence}/images
  ground_truth:
    format: instance-png
    path: sequences/{partition}/{sequence}/ground_truth
    options:
      class_divisor: 1000
      background_id: 0
      ignore_ids: [10000]
  calibration:
    format: kitti-p2
    path: sequences/{partition}/{sequence}/calibration.txt
  poses:
    format: camera-to-world-npy
    path: sequences/{partition}/{sequence}/poses.npy
  detections_2d:
    format: trackrcnn
    path: predictions/image/{partition}/{sequence}.txt
  detections_3d:
    format: kitti-detections
    paths:
      - predictions/car/{partition}/{sequence}
      - predictions/pedestrian/{partition}/{sequence}
    options:
      score_transform: odds
      ignore_classes: [Cyclist]
```

`id` uses lowercase kebab case. Split, partition, and sequence names are safe
directory names, such as `validation`, `recordings`, and `drive-001`; they do
not imply official KITTI membership. Quote numeric names such as `"0002"`.
Paths use `/`, cannot contain `..` or an absolute root, and accept
`{partition}` and `{sequence}` placeholders. The directory names can match
your existing export layout; the YAML selects their meaning and encoding.

Each modality declares `format`, either `path` or `paths`, and optional
parser `options`. For spatial detections, several directories can feed the
same modality, or one directory can contain every class. A split's `modalities`
mapping replaces selected modality declarations; set an entry to `null` to
omit it for that split. The resulting declarations select the inputs for the
experiment. `eval` and `tune` require the selected tracker to consume every
declared tracking input; an input marked `Unused` in its capability matrix is
an incompatibility. Ground truth is consumed separately for scoring or calibration
and is never passed to the tracker. `has_ground_truth` must agree with the
presence of either `ground_truth` or `ground_truth_3d`. Evaluation requires
annotations for the selected metric and split: `ground_truth` for masks, or
`ground_truth_3d` with `--eval-3d`.
Unused scoring annotations are not loaded, even when their modalities remain declared.

Sequence layouts require a finite positive dataset `fps`. This template sets `10`; it
sets timestamps (`frame_index / fps`) and saved video playback speed. Tracking
still advances once per image; this setting does not resample frames or
enable variable-time motion.

### Supported encodings

| Modality / encoding | File contract |
| --- | --- |
| `images` / `image-directory` | PNG frames named `000000.png`, `000001.png`, etc.; contiguous, zero-based, with constant dimensions per sequence |
| `ground_truth` / `instance-png` | Matching single-channel uint16 PNGs encoding `class_id * 1000 + instance_id`; the template uses car `1`, pedestrian `2`, background `0`, ignore `10000` |
| `ground_truth_3d` / `kitti-tracking-labels` | Sequence text file with 17 KITTI tracking label fields, including zero-based frame and stable object identity; required for `eval --eval-3d` or 3D Kalman calibration |
| `calibration` / `kitti-p2` | `P2:` followed by 12 row-major values of a `3 x 4` camera-to-pixel projection |
| `poses` / `camera-to-world-npy` | Numeric `(N, 4, 4)` absolute camera-to-world rigid transforms; identity poses for a stationary camera |
| `detections_2d` / `trackrcnn` | One sequence text file with 138 fields per detection: frame, AABB, score, class, full-image RLE mask, 128 embedding fields |
| `detections_3d` / `kitti-detections` | Six-digit frame text files with 16 KITTI detection fields; dimensions and bottom-face centers in meters, camera x right/y down/z forward, yaw about +y |

The encoding names specify serialization; predictions can come from your own
models. `kitti-detections` uses `options.score_transform: odds` for PointGNN
scores, mapping `s` to `s / (1 + s)`. Use `identity` for scores already in
`[0, 1]`. Set `options.class_map` to translate source labels into dataset class
names or IDs; `ignore_classes: [Cyclist]` explicitly skips that source label in
the PointGNN example. Other undeclared classes are errors. The template README
specifies every field, coordinate convention,
mask encoding, and empty-frame behavior. Format parsers live under
`boxmot/datasets/readers`; adding an encoding belongs there, with consumers
continuing to use canonical observations.

### 3D ground truth for Kalman calibration

To use `eval --eval-3d`, `eval --calibrate-kf`, or `tune --calibrate-kf` with
EagerMOT, add 3D tracking annotations independently of the mask ground truth.
Uncomment the optional `ground_truth_3d` block in the sensor template after
placing a label file for each selected sequence at the declared path:

```yaml
modalities:
  ground_truth_3d:
    format: kitti-tracking-labels
    path: annotations/{partition}/{sequence}.txt
    options:
      ignore_classes: [DontCare, Van, Truck, Cyclist, Person, Person_sitting, Tram, Misc]
```

Each row has exactly 17 whitespace-separated fields, with no detector score:

```text
frame track_id type truncated occluded alpha x1 y1 x2 y2 height width length x y z rotation_y
```

Frame indices align with the zero-based image timeline. Keep nonnegative
object IDs stable across frames, with one row per class/ID in a frame.
Dimensions are positive meters; `(x, y, z)` is the bottom-face center in the
same calibrated camera coordinates as the predictions, and `rotation_y` is
yaw about +y in radians. Retained numeric fields must be finite. The reader
supports `class_map` and explicitly ignored labels as for spatial detections;
ignored `DontCare` rows may use KITTI's placeholder 3D geometry.

Missing object annotations break that object's motion samples; they are not
interpolated. Calibration transforms matched detections and annotations using
the supplied absolute ego poses. It fits filter noise, without adjusting those
poses or changing the selected evaluation metric. See
[EagerMOT calibration](../trackers/eagermot.md#calibrate-3d-kalman-noise) for commands
and saved profiles. Mask evaluation and tuning without calibration do not
require or load this modality.

For spatial scoring, run:

```bash
boxmot eval --dataset ./kitti-mots --tracker eagermot \
  --split val --eval-3d --project runs/kitti-3d
```

This evaluates car and pedestrian 3D boxes with volumetric IoU HOTA, CLEAR,
and Identity metrics and writes `kitti_3d/<sequence>.txt` predictions. It is a
custom evaluation without official KITTI difficulty, visibility, or
DontCare-region rules. Ground-truth instance PNGs are not required in this mode.

### Split-specific inputs

For official KITTI MOTS data, preserve the original four-digit sequence names
and train/validation assignments. Starting with the template's sequence paths,
replace its splits and prediction declarations to match your downloads. For
example, these split definitions select the official partitions and use the
T2 car predictions for validation:

```yaml
splits:
  train:
    partition: training
    has_ground_truth: true
    sequences: ["0000", "0001", "0003", "0004", "0005", "0009", "0011", "0012", "0015", "0017", "0019", "0020"]
  val:
    partition: training
    has_ground_truth: true
    sequences: ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"]
    modalities:
      detections_3d:
        format: kitti-detections
        paths:
          - predictions/pointgnn-car-t2/{partition}/{sequence}
          - predictions/pointgnn-pedestrian/{partition}/{sequence}
        options:
          score_transform: odds
          ignore_classes: [Cyclist]
  fulltrain:
    partition: training
    has_ground_truth: true
    sequences: ["0000", "0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019", "0020"]
```

Set the top-level `detections_2d` path to
`predictions/trackrcnn/{partition}/{sequence}.txt`, and its `detections_3d`
paths to `predictions/pointgnn-car-t3/{partition}/{sequence}` and
`predictions/pointgnn-pedestrian/{partition}/{sequence}`. Training and
`fulltrain` then use T3; validation replaces the entire 3D declaration with
T2. The [KITTI download instructions](#download-kitti-mots-data) describe the
source image and instance archives. Keep checkpoint identifiers and training
data provenance in the dataset README or YAML comments.

### Tune and evaluate

Install the `mots` and `evolve` extras using the
[MOTS evaluation setup](../guides/evaluation.md#kitti-mots-evaluation), then
populate the template's payloads and run:

```bash
boxmot tune --dataset ./my-sensor-dataset --tracker eagermot \
  --split train --n-trials 50 --seed 0
boxmot eval --dataset ./my-sensor-dataset --tracker eagermot \
  --split val --class-config runs/eagermot-tune/train/best.yaml
```

Use the `best.yaml` path printed by tuning; repeated runs increment the split
directory name. Omit `--class-config` to evaluate the initial KITTI presets.
Repeat `--sequence` to select several sequences within a split. No built-in
dataset registration or perception build is required.

Dataset modalities and classes are generic configuration. The current sensor
`eval` and `tune` consumers use EagerMOT with one calibrated camera per sequence.
Both default to car and pedestrian mask metrics; `eval --eval-3d` selects
spatial metrics instead. Mask metrics require predicted and ground-truth
masks, although the Python EagerMOT tracker can use image boxes without masks.
Arbitrary-class metrics and multiple-camera ingestion are not supplied by
these consumers. Use separate sequences for tuning and evaluation. See
[EagerMOT tuning](../trackers/eagermot.md#tune-separate-class-profiles) for
supported options and outputs.

When a tracker selection is incompatible, `eval` and `tune` report a short
reason and next step based on the selected split's declarations and registered
[tracker inputs](../trackers/index.md#input-support). The message names unused
tracking inputs, missing required inputs, or an unavailable backend or workflow.
Silently discarding a declared input would change the configured experiment.
These checks use the configuration; input files are validated separately.
Ground-truth masks do not substitute for
predicted masks, and TrackR-CNN's stored embeddings are not exposed by its reader.

The current direct saved-sensor `eval` and `tune` workflows require
`--tracker eagermot --tracker-backend python`.

To intentionally run an image-only experiment, author a separate dataset config
or explicit split override selecting only `images` and `ground_truth`, then
select a perception build or detector through ordinary `eval` or `tune`.
The payload files can stay in place; the selected configuration must express
which inputs the experiment uses.
