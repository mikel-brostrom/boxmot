# Custom sensor dataset

Copy the [sensor fusion config](../../../boxmot/configs/datasets/sensor-fusion.yaml)
to describe your own synchronized camera and 3D detections using BoxMOT's
dataset configuration schema. Save it as `dataset.yaml`: add your images,
annotations, calibration, poses, and saved
predictions before running. `format.layout: sequence` selects explicit
sequence modalities. Each modality declares its encoding and relative paths.
The `trackrcnn` and `kitti-detections` names specify encodings; your recordings
and detector models do not need to come from KITTI. Evaluation and tuning
use EagerMOT with car and pedestrian segmentation metrics by default.
`eval --eval-3d` adds official KITTI object AP40 by difficulty and 2D tracking
when exact object and tracking annotations are available.

## Copy and populate

From the repository root:

```bash
mkdir -p ./my-sensor-dataset
cp boxmot/configs/datasets/sensor-fusion.yaml ./my-sensor-dataset/dataset.yaml
```

The supplied config selects `drive-001` for training and `drive-002` for
validation. Populate this layout for **both** sequences:

```text
my-sensor-dataset/
├── dataset.yaml
├── annotations/recordings/drive-001.txt, drive-002.txt
├── sequences/recordings/
│   ├── drive-001/
│   │   ├── images/000000.png, 000001.png, ...
│   │   ├── ground_truth/000000.png, 000001.png, ...
│   │   ├── calibration.txt
│   │   └── poses.npy
│   └── drive-002/
│       ├── images/000000.png, 000001.png, ...
│       ├── ground_truth/000000.png, 000001.png, ...
│       ├── calibration.txt
│       └── poses.npy
└── predictions/
    ├── image/
    │   └── recordings/drive-001.txt, drive-002.txt
    ├── car/
    │   └── recordings/
    │       ├── drive-001/000000.txt, 000001.txt, ...
    │       └── drive-002/000000.txt, 000001.txt, ...
    └── pedestrian/
        └── recordings/
            ├── drive-001/000000.txt, 000001.txt, ...
            └── drive-002/000000.txt, 000001.txt, ...
```

Edit `dataset.yaml` to set your dataset `id`, nominal `fps`, split names,
partitions, and sequence selections. `id` is lowercase kebab case. Split,
partition, and sequence names are directory names, such as `validation`,
`recordings`, and `drive-001`; do not use paths, `.` or `..`, colons, or
surrounding whitespace. Quote numeric names such as `"0002"` in YAML.
Custom splits are explicit selections; names such as `train` and `val` do
not imply official KITTI membership.

The config uses the same `id`, `format`, `storage`, `classes`, and `splits`
fields as BoxMOT's built-in datasets. `modalities` declares the available
`images`, `ground_truth`, `ground_truth_3d`, `calibration`, `poses`, `detections_2d`, and
`detections_3d`, with a commented `ground_truth_objects` example for official AP.
Each entry selects a `format` and either one `path` or a list
of `paths`, with parser settings in `options`. Spatial detections can share
one directory or be combined from several, as in this template. Directory
names describe your data; no detector-name directory conventions are required.

All modality paths resolve beneath `storage.root`, which is `.` relative to
this local `dataset.yaml`. Selecting the built-in `--dataset sensor-fusion`
instead resolves `root: .` beneath `datasets/mot` in the working directory;
use a local copy of the YAML for another payload folder. Saved sensor `eval`
and `tune` do not accept `--data-root`.
Paths use `/`, cannot contain `..` or an absolute
root, and accept `{partition}` and `{sequence}` placeholders. Keeping actual
files inside the folder makes the dataset portable. Use a split's `modalities`
mapping to replace a modality for that split, for example to select predictions
from a different detector checkpoint. Specify the replacement's complete
`format`, `path`/`paths`, and `options`. Set a split modality to `null` to omit
it when its consumer permits that input to be absent.

## Sequence files

### Images and timeline

Save RGB images as PNGs named `000000.png`, `000001.png`, and so on, with no
gaps. Every sequence must contain at least one image, and image dimensions
must remain constant within that sequence. Use six-digit frame names across
images, annotation masks, and 3D predictions. Frame `0` is the first image.
Images define the complete timeline; detector outputs must refer to those
same frames and be synchronized with them.

The dataset `fps` is a required finite positive number for sequence layouts.
The template sets `10`; change it to the nominal rate of your recordings. It sets replay timestamps
(`frame_index / fps`) and saved video playback speed. Sensor tracking advances
once per image; this field does not resample frames or enable variable-time
motion.

### Ground truth

Every image needs a matching, single-channel **uint16 PNG** in
`ground_truth/`, with the same filename and dimensions. Store each object's
pixels as `class_id * 1000 + instance_id`:

| Value | Meaning |
| --- | --- |
| `0` | Background |
| `1000`–`1999` | Car, class `1`, instance `0`–`999` |
| `2000`–`2999` | Pedestrian, class `2`, instance `0`–`999` |
| `10000` | Ignore region |

Keep instance IDs stable across frames and unique within each class and
sequence. IDs may restart in another sequence. Use an all-zero uint16 PNG
for an annotated frame without objects. Missing annotations are errors.
Predicted masks must come from your detector, independently of these labels.

### 3D ground truth

For `eval --calibrate-kf`, `tune --calibrate-kf`, or `eval --eval-3d`, the template's
`ground_truth_3d` modality selects one `annotations/PARTITION/SEQUENCE.txt`
file per selected sequence. Each row has exactly 17 KITTI tracking label fields:

```text
frame track_id type truncated occluded alpha x1 y1 x2 y2 height width length x y z rotation_y
```

Use zero-based image frame indices and stable nonnegative object IDs. Box
coordinates follow the camera convention described below, including bottom
centers, dimensions in meters, and yaw about +y. Annotations have no detection
score. The template explicitly ignores KITTI labels outside the car and
pedestrian classes, including `DontCare`; change this list or use `class_map`
to match your exports. See the [3D annotation contract](../../../docs/config/datasets.md#3d-ground-truth-for-kalman-calibration).

Calibration fits five 3D Kalman covariance scales per class using annotations
and saved detections transformed with the fixed ego poses. The resulting
`kf-tuning/calibrated.yaml` can be reused with `--class-config`. Tuning holds
the fitted noise and angular-motion choice fixed. Ordinary evaluation and
tuning need only the instance masks above; these 3D labels are optional.

Official object AP also requires the template's commented `ground_truth_objects`
modality. Supply one 15-field object-label file per exact image at
`annotations/objects/PARTITION/SEQUENCE/000000.txt`, `000001.txt`, etc.
Keep native fractional truncation, visibility, classes and DontCare rows.
Object labels cannot be derived from tracking truncation categories or paired
with a separately numbered dataset by filename. See the
[exact object-label contract](../../../docs/config/datasets.md#exact-object-labels-for-official-ap)
and [evaluator setup](../../../docs/trackers/eagermot.md#evaluate-3d-tracks).

### Calibration

`calibration.txt` contains one `P2:` entry with the **12 row-major values**
of the camera's `3 x 4` projection matrix:

```text
P2: p00 p01 p02 p03 p10 p11 p12 p13 p20 p21 p22 p23
```

Replace the symbols with your actual finite numbers. The left `3 x 3` block
must be nonsingular. The matrix projects the 3D detection coordinate system
into the saved images: `q = P2 @ [x, y, z, 1]`, then
`pixel = (q[0] / q[2], q[1] / q[2])`. Coordinates follow the rectified camera
convention: x right, y down, z forward. Convert LiDAR detections into this
coordinate system before writing them. Supply rectified images and matching
calibration; the reader does not apply distortion correction or LiDAR
extrinsics. Calibration is fixed for the sequence.

### Poses

`poses.npy` is a real numeric NumPy array of shape `(N, 4, 4)`, with one
**absolute camera-to-world** rigid transform per image. Use finite float32
values, a proper orthonormal rotation with determinant `+1`, and last row
`[0, 0, 0, 1]`. Translation is in meters, in the same scale as the 3D boxes.
The transform at array index `i` maps frame `i`'s camera coordinates into a
single fixed world coordinate system. Relative odometry increments and
world-to-camera matrices must be converted before saving; the reader does
not accumulate or invert them. For a stationary camera, provide `N` identity
matrices. Poses are required for this folder workflow.
EagerMOT transforms box centers using the full rigid pose but represents box
orientation with yaw only; roll and pitch are approximated.

## Prediction files

The file formats below accept predictions from any model exported to their
contracts. No detector inference or perception build runs during sensor
evaluation or tuning. Every declared spatial prediction directory and the
image prediction file must exist for every selected sequence, even if empty.

### Image detections: TrackR-CNN text format

Write one whitespace-separated row per detection in
`predictions/image/recordings/SEQUENCE.txt`. Each row has **138 fields**:

```text
frame x1 y1 x2 y2 score class_id mask_height mask_width mask_rle embedding_0 ... embedding_127
```

- `frame` is the zero-based image index, with no tracking ID column.
- `(x1, y1, x2, y2)` is a finite float32 AABB in image pixels with positive
  area, using top-left and bottom-right coordinates. Use exclusive maximum
  coordinates, consistent with masks. `score` is between `0` and `1`.
- `class_id` is `1` for car or `2` for pedestrian.
- `mask_height` and `mask_width` are integers matching the full image size.
- `mask_rle` is a single ASCII token containing compressed COCO RLE counts
  for that detection's full-resolution binary mask, encoded in column-major
  order. Export the ASCII `counts` from COCO mask encoding, without quotes,
  a Python bytes prefix, or the enclosing JSON object. Runs must cover
  exactly `mask_height * mask_width` pixels.
- The final 128 fields are the appearance embedding. Sensor replay ignores
  them, but all fields must be present; write 128 zeros if your model has no
  embedding.

For a frame without image detections, omit its rows. An empty sequence
prediction file is valid; a missing file is an error. Empty masks are read
but omitted from emitted MOTS predictions. Prediction rows may not reference
frames outside the image timeline.

### Spatial detections: PointGNN/KITTI text format

Write one whitespace-separated row per detection in each configured directory's
`recordings/SEQUENCE/FRAME.txt`. The frame filename has exactly six digits,
such as `000000.txt`. Each row has **16 fields**:

```text
type truncated occluded alpha x1 y1 x2 y2 height width length x y z rotation_y score
```

- Use `Car` in the car set and `Pedestrian` in the pedestrian set. These map
  to IDs `1` and `2` in this template. The 3D parser can read other class names
  declared in the dataset, but the current EagerMOT mask evaluation workflow
  evaluates cars and pedestrians only. The template explicitly sets
  `options.ignore_classes: [Cyclist]` for existing PointGNN exports. Other
  undeclared source classes are errors. Use `options.class_map` to map source
  labels to your dataset's class names or IDs when the names differ.
- All remaining fields are finite numbers. `truncated`, `occluded`, `alpha`,
  and the four image-box values are retained format fields; replay does not
  use them for spatial geometry. Use finite placeholders if unavailable.
- `height`, `width`, and `length` are positive dimensions in meters. `(x, y, z)`
  is the **bottom-face center** in the calibrated camera coordinate system,
  in meters. `rotation_y` is yaw in radians about camera `+y`. The tracker
  reads these into `(x, y, z, yaw, length, width, height)`.
- With `options.score_transform: odds`, `score` must be nonnegative and is
  mapped to `score / (1 + score)`, even when the original value is already
  between `0` and `1`. Use `score_transform: identity` for probabilities in
  `[0, 1]` that should retain their values. Tuned 3D thresholds use the
  transformed score.

An empty or missing frame text file means no observations from that spatial
set on that frame; the sequence directory must still exist. Every image
advances tracking, including when both sensor streams have no observations.
Frame filenames outside the image timeline are errors.

### Provenance

Record each prediction source's model/version, checkpoint identifier, and
training datasets or sequence lists in your dataset README or YAML comments.
BoxMOT does not independently verify this information. Record different
checkpoint training sets accurately; use held-out sequences to evaluate tuned
settings.

## Evaluate and tune

Install the evaluation and tuning dependencies described in the
[MOTS evaluation guide](../../../docs/guides/evaluation.md#kitti-mots-evaluation).
From the repository root, after adding your payloads:

```bash
uv run --no-sync boxmot eval \
  --dataset ./my-sensor-dataset --tracker eagermot --split val

uv run --no-sync boxmot tune \
  --dataset ./my-sensor-dataset --tracker eagermot --split train \
  --n-trials 50 --seed 0

uv run --no-sync boxmot eval \
  --dataset ./my-sensor-dataset --tracker eagermot --split val \
  --class-config runs/eagermot-tune/train/best.yaml
```

Tune writes `best.yaml` under a new split run directory (`train`, then
`train2`, and so on); use the path printed by your run. Add
`--sequence drive-002` to select one validation sequence, or `--save --show-3d`
to save videos with projected cuboids. An absolute dataset folder or
`dataset.yaml` path also works from another working directory. No registration
in BoxMOT's built-in dataset catalog is needed.

This workflow supports one calibrated camera per sequence and evaluates car
and pedestrian **segmentation tracking**. It requires ground truth, image
detections, and spatial detections for evaluation and tuning. The dataset
class catalog is configurable, but these workflows do not
provide arbitrary-class metrics, multi-camera ingestion, raw LiDAR processing,
or 3D ground-truth box metrics. The initial EagerMOT profiles are KITTI presets;
use your training split to tune them for your data.
