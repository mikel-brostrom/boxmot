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

`storage.root` is a safe POSIX-style path relative to the selected raw-data
root. That root defaults to `./datasets/mot`; pass `--data-root` explicitly to
use another location. This keeps downloaded tracking data outside the
importable `boxmot.datasets` Python package.

Dataset download URIs stay in the dataset config. Detector checkpoint URIs live
in detector configs, and ReID weight URIs live in ReID configs. Perception
artifacts for image tracker replay are published as canonical keyed builds.
Legacy `.npy`, `.npz`, and text-only perception roots remain unsupported in
that workflow. [KITTI fusion datasets](#kitti-fusion-datasets) provide the
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
lists. For `layout: kitti-mots`, `annotations` points to the instance directory;
every selected image must have a matching `<sequence>/<frame>.png` mask.
Annotation content and dimensions participate in catalog validation and build
identity. Native frame numbers start at zero, with timestamps at 10 Hz;
`--fps` uses the existing frame sampling behavior.

Load the raw annotated frames directly in Python:

```python
from pathlib import Path

from torch.utils.data import DataLoader

from boxmot.datasets import KittiMotsDataset
from boxmot.datasets.config import load_dataset_config

root = Path.home() / "Downloads"
profile = load_dataset_config("kitti-mots")
dataset = KittiMotsDataset(
    root / "data_tracking_image_2/training/image_02",
    root / "instances",
    split="train",
    sequence_ids=profile["splits"]["train"]["sequences"],
)
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
Omitting `sequence_ids` loads every sequence under the supplied image root;
`split` labels sample identities and does not select a partition by itself.
For test images, omit `instances_root`; `ground_truth` and `ignore_mask` are
then `None`.

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

## KITTI fusion datasets

`boxmot tune --dataset ./kitti-mots --tracker eagermot` reads sequence data
and saved predictions from a local dataset. Each sequence owns its images,
ground-truth masks, calibration, and ego poses. Prediction sets have their own
manifests, and `replay.yaml` selects which sets to use for each split.

Use `--dataset ./kitti-mots` to select this local directory. The bare name
`--dataset kitti-mots` selects the built-in image dataset profile described
above.

```text
kitti-mots/
  dataset.yaml
  replay.yaml
  sequences/
    training/0002/
      images/000000.png
      ground_truth/000000.png
      calibration.txt
      poses.npy
    testing/0002/
      images/000000.png
      calibration.txt
  predictions/
    trackrcnn/
      manifest.yaml
      training/0002.txt
    pointgnn-car-t2/
      manifest.yaml
      training/0002/000000.txt
    pointgnn-car-t3/
      manifest.yaml
      training/0002/000000.txt
    pointgnn-pedestrian/
      manifest.yaml
      training/0002/000000.txt
```

Sequence and frame names use four and six digits. The training partition
contains all 21 annotated KITTI sequences; the testing partition preserves
the 29 image sequences and calibration files. Store image and annotation
files in the dataset so it can move independently of the original downloads.
The [KITTI download instructions](#download-kitti-mots-data) describe the
source image and instance archives.

### Dataset manifest

`dataset.yaml` defines classes, sequence locations, and official splits:

```yaml
format: kitti-fusion
version: 1
id: kitti-mots
classes:
  1: car
  2: pedestrian
default_split: val
replay: replay.yaml

sequence_layout:
  images: sequences/{partition}/{sequence}/images
  ground_truth: sequences/{partition}/{sequence}/ground_truth
  calibration: sequences/{partition}/{sequence}/calibration.txt
  poses: sequences/{partition}/{sequence}/poses.npy

splits:
  val:
    partition: training
    sequences: ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"]
  train:
    partition: training
    sequences: ["0000", "0001", "0003", "0004", "0005", "0009", "0011", "0012", "0015", "0017", "0019", "0020"]
  fulltrain:
    partition: training
    sequences: ["0000", "0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019", "0020"]
```

The `val` and `train` splits select 9 and 12 sequences from `training`;
`fulltrain` selects all 21. Ground truth, poses, and 2D predictions are
required for EagerMOT evaluation and tuning. The preserved testing partition
does not have these inputs and is not an evaluation or tuning split.

### Replay selection

`replay.yaml` connects each dataset split to three prediction manifests:

```yaml
version: 1
splits:
  val:
    image: predictions/trackrcnn/manifest.yaml
    car: predictions/pointgnn-car-t2/manifest.yaml
    pedestrian: predictions/pointgnn-pedestrian/manifest.yaml
  train:
    image: predictions/trackrcnn/manifest.yaml
    car: predictions/pointgnn-car-t3/manifest.yaml
    pedestrian: predictions/pointgnn-pedestrian/manifest.yaml
  fulltrain:
    image: predictions/trackrcnn/manifest.yaml
    car: predictions/pointgnn-car-t3/manifest.yaml
    pedestrian: predictions/pointgnn-pedestrian/manifest.yaml
```

To change detector inputs, add a prediction set and update the relevant
reference in `replay.yaml`. The dataset's sequence files and split definitions
stay independent of that choice. T2 car predictions cover the nine validation
sequences; T3 car predictions cover all 21 training sequences.

### Prediction manifests

Each `predictions/NAME/manifest.yaml` declares its format, classes, file layout,
sequence coverage, and provenance. For example, the T2 car manifest is:

```yaml
format: pointgnn
version: 1
id: pointgnn-car-t2
classes:
  1: car
path: "{partition}/{sequence}"
sequences:
  training: ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"]
provenance:
  source_directory: results_tracking_car_auto_t2_train
  training: not-independently-verified
```

PointGNN stores one `FRAME.txt` file per frame below the expanded path.
TrackR-CNN uses `format: trackrcnn` and
`path: "{partition}/{sequence}.txt"`, with exactly the classes `1: car` and
`2: pedestrian`. PointGNN car manifests require `1: car`, and the pedestrian
manifest requires `2: pedestrian`; PointGNN manifests may additionally
declare `3: cyclist` when present in the source predictions. Each manifest
must declare the partition and sequence coverage of its prediction files.

Paths resolve from the directory containing each manifest: sequence paths
from `dataset.yaml`, prediction manifest references from `replay.yaml`, and
prediction payload paths from their own `manifest.yaml`. The
`provenance.source_directory` field records the original source; runtime
replay uses the local payload paths. Prediction training provenance has not
been independently verified. Testing prediction files can be preserved in
their respective sets even though the dataset's testing partition is not
eligible for tuning.

### Tune and evaluate

Install the `mots` and `evolve` extras using the
[MOTS evaluation setup](../guides/evaluation.md#kitti-mots-evaluation), then
pass the dataset directory or `dataset.yaml` to the command:

```bash
boxmot tune --dataset ./kitti-mots --tracker eagermot --n-trials 50 --seed 0
boxmot tune --dataset /path/to/kitti-mots/dataset.yaml \
  --tracker eagermot --split val --sequence 0002 --n-trials 1 --seed 0
boxmot eval-eagermot --dataset ./kitti-mots --split val
```

`--split` selects a dataset split and `--sequence` selects one of its sequences;
repeat `--sequence` for several. Fusion tuning replays the selected prediction
sets and optimizes class-average mask HOTA on CPU. These manifests describe
the sequence and prediction files used by the fusion workflow. See
[EagerMOT tuning](../trackers/eagermot.md#tune-separate-class-profiles) for
supported options and saved profiles.
