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
artifacts are published only as canonical keyed builds. Legacy `.npy`, `.npz`,
and text-only perception roots remain unsupported.

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
instance PNGs from this layout beneath `--data-root`:

```text
KITTI-MOTS/
├── data_tracking_image_2/
│   ├── training/image_02/0000/000000.png
│   └── testing/image_02/0000/000000.png
└── instances/0000/000000.png
```

Extract the [KITTI tracking images](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)
and [MOTS instance annotations](https://www.vision.rwth-aachen.de/page/mots)
locally. If both extracted directories already share a directory such as
`~/Downloads`, copy the dataset YAML, set `storage.root: .`, and use that
directory as `--data-root`. No conversion of images or labels is required.

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
