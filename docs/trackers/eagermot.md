# EagerMot

[Paper: EagerMOT: 3D Multi-Object Tracking via Sensor Fusion](https://arxiv.org/abs/2104.14682)

EagerMOT associates 3D detections with image detections, tracks objects in 3D,
and uses remaining image observations to sustain tracks when 3D association
fails. BoxMOT integrates the
[authors' Python implementation](https://github.com/aleksandrkim61/EagerMOT)
under `boxmot/trackers/eagermot`.

## Required inputs

Use the Python tracker API with independent `Detections` and `Detections3D`
batches plus a `CameraModel` on every update. Both batches share a `sample_id`,
but their rows and lengths are independent. Pass an explicitly empty batch
when either detector has no observations. Map both detectors into the same
class-ID catalog before fusion.

- Image detections use AABB geometry. Instance masks and a `Frame` are optional;
  the algorithm does not require image pixels or ReID embeddings.
- `Boxes3D` stores CPU-contiguous `float32[N,7]` values in the order
  `(x, y, z, yaw, length, width, height)`. Position is the bottom-face center,
  dimensions are meters, and yaw is radians. Camera axes are x right, y down,
  z forward, with yaw about +y.
- `CameraModel.projection` is a CPU-contiguous `float32[3,4]` matrix that projects
  camera coordinates to pixels. `image_size` is `(height, width)`.
- For a moving camera, supply its current `camera_to_world` rigid transform.
  Full rotation and translation transform box centers. The tracking state
  retains yaw-only cuboids, so their roll and pitch are approximated. Omit the
  pose only for a stationary camera, and keep pose availability consistent
  throughout a sequence.

Use `create_tracker(TrackerSpec("eagermot"))` or `boxmot.EagerMot` for the
Python API. `boxmot tune --dataset ./kitti-mots --tracker eagermot` reads
these inputs from a [KITTI fusion dataset](../config/datasets.md#kitti-fusion-datasets).
The dedicated `eval-eagermot` and `tune-eagermot` commands also accept
`--dataset ./kitti-mots`. Image tracking, `TrackingPipeline`, and
cached perception replay cannot supply the required sensor inputs and reject
`eagermot` before running perception.

## Evaluate downloaded KITTI predictions

Run from the repository root with the existing environment and the `mots`
extra installed. This replays saved detector predictions on CPU and evaluates
**segmentation tracking** against KITTI MOTS instance PNGs:

```bash
uv run --no-sync python -m boxmot.engine.cli eval-eagermot \
  --dataset ./kitti-mots \
  --split val \
  --project runs/eagermot
```

The dataset stores each sequence's observations together and declares saved
predictions separately:

```text
kitti-mots/
  dataset.yaml
  replay.yaml
  sequences/training/0002/
    images/000000.png
    ground_truth/000000.png
    calibration.txt
    poses.npy
  predictions/
    trackrcnn/manifest.yaml
    trackrcnn/training/0002.txt
    pointgnn-car-t2/manifest.yaml
    pointgnn-car-t2/training/0002/000000.txt
    pointgnn-car-t3/manifest.yaml
    pointgnn-car-t3/training/0002/000000.txt
    pointgnn-pedestrian/manifest.yaml
    pointgnn-pedestrian/training/0002/000000.txt
```

Sequence names have four digits and frame names have six digits. The default
MOTS validation split is `0002, 0006, 0007, 0008, 0010, 0013, 0014, 0016, 0018`.
Add `--sequence 0002` for a smaller run; repeat the option to select several
sequences. To evaluate all 21 annotated sequences, use `--split fulltrain`.
The supplied `replay.yaml` selects the T2 car predictions for validation and
T3 for training or full training. Edit its prediction manifest references to
change detector inputs. See the [dataset layout and manifests](../config/datasets.md#kitti-fusion-datasets).
Testing images and calibration are retained, but this partition lacks the
ground truth, ego poses, and 2D predictions needed for evaluation and tuning.

Results are written under `runs/eagermot/val` (then `val2`, and so on):

- `metrics.json`: mask HOTA, DetA, AssA, LocA, CLEAR and Identity metrics,
  including per-sequence results and class/detection averages. Percentage
  metrics use the 0–100 scale, with signed CLEAR scores where applicable.
- `metrics.csv`: combined results for each class and aggregate.
- `mots/SEQUENCE.txt`: official MOTS predictions with real, disjoint masks.
- `videos/SEQUENCE.mp4`: annotated video when `--save` is enabled.
- `run.json`: dataset, replay, and prediction manifest paths, selected
  sequences, and tracker presets.

By default, the runner uses the released car and pedestrian presets separately, with
globally unique output identities. Image confidence resolves overlapping
masks; empty masks are omitted. Every image frame advances tracking, including
frames with no TrackR-CNN rows or missing PointGNN files. RGB pixels are not
needed by EagerMOT; image headers establish the timeline and dimensions.
Previewing or saving annotated videos reads the images for rendering.

PointGNN scores can exceed one. The reader maps each nonnegative score `s` to
`s / (1 + s)` for the canonical score contract. This is a bounded ranking
score, not a calibrated probability; the KITTI presets retain the zero 3D
score threshold, so this mapping does not change accepted detections or mask
association. Ego-motion arrays contain absolute camera-to-world poses and
are indexed directly, without accumulation.

These results do not reproduce the paper's benchmark setup: the image
predictions come from TrackR-CNN, detector checkpoint training provenance has
not been independently verified, and pedestrian detections use the supplied
`trainval` variant. This command evaluates masks; KITTI 3D box evaluation
requires separate 3D ground-truth labels and an evaluator.

## Preview or save a sequence

Use `--show` to preview tracked masks, identities and classes, and `--save` to
write an annotated MP4 for each selected sequence. Add `--show-3d` to overlay
the estimated 3D bounding boxes:

```bash
uv run --no-sync python -m boxmot.engine.cli eval-eagermot \
  --dataset ./kitti-mots \
  --class-config runs/eagermot-tune/val/best.yaml \
  --sequence 0016 \
  --show \
  --save \
  --show-3d \
  --project runs/eagermot-3d-preview
```

`--class-config` loads the saved profiles once when evaluation starts. If the
tuner is still running, this uses a snapshot of its best completed trial;
later improvements do not change the ongoing evaluation. Omit `--class-config`
to use the default KITTI car and pedestrian presets.

This command writes `runs/eagermot-3d-preview/val/videos/0016.mp4` at 10 FPS along
with the normal masks and metrics. Repeated runs create `val2`, and so on.
Remove `--show` on a headless machine. Press **q** or **Esc** to close the
preview while evaluation and video saving continue.

`--show-3d` requires `--show` or `--save`. It projects EagerMOT's current 3D
track estimates through the sequence's full camera projection matrix. The
cuboids use PointGNN observations and the Kalman motion state, including
estimates sustained by a 2D observation when a 3D detection is missing.
Only confirmed tracks updated by at least one sensor in the current frame
are shown. A visible 3D-only track can appear without a mask; boxes behind the
camera are skipped. Cuboids and masks share the same global track IDs and
colors. The yaw-only box approximation described below still applies.
This overlay affects visualization only; MOTS predictions, metrics and tuning
remain unchanged.

## Tune separate class profiles

With the `mots` and `evolve` extras installed, tune the `kitti-mots` dataset
through the main command:

```bash
boxmot tune \
  --dataset ./kitti-mots \
  --tracker eagermot \
  --n-trials 50 \
  --seed 0
```

The [dataset manifests](../config/datasets.md#kitti-fusion-datasets) record
sequence locations, classes, official splits, and prediction choices. Images
and ground truth are stored inside the dataset, so it can move independently
of the original downloads. You can pass
`--dataset /path/to/kitti-mots/dataset.yaml` from any working directory;
relative paths remain anchored to their containing manifest.

The dataset's default split is used unless you pass `--split`. `replay.yaml`
selects the image, car, and pedestrian predictions for each split. Every trial
evaluates car and pedestrian profiles together and maximizes their
**class-average mask HOTA**.

The sensor workflow uses serial CPU execution and `--search-alg optuna`.
Explicit device and concurrency settings must preserve that execution mode,
and objective selectors must use `HOTA`. Perception and build options,
`--calibrate-kf`, and `--resume-tune` are unavailable for fusion datasets.
`--project` changes the results root from `runs/eagermot-tune`.

The standalone command accepts the same dataset:

```bash
uv run --no-sync python -m boxmot.engine.cli tune-eagermot \
  --dataset ./kitti-mots \
  --split val \
  --n-trials 50 \
  --seed 0 \
  --project runs/eagermot-tune
```

The trial count includes the first trial with the default KITTI car and
pedestrian profiles. Remaining trials independently sample each class's
parameters from the shared EagerMOT YAML search ranges. Distance and 3D IoU
thresholds are sampled only for their corresponding matching methods;
`max_age_2d` and `asso_func` stay fixed. Trials run serially on CPU using saved
predictions. The tuned `det_thresh_3d` applies to the transformed PointGNN
score `s / (1 + s)`.

`--sequence 0002` restricts tuning to one sequence; repeat it for several.
Results go to `runs/eagermot-tune/val`, then `val2`, and so on:

- `study.sqlite3`: the Optuna study and trial history.
- `run.json`: input selection, sampling settings and search metadata.
- `best.yaml`: complete scalar tracker profiles under `car` and `pedestrian`.
- `trials/0000/metrics.json` and `trials/0000/mots/SEQUENCE.txt`: metrics and
  mask predictions for each numbered trial, starting with the baseline.

Pass the winning profiles to evaluation using `--class-config`:

```bash
uv run --no-sync python -m boxmot.engine.cli eval-eagermot \
  --dataset ./kitti-mots \
  --split val \
  --class-config runs/eagermot-tune/val/best.yaml \
  --project runs/eagermot-tuned
```

This example reevaluates the fitting split. To measure generalization, tune
and evaluate on separate sequences with detector checkpoints trained without
the evaluation sequences.

## Sensor fusion example

This example supplies one synthetic camera and matching 2D/3D observations.
Replace them with calibrated observations for each frame in a real sequence.

```python
import torch

from boxmot import create_tracker
from boxmot.structures import Boxes, Boxes3D, CameraModel, Detections, Detections3D
from boxmot.trackers import TrackerSpec

tracker = create_tracker(TrackerSpec("eagermot"))
camera = CameraModel(
    projection=torch.tensor(
        [[100, 0, 100, 0], [0, 100, 50, 0], [0, 0, 1, 0]],
        dtype=torch.float32,
    ),
    image_size=(100, 200),
)
detections = Detections(
    geometry=Boxes(torch.tensor([[78, 39, 122, 62]], dtype=torch.float32)),
    scores=torch.tensor([0.95], dtype=torch.float32),
    class_ids=torch.tensor([0], dtype=torch.int64),
    sample_id="example:0",
)
detections_3d = Detections3D(
    geometry=Boxes3D(torch.tensor([[0, 1, 10, 0, 4, 2, 2]], dtype=torch.float32)),
    scores=torch.tensor([0.9], dtype=torch.float32),
    class_ids=torch.tensor([0], dtype=torch.int64),
    sample_id=detections.sample_id,
)

result = tracker.update(detections, detections_3d=detections_3d, camera=camera)
print(result.image_tracks.track_ids, result.spatial_tracks.track_ids)
print(result.spatial_tracks.geometry.values)
tracker.reset()
```

## Outputs and lifecycle

`update()` returns `MultimodalTracks` with two independent collections:

- `image_tracks` contains confirmed tracks with a current 2D observation.
  Detection indices reference the 2D batch; supplied masks stay row-aligned.
- `spatial_tracks` contains confirmed 3D tracks updated by either sensor in
  the current frame, including objects outside the camera image. Detection
  indices reference the 3D batch and are `-1` for updates supported only by 2D.

Join the collections by `track_ids`, not row position. Spatial boxes are
returned in the current camera coordinate system. To render 3D-only tracks in
an image, project their box corners with the camera matrix. Prediction-only
tracks remain internal and are not emitted.

As in the supplied source, a 3D observation starts a new track. A 2D-only
observation can sustain an existing track but does not create an identity.
Motion prediction advances on every frame, including frames with no 3D
detections; this corrects the source's prediction freeze during such dropouts.
Matching always respects class IDs, including when `per_class=False`.
Confirmation follows the source's initial startup grace period; after startup,
`min_hits` observations are required. `max_age` limits time without support
from either sensor, while `max_age_2d` controls confidence decay when image
support is absent. Spatial scores apply this decay to the latest 3D detector
confidence; even a new 3D-only track receives a reduced score, matching the
source's initial missing-image age. Image scores retain their current detector
confidence.

## Configuration and scope

`boxmot/configs/trackers/eagermot.yaml` uses scalar defaults from the upstream
KITTI car preset. `first_matching_method` selects the 3D association metric;
`distance_threshold` is a positive maximum distance, while
`iou_3d_threshold` applies to 3D IoU matching. `fusion_iou_threshold` controls
2D/3D observation fusion and `iou_threshold` controls second-stage image
association. Set `iou_threshold=1.0` to disable that second stage. Image
association supports `asso_func="iou"` only.

The integration supports one camera per sequence and includes the KITTI input
reader used above. The nuScenes multiple-camera workflow, native C++ execution,
and OBB image geometry are not included. Image association requires at least four box corners in
front of the camera, following the upstream nuScenes visibility rule; the
upstream KITTI projection path handles boxes behind the camera differently.
Spatial output still retains objects without a valid image projection.
Full ego poses transform centers exactly, while projection and 3D IoU use
regenerated yaw-only cuboids rather than separately stored tilted corners.
The pose yaw convention extends the source's XYZ Euler y angle beyond 90°
to preserve full camera turns. These choices can change results relative to
the original implementation.

The upstream MIT license is retained in the implementation
package; the original checkout and dataset SDKs are not runtime dependencies.
The integration has not reproduced the upstream KITTI or nuScenes benchmark
results.

::: boxmot.EagerMot
