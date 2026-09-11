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
these inputs from a [multimodal sequence dataset](../config/datasets.md#multimodal-sequence-datasets).
The `boxmot eval --tracker eagermot` command also accepts `--dataset ./kitti-mots`. Image tracking, `TrackingPipeline`, and
cached perception replay cannot supply the required sensor inputs and reject
`eagermot` before running perception.

## Use your own sensor data

Copy the [sensor dataset template](../config/datasets.md#bring-your-own-sensor-dataset)
and supply your synchronized images, 2D/3D detections, camera projection,
absolute camera-to-world poses, and ground-truth instance masks:

```bash
mkdir -p ./my-sensor-dataset
cp boxmot/configs/datasets/sensor-fusion.yaml ./my-sensor-dataset/dataset.yaml
# Populate the sequence and prediction files described in the template README.
boxmot eval --dataset ./my-sensor-dataset --tracker eagermot --split val
boxmot tune --dataset ./my-sensor-dataset --tracker eagermot \
  --split train --n-trials 50 --seed 0
```

Dataset splits, partition names, and sequence names such as `drive-001` are
authored in `dataset.yaml`, using the same schema as the built-in datasets.
Its `modalities` select encodings and relative paths for images, annotations,
calibration, poses, and independent 2D/3D detections. Split overrides can
select other predictions. `trackrcnn` and `kitti-detections` identify file
encodings, so your own detector can export them. The template documents
the complete field order, mask encoding, coordinates, and empty-frame rules.

This workflow evaluates car and pedestrian segmentation tracking. Image
prediction masks and ground-truth instance PNGs are required by evaluation
and tuning, although masks remain optional for the Python tracker API.
It supports one camera per sequence, with fixed calibration and synchronized
sensor observations. Set dataset `fps` to your recording rate; it controls
timestamps and saved video playback speed. EagerMOT still advances one motion
step per image. Initial class profiles use the KITTI presets, so tune on your training
sequences and evaluate the saved `best.yaml` on held-out recordings.

## Evaluate downloaded KITTI predictions

Run from the repository root with the existing environment and the `mots`
extra installed. This replays saved detector predictions on CPU and evaluates
**segmentation tracking** against KITTI MOTS instance PNGs:

```bash
uv run --no-sync python -m boxmot.engine.cli eval --tracker eagermot \
  --dataset ./kitti-mots \
  --split val \
  --project runs/eagermot
```

The dataset stores each sequence's observations together and declares all
input paths and encodings in `dataset.yaml`:

```text
kitti-mots/
  dataset.yaml
  sequences/training/0002/
    images/000000.png
    ground_truth/000000.png
    calibration.txt
    poses.npy
  predictions/
    trackrcnn/training/0002.txt
    pointgnn-car-t2/training/0002/000000.txt
    pointgnn-car-t3/training/0002/000000.txt
    pointgnn-pedestrian/training/0002/000000.txt
```

KITTI sequence names have four digits and frame names have six digits. The default
MOTS validation split is `0002, 0006, 0007, 0008, 0010, 0013, 0014, 0016, 0018`.
Add `--sequence 0002` for a smaller run; repeat the option to select several
sequences. To evaluate all 21 annotated sequences, use `--split fulltrain`.
Sequences replay in parallel. The [automatic worker count](../modes/eval.md#sequence-parallelism)
uses at most the selected sequence count or the logical CPU count minus two,
with a minimum of one worker. Set `--sequence-workers N` to override it with a
positive integer cap, still bounded by the selected sequence count.

In `dataset.yaml`, the validation split can override `detections_3d` to use
T2 car predictions while training and full training use T3. Edit the modality
paths to change detector inputs. See the [dataset schema and split overrides](../config/datasets.md#split-specific-inputs).
Testing images and calibration are retained, but this partition lacks the
ground truth, ego poses, and 2D predictions needed for evaluation and tuning.

Results are written under `runs/eagermot/val` (then `val2`, and so on):

- `metrics.json`: mask HOTA, DetA, AssA, LocA, CLEAR and Identity metrics,
  including per-sequence results and class/detection averages. Percentage
  metrics use the 0–100 scale, with signed CLEAR scores where applicable.
- `metrics.csv`: combined results for each class and aggregate.
- `mots/SEQUENCE.txt`: official MOTS predictions with real, disjoint masks.
- `videos/SEQUENCE.mp4`: annotated video when `--save` is enabled.
- `run.json`: dataset config, resolved modality formats/paths/options,
  selected sequences, and tracker presets.

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

## Saved KITTI MOTS validation preset

`boxmot/configs/trackers/presets/eagermot-kitti-mots-val.yaml` contains the
best saved car and pedestrian profiles from a 200-trial Optuna run. Use both
profiles together:

```bash
boxmot eval --dataset ./kitti-mots --tracker eagermot \
  --class-config boxmot/configs/trackers/presets/eagermot-kitti-mots-val.yaml
```

The winning trial scored 67.35 class-average mask HOTA on the nine validation
sequences (car: 78.57, pedestrian: 56.13). These are scores on the fitting
split. The YAML comments record the source run, detector inputs and sequences.
The preset uses the same `car`/`pedestrian` format as tuning's `best.yaml`.

## Preview or save a sequence

Use `--show` to preview tracked masks, identities and classes, and `--save` to
write an annotated MP4 for each selected sequence. Add `--show-3d` to overlay
the estimated 3D bounding boxes:

```bash
uv run --no-sync python -m boxmot.engine.cli eval --tracker eagermot \
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
`--show` processes sequences one at a time on the main thread; `--save` alone
allows workers to render and save their assigned sequences in parallel.

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

The [dataset config](../config/datasets.md#multimodal-sequence-datasets) records
sequence locations, classes, splits, and all modality encodings and paths. Images
and ground truth are stored inside the dataset, so it can move independently
of the original downloads. You can pass
`--dataset /path/to/kitti-mots/dataset.yaml` from any working directory;
relative paths remain anchored to `storage.root` beside that YAML.

The dataset's default split is used unless you pass `--split`. Per-split
modality overrides can select different image or spatial prediction files. Every trial
evaluates car and pedestrian profiles together and maximizes their
**class-average mask HOTA**.

The sensor workflow runs one Optuna trial at a time on CPU with
`--search-alg optuna`. Sequences within each trial replay in parallel using the
same automatic worker count as evaluation. `--sequence-workers N` sets a
positive worker cap per trial. An explicit device must be `cpu`.
`--max-concurrent-trials` accepts `0` (default) or `1`, keeping trials serial;
objective selectors must use `HOTA`. Perception and build options and
`--resume-tune` are unavailable for fusion datasets.
`--project` changes the results root from `runs/eagermot-tune`.

The trial count includes the first trial with the starting car and pedestrian
profiles, including any loaded or calibrated settings. Remaining trials independently
sample each class's parameters from the shared EagerMOT YAML search ranges. Distance and 3D IoU
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
uv run --no-sync python -m boxmot.engine.cli eval --tracker eagermot \
  --dataset ./kitti-mots \
  --split val \
  --class-config runs/eagermot-tune/val/best.yaml \
  --project runs/eagermot-tuned
```

This example reevaluates the fitting split. To measure generalization, tune
and evaluate on separate sequences with detector checkpoints trained without
the evaluation sequences.

## Calibrate 3D Kalman noise

Add `--calibrate-kf` to `eval` or `tune` after supplying the optional
[`ground_truth_3d` modality](../config/datasets.md#3d-ground-truth-for-kalman-calibration).
It requires 3D annotations with stable object identities; instance masks alone
cannot provide the required 3D trajectories. The bundled KITTI folder does not
include these labels, so add them and their YAML declaration before running:

```bash
boxmot eval --dataset ./kitti-mots --tracker eagermot \
  --split train --calibrate-kf --project runs/eagermot-calibration
boxmot eval --dataset ./kitti-mots --tracker eagermot \
  --split val --class-config runs/eagermot-calibration/train/kf-tuning/calibrated.yaml
```

Calibration matches 3D detections to annotations by class and 3D IoU, then
transforms both into world coordinates with the supplied ego poses. It fits
the same five covariance multipliers as [2D Kalman calibration](../modes/eval.md#kalman-calibration),
separately for cars and pedestrians. Process and initial covariance distinguish
the seven measured box coordinates from their modeled velocities; measurement
noise has one shared multiplier. The values scale covariance, not standard
deviation. Ego poses remain fixed, and pose uncertainty is not fitted.

Each run saves complete class profiles in `kf-tuning/calibrated.yaml` and
calibration evidence in `kf-tuning/calibration.json`. Reuse the profiles with
`--class-config` in `eval` or `tune`. `tune --calibrate-kf` calibrates once before
Optuna starts; the fitted scales and each class's `is_angular` setting stay
fixed throughout the search and are included in `best.yaml`. Prediction still
advances one frame per image; variable-time prediction is unsupported.
Loading profiles with `--class-config` also keeps their `is_angular` choices
fixed. Tuning without either flag can search the angular-motion choice.

Add `--cache-inputs` to reuse all declared sensor observations and annotations
across trials and later runs. Calibration consumes the same cached 3D inputs.
The cache stores unfiltered detections and packed masks; each trial still applies
its own thresholds and starts fresh trackers. Worker processes remain available
throughout the study. See [input caching](../modes/eval.md#cache-replay-inputs-for-repeated-runs)
for storage and invalidation behavior.

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
