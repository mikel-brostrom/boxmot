# 2D/3D Fusion

Combine image detections with calibrated 3D detections to maintain identities
across sensor dropouts. [EagerMot](../trackers/eagermot.md) tracks objects in 3D
and uses 2D observations to sustain existing tracks when 3D detections are missing.

## Variants

| Variant | Inputs on every update | Motion coordinates |
| --- | --- | --- |
| Calibrated 2D/3D fusion | Independent AABB and 3D detections, camera projection, image dimensions | Camera coordinates; motion includes any camera movement |
| Fusion with ego-motion compensation | The same inputs plus an absolute camera-to-world pose | World coordinates internally; outputs return to the current camera coordinates |

For ego-motion compensation, provide the current `camera_to_world` transform
on every update. Otherwise, omit poses throughout the sequence. Call `reset()`
before starting a new sequence or changing pose availability. Full poses
transform box centers, while box orientations remain yaw-only: roll and pitch
are approximated.

## Inputs

The canonical API accepts `Detections`, `Detections3D`, and `CameraModel`:

- **2D detections:** AABB `Boxes` in image pixels, scores, and class IDs.
- **3D detections:** `Boxes3D` in camera coordinates, scores, and class IDs.
  Boxes use bottom-face centers and `(x, y, z, yaw, length, width, height)`;
  distances are meters and yaw is radians. Camera axes are x right, y down,
  and z forward.
- **Calibration:** a `float32[3,4]` projection matrix and `(height, width)`
  image dimensions. An optional `float32[4,4]` pose maps camera coordinates
  into world coordinates.

A NumPy projection requires a supplied frame for its image dimensions. Use
`CameraModel.image_size` when tracking without a frame.

The two detection batches have independent row counts. Synchronize observations
and map both detectors to the same class IDs before tracking. Pass an empty
batch when either sensor has no detections; a 3D observation is required to
create an identity. Advance the tracker on every frame, including frames where
both batches are empty. Motion advances by one fixed frame step per update.

RGB pixels and ReID embeddings are unused. Optional full-frame instance masks
attached to 2D detections are preserved on image tracks; they are not a ReID
variant or a substitute for the required 3D inputs and calibration.

## Run fusion

For online use, call `EagerMot.update()` directly with synchronized sensor
inputs. Start with the runnable [sensor fusion example](../trackers/eagermot.md#sensor-fusion-example).
The image `TrackingPipeline` and `boxmot track` cannot supply 3D detections
and calibration.

For saved observations, prepare a [sensor dataset](../config/datasets.md#bring-your-own-sensor-dataset)
and run:

```bash
boxmot eval --dataset ./my-sensor-dataset --tracker eagermot --split val
boxmot tune --dataset ./my-sensor-dataset --tracker eagermot --split train
```

Default evaluation and tuning use mask HOTA and require detection masks and
mask ground truth. Evaluation also supports [3D tracking metrics](../trackers/eagermot.md#evaluate-3d-tracks)
with `--eval-3d` and 3D annotations; this does not require masks. Tuning continues
to optimize mask HOTA. See [per-class tuning](../trackers/eagermot.md#tune-separate-class-profiles)
for training profiles and evaluating them on held-out sequences.

## Outputs

Canonical 2D `Detections` make `update()` return `MultimodalTracks` containing
separate `image_tracks` and `spatial_tracks`. NumPy 2D detections instead return
two packed arrays with 8 and 11 columns, respectively. Join either pair by
track ID: the collections have independent row counts and detection indices.

Spatial boxes are returned in current camera coordinates in both variants.
Image outputs require a current 2D observation; supplied masks remain aligned.
Prediction-only tracks stay internal. See [output layouts and lifecycle](../trackers/eagermot.md#outputs-and-lifecycle)
for column definitions and emission rules.

The current integration supports one camera per sequence, AABB image geometry,
and the Python backend. For image tracking driven by masks, see the separate
[Masks task](masks.md).
