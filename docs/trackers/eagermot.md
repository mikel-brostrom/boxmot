# EagerMot

[Paper: EagerMOT: 3D Multi-Object Tracking via Sensor Fusion](https://arxiv.org/abs/2104.14682)

EagerMOT associates 3D detections with image detections, tracks objects in 3D,
and uses remaining image observations to sustain tracks when 3D association
fails. BoxMOT integrates the
[authors' Python implementation](https://github.com/aleksandrkim61/EagerMOT)
under `boxmot/trackers/multimodal/eagermot`.

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
  This supports translation and upright yaw rotation; roll and pitch are
  unsupported. Omit the pose only for a stationary camera, and keep pose
  availability consistent throughout a sequence.

The image tracking CLI, `TrackingPipeline`, and cached evaluation/tuning flows
cannot supply these sensor inputs. They reject `eagermot` before running
perception. Use `create_tracker(TrackerSpec("eagermot"))` or `boxmot.EagerMot`.

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

The integration supports one camera per sequence. Dataset loaders, the
nuScenes multiple-camera workflow, native C++ execution, and OBB image geometry
are not included. Image association requires at least four box corners in
front of the camera, following the upstream nuScenes visibility rule; the
upstream KITTI projection path handles boxes behind the camera differently.
Spatial output still retains objects without a valid image projection.

The upstream MIT license is retained in the implementation
package; the original checkout and dataset SDKs are not runtime dependencies.
The integration has not reproduced the upstream KITTI or nuScenes benchmark
results.

::: boxmot.EagerMot
