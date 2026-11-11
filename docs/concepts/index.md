# Canonical Geometry

Geometry mode is explicit in both the structure type and `TrackerSpec`.

| Mode | Canonical detections | Canonical tracks | Packed rows |
| --- | --- | --- | --- |
| AABB | `Boxes(float32[N,4])` in `xyxy` | `Tracks` with `Boxes` | AABB6 input / AABB8 output |
| OBB | `OrientedBoxes(float32[N,5])` in `cxcywha` | `Tracks` with `OrientedBoxes` | OBB7 input / OBB9 output |

Scores and geometry are `float32`; class IDs, track IDs, and detection indices
are `int64`. Tensors are CPU-contiguous. Constructors reject invalid dtype,
device, rank, contiguity, finiteness, extent, score range, or row alignment
instead of correcting inputs silently.

Those dtypes describe canonical structures. A plain NumPy input uses the
packed row boundary and returns a C-contiguous `float64` AABB8 or OBB9 matrix;
a `Detections` input returns canonical `Tracks`.

A tracker instance has one fixed geometry mode. Its update validates that
`Detections.geometry` matches. Evaluation also validates the dataset and build
geometry before replay.

`Tracks.detection_indices` maps a track row to the current detection row. `-1`
marks a propagated track without a current detection.

The dataclass fields are frozen, but their tensors retain the caller's storage
and remain mutable. Treat them as read-only after construction; clone tensors
when you need independent writable storage. `Detections.select(indices)`
reorders all attached masks, embeddings, and instance IDs together.

## OBB continuity

OBB angles use radians and remain finite but unwrapped. Trackers resolve
equivalent width/height/angle representations relative to the current state,
then retain damped angular velocity to avoid flip artifacts.

All nine box-state trackers support OBB detections. MafHda and EagerMot accept
AABB image geometry only. See the [tracker input matrix](../trackers/index.md#input-support)
before choosing a tracker; geometry is validated during construction and update.

## Calibrated 3D geometry

`Boxes3D` uses `float32[N,7]` `(x, y, z, yaw, length, width, height)` values.
Coordinates are meters in the rectified-camera convention: x right, y down,
z forward. Position is the bottom-face center and yaw is radians about +y.

[EagerMot](../trackers/eagermot.md#required-inputs) combines independent 2D and
3D detections using a `CameraModel`. Its `MultimodalTracks` result has separate
image and spatial collections. Their lengths can differ; join them by track
ID, and interpret each collection's detection indices against its own input
batch.
