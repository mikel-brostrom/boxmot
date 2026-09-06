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

A tracker instance has one fixed geometry mode. Its update validates that
`Detections.geometry` matches. Evaluation also validates the dataset and build
geometry before replay.

`Tracks.detection_indices` maps a track row to the current detection row. `-1`
marks a propagated track without a current detection.

## OBB continuity

OBB angles use radians and remain finite but unwrapped. Trackers resolve
equivalent width/height/angle representations relative to the current state,
then retain damped angular velocity to avoid flip artifacts.

All registered Python trackers support OBB detections. Native factory
validation rejects any unsupported tracker/geometry combination before state is
created.
