# Detection Layouts

BoxMOT switches tracking behavior from the detection tensor shape rather than from a separate runtime flag.

## Input and output schemas

| Geometry | Input detections | Output tracks |
| --- | --- | --- |
| AABB | `(N, 6)` = `(x1, y1, x2, y2, conf, cls)` | `(N, 8)` = `(x1, y1, x2, y2, id, conf, cls, det_ind)` |
| OBB | `(N, 7)` = `(cx, cy, w, h, angle, conf, cls)` | `(N, 9)` = `(cx, cy, w, h, angle, id, conf, cls, det_ind)` |

## Shared behavior

- OBB mode is enabled automatically when OBB detections are provided.
- The first valid detection tensor fixes the tracker's layout; later updates must
  keep the same column count.
- `track` and tracker internals use the tensor layout to choose AABB vs OBB behavior.
- Evaluation also depends on dataset `box_type`, so runtime geometry and dataset
  geometry stay aligned.
- `det_ind` lets you map a track back to the detector output row.

## Current OBB tracker support

All currently registered Python trackers support OBB detections:

- `boosttrack`
- `botsort`
- `bytetrack`
- `deepocsort`
- `hybridsort`
- `occluboost`
- `ocsort`
- `sam2mot`
- `sfsort`
- `strongsort`
