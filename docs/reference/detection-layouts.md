# Detection Layouts

BoxMOT switches tracking behavior from the detection tensor shape rather than from a separate runtime flag.

## Supported layouts

| Geometry | Input detections | Output tracks |
| --- | --- | --- |
| AABB | `(N, 6)` = `(x1, y1, x2, y2, conf, cls)` | `(N, 8)` = `(x1, y1, x2, y2, id, conf, cls, det_ind)` |
| OBB | `(N, 7)` = `(cx, cy, w, h, angle, conf, cls)` | `(N, 9)` = `(cx, cy, w, h, angle, id, conf, cls, det_ind)` |

## Notes

- OBB mode is enabled automatically when OBB detections are provided.
- `det_ind` lets you map a track back to the detector output row.
- OBB-capable trackers in the current repo are `bytetrack`, `botsort`, `ocsort`, and `sfsort`.
