# Add OBB Support

When adding oriented bounding box support to a tracker:

- set `supports_obb = True` on the tracker class
- keep `BaseTracker.setup_decorator`
- use shared detection-layout helpers instead of hardcoded column indices
- keep AABB and OBB parsing paths explicit
- keep motion and association logic OBB-aware in OBB mode
- preserve both `xywha` and compatibility-friendly `xyxy` accessors when needed
- emit the correct OBB output schema:
  `(cx, cy, w, h, angle, id, conf, cls, det_ind)`

## Tests to add

- tracker accepts OBB detections
- tracker returns 9-column OBB outputs
- OBB association uses oriented geometry
- plotting/history remains stable across frames
- angle updates remain smooth without discontinuous jumps
