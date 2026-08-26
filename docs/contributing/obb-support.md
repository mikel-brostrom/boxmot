# Add OBB Support

When adding oriented bounding box support to a tracker:

- set `supports_obb = True` on the tracker class
- keep `BaseTracker.setup_decorator`
- use the shared helpers in
  `boxmot/trackers/common/detections/layout.py` instead of hardcoded column
  indices
- keep AABB and OBB parsing paths explicit
- keep motion and association logic OBB-aware in OBB mode, using shared
  geometry helpers from `boxmot/trackers/common/geometry/obb.py` where possible
- preserve both `xywha` and compatibility-friendly `xyxy` accessors when needed
- emit output through the shared formatting contract and preserve the exact OBB
  schema:
  `(cx, cy, w, h, angle, id, conf, cls, det_ind)`
- if the tracker returns masks, keep them row-aligned with the emitted
  `TrackResults`

## Tests to add

- tracker accepts OBB detections
- tracker returns 9-column OBB outputs
- OBB association uses oriented geometry
- plotting/history remains stable across frames
- angle updates remain smooth without discontinuous jumps

Relevant shared coverage lives in
`tests/unit/trackers/bbox/test_bbox_tracker_contract.py`,
`tests/unit/trackers/common/test_common_obb.py`, and
`tests/unit/engine/tracking/test_inference.py`. Add tracker-specific cases when
the motion, association, history, or mask behavior is algorithm-specific.
