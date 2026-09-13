# Add OBB Support

When adding oriented bounding box support to a tracker:

- include `GeometryKind.OBB` in the box tracker's
  `supported_geometry_kinds`; `BoxTracker` derives its static capabilities and
  kernel guard from that declaration
- inherit `BoxTracker` and implement `_track_detections()` so the
  structured wrapper owns the one private NumPy-kernel conversion
- use the shared helpers in
  `boxmot/trackers/common/box/geometry.py` and
  `boxmot/trackers/common/detections/layout.py` instead of hardcoded column
  indices
- keep AABB and OBB parsing paths explicit
- keep motion and association logic OBB-aware in OBB mode, using shared
  geometry helpers from `boxmot/trackers/common/geometry/obb.py` where possible
- reuse motion adapters and Kalman filters under
  `boxmot/trackers/common/motion/`; `cmc/integration.py` preserves the detection
  layout when passing boxes to camera-motion estimators
- emit canonical `Tracks` with `OrientedBoxes` for canonical input and packed
  NumPy rows for packed input; use this exact OBB9 order:
  `(cx, cy, w, h, angle, id, conf, cls, detection_index)`
- if the tracker returns masks, keep them row-aligned with the emitted
  `Tracks`

## Tests to add

- tracker accepts canonical OBB detections
- canonical input returns OBB `Tracks`; packed input returns nine NumPy columns
- OBB association uses oriented geometry
- plotting/history remains stable across frames
- angle updates remain smooth without discontinuous jumps

Relevant shared coverage lives in
`tests/unit/trackers/box/test_box_tracker_contract.py`,
`tests/unit/trackers/common/test_common_obb.py`,
`tests/unit/trackers/common/motion/`, and
`tests/unit/engine/tracking/test_inference.py`. Add tracker-specific cases when
the motion, association, history, or mask behavior is algorithm-specific.
