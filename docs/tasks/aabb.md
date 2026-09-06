# Axis-aligned Box Tracking

Use AABB tracking when object orientation is not part of the output.

```bash
boxmot track \
  --detector yolov8n \
  --geometry aabb \
  --tracker bytetrack \
  --source video.mp4 \
  --save
```

## Canonical contract

```python
import torch

from boxmot.structures import Boxes, Detections

detections = Detections(
    geometry=Boxes(xyxy_float32_cpu.contiguous()),
    scores=scores_float32_cpu.contiguous(),
    class_ids=class_ids_int64_cpu.contiguous(),
    sample_id="sequence-1:42",
)
tracks = tracker.update(detections)

print(tracks.geometry.values)       # float32[M,4] xyxy
print(tracks.track_ids)             # int64[M]
print(tracks.detection_indices)     # int64[M], -1 for coasting rows
```

`Boxes` validates finite `float32[N,4]` values with `x2 > x1` and `y2 > y1`.
`Detections` and `Tracks` keep geometry, scores, classes, IDs, masks, and
embeddings aligned during immutable selection.

For a standalone box-only call, `tracker.update(rows)` also accepts an exact
NumPy `N x 6` matrix in `(x1, y1, x2, y2, confidence, class_id)` order. The
return value is still `Tracks`; use `to_aabb_rows()` only when an external file
or wire format needs an AABB8 matrix. Use `Detections` for embeddings, masks,
explicit sample identity without a `Frame`, and pipeline composition.

All registered Python trackers support AABB geometry. Start with ByteTrack for
a lightweight baseline, then choose an appearance-assisted tracker when
occlusion-driven identity switches matter more than minimum latency.
