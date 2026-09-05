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

The AABB6 detection and AABB8 track matrices exist only as explicit
`to_aabb_rows()` serializers for file or wire boundaries.

All registered Python trackers support AABB geometry. Start with ByteTrack for
a lightweight baseline, then choose an appearance-assisted tracker when
occlusion-driven identity switches matter more than minimum latency.
