# Oriented Box Tracking

Use OBB tracking when object angle matters, such as aerial imagery or rotated
documents.

```bash
boxmot track \
  --detector yolo11n-obb.pt \
  --geometry obb \
  --tracker ocsort \
  --source video.mp4 \
  --save
```

## Canonical contract

```python
import torch

from boxmot.structures import Detections, OrientedBoxes

detections = Detections(
    geometry=OrientedBoxes(cxcywha_float32_cpu.contiguous()),
    scores=scores_float32_cpu.contiguous(),
    class_ids=class_ids_int64_cpu.contiguous(),
    sample_id="sequence-1:42",
)
tracks = tracker.update(detections)
```

`OrientedBoxes` contains `float32[N,5]` `(cx, cy, w, h, angle)` values. Width
and height are positive, angles are radians, and finite angles remain unwrapped
for temporal continuity. Tracker instances have a fixed AABB or OBB mode.

Before updates, OBB trackers resolve equivalent rectangle representations
relative to the track state and retain damped angular motion. Association uses
oriented geometry in OBB mode.

For a standalone box-only call, `tracker.update(rows)` also accepts an exact
NumPy `N x 7` matrix in `(cx, cy, w, h, angle, confidence, class_id)` order.
The return value is a C-contiguous `float64` `M x 9` matrix in
`(cx, cy, w, h, angle, track_id, confidence, class_id, detection_index)` order.
Use `Detections` for `Tracks`, enrichments, sample metadata, and pipelines.

Review [association configuration](../config/trackers.md#association-function)
and [native support](../native/index.md) before comparing implementations.
