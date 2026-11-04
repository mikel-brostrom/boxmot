# 2D Box Tracking (AABB)

Use AABB tracking when object orientation is not part of the output. Choose
motion-only association or add ReID appearance features; both variants emit
axis-aligned boxes and track IDs.

## Variants

### Motion-only

Start with ByteTrack for association without an appearance encoder:

```bash
boxmot track \
  --detector yolov8n \
  --geometry aabb \
  --tracker bytetrack \
  --source video.mp4 \
  --save
```

### Appearance-assisted (ReID)

Use a tracker such as BotSort to combine motion and appearance. This example
uses an OSNet checkpoint trained for person ReID:

```bash
boxmot track \
  --detector yolov8n \
  --geometry aabb \
  --tracker botsort \
  --reid osnet_x0_25_msmt17 \
  --classes 0 \
  --source video.mp4 \
  --save
```

For COCO-trained detectors such as `yolov8n`, class `0` selects people. Use a
ReID checkpoint suited to your objects when tracking other classes. In Python,
attach embeddings to `Detections` or let the configured tracker extract them
from the current frame. See [live ReID](../python/index.md#live-embeddings-in-reid-enabled-trackers).

## Canonical contract

```python
import torch

from boxmot import ByteTrack
from boxmot.structures import Boxes, Detections

tracker = ByteTrack()
detections = Detections(
    geometry=Boxes(torch.tensor([[20, 30, 80, 150]], dtype=torch.float32)),
    scores=torch.tensor([0.9], dtype=torch.float32),
    class_ids=torch.tensor([0], dtype=torch.int64),
    sample_id="sequence-1:42",
)
tracks = tracker.update(detections)

print(tracks.geometry.values)       # float32[M,4] xyxy
print(tracks.track_ids)             # int64[M]
print(tracks.detection_indices)     # int64[M], -1 for coasting rows
```

`Boxes` validates finite `float32[N,4]` values with `x2 > x1` and `y2 > y1`.
`Detections.select()` keeps geometry, scores, classes, instance IDs, masks,
and embeddings aligned. `Tracks.select()` keeps geometry, track IDs, scores,
classes, detection indices, and masks aligned; track outputs do not contain
appearance embeddings.

Keep the tracker for the entire sequence. Call `update()` with empty detections
on frames without observations, then `reset()` before a new sequence. A
detected object may need several observations before the tracker emits its ID.

For a standalone box-only call, `tracker.update(rows)` also accepts an exact
NumPy `N x 6` matrix in `(x1, y1, x2, y2, confidence, class_id)` order. The
return value is a C-contiguous `float64` `M x 8` matrix in
`(x1, y1, x2, y2, track_id, confidence, class_id, detection_index)` order.
Use `Detections` for `Tracks`, embeddings, masks, sample metadata, and pipeline
composition.

All registered Python trackers support AABB image geometry; MafHda additionally
requires masks, and EagerMot requires 3D detections and calibration. Start with ByteTrack for
a lightweight baseline, then choose an appearance-assisted tracker when
occlusion-driven identity switches matter more than minimum latency.

The [tracker classification](../trackers/index.md#tasks-and-variants) lists
each algorithm's default variant and whether ReID can be disabled.
