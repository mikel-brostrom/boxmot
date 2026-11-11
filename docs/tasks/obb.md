# Oriented Box Tracking

Use OBB tracking when object angle matters, such as aerial imagery or rotated
documents. Both motion-only and appearance-assisted variants retain oriented
geometry during association and in their outputs.

## Variants

### Motion-only

Use OcSort with an OBB detector to track position, size, and angle without ReID:

```bash
boxmot track \
  --detector yolo11n-obb.pt \
  --geometry obb \
  --tracker ocsort \
  --source video.mp4 \
  --save
```

### Appearance-assisted (ReID)

Use an OBB-capable appearance tracker such as BotSort. In this example,
`models/obb_reid.pt` is a local BoxMOT-compatible ReID checkpoint trained for
the object classes in your footage; supply that file before running:

```bash
boxmot track \
  --detector yolo11n-obb.pt \
  --geometry obb \
  --tracker botsort \
  --reid models/obb_reid.pt \
  --source video.mp4 \
  --save
```

Choose a checkpoint suited to your imagery and classes. The bundled person
ReID weights are trained on people; aerial vehicles and other domains need
their own validation. See [ReID training](../modes/train.md) and
[runtime profiles](../config/reid.md).

## Canonical contract

```python
import torch

from boxmot import OcSort
from boxmot.structures import Detections, OrientedBoxes

tracker = OcSort(is_obb=True)
detections = Detections(
    geometry=OrientedBoxes(torch.tensor([[50, 90, 60, 120, 0.2]], dtype=torch.float32)),
    scores=torch.tensor([0.9], dtype=torch.float32),
    class_ids=torch.tensor([0], dtype=torch.int64),
    sample_id="sequence-1:42",
)
tracks = tracker.update(detections)
```

`OrientedBoxes` contains `float32[N,5]` `(cx, cy, w, h, angle)` values. Width
and height are positive, angles are radians, and finite angles remain unwrapped
for temporal continuity. Tracker instances have a fixed AABB or OBB mode.
Use an OBB detector or explicitly supply OBB measurements: `--geometry obb`
does not convert an AABB detector's predictions to oriented boxes. All nine
box-state trackers support OBB; MafHda and EagerMot do not.

Before updates, OBB trackers resolve equivalent rectangle representations
relative to the track state and retain damped angular motion. Association uses
oriented geometry in OBB mode.

For a standalone box-only call, `tracker.update(rows)` also accepts an exact
NumPy `N x 7` matrix in `(cx, cy, w, h, angle, confidence, class_id)` order.
The return value is a C-contiguous `float64` `M x 9` matrix in
`(cx, cy, w, h, angle, track_id, confidence, class_id, detection_index)` order.
Use `Detections` for `Tracks`, enrichments, sample metadata, and pipelines.

Use the [tracker classification](../trackers/index.md#tasks-and-variants) to
find OBB trackers for each variant. Review
[association configuration](../config/trackers.md#association-function) and
[native support](../native/index.md) before comparing implementations.
