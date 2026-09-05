# Integrations

Choose the boundary that matches your application. BoxMOT does not expose a
workflow facade: Python integrations compose explicit components and pipelines,
while the CLI handles complete source-to-sink workflows.

## Bring your own detections

Convert external results once at the boundary, then keep canonical structures
inside the pipeline:

```python
import torch

from boxmot import create_tracker
from boxmot.pipelines import TrackingPipeline
from boxmot.structures import Boxes, Detections, Frame
from boxmot.trackers import TrackerSpec

frame = Frame(
    image=rgb_chw_uint8_cpu.contiguous(),
    sample_id="stream-a:15",
    sequence_id="stream-a",
    frame_index=15,
)
detections = Detections(
    geometry=Boxes(xyxy_float32_cpu.contiguous()),
    scores=scores_float32_cpu.contiguous(),
    class_ids=class_ids_int64_cpu.contiguous(),
    sample_id=frame.sample_id,
)

tracker = create_tracker(TrackerSpec(name="bytetrack", geometry="aabb"))
pipeline = TrackingPipeline(detector=None, tracker=tracker)
result = pipeline.step_detections(frame, detections)
```

The example variables deliberately mark the boundary conversion: `Frame`
requires RGB `uint8[3,H,W]`; geometry and scores require `float32`; class IDs
require `int64`; all tensors must be CPU-contiguous.

## Add perception components

Use `create_detector`, `create_segmentor`, and `create_reid_encoder` with their
immutable specs, then pass the instances to `PerceptionPipeline` or
`TrackingPipeline`. The pipeline inspects the resolved tracker's requirements
and executes only missing enrichment. Runtime payloads are still validated;
capability declarations are only an early check.

## HTTP service

The `/v1` wire format remains compatible. The engine decodes images and
detection rows into canonical structures, enriches them with shared engine-owned
models when necessary, and advances one decoupled tracker per stream. Requests
for one stream must remain ordered. See [Deployment](../guides/deployment.md).

## Native C++

Native trackers use the typed v2 C ABI: geometry, scores, and embeddings use
floating-point buffers; IDs, classes, and detection indices use `int64`. The
library allocates each output object and callers release it with the matching
free call. Cached evaluation streams keyed Parquet data through this live API;
there is no positional NumPy replay format. See [Native C++](../native/index.md).
