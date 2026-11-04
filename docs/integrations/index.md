# Integrations

Choose the boundary that matches your application. BoxMOT does not expose a
workflow facade: Python integrations compose explicit components and pipelines,
while the CLI handles complete source-to-sink workflows.

## Bring your own detections

For a standalone box-only tracker, pass exact NumPy AABB `N x 6` or OBB `N x 7`
rows directly; `update()` returns packed `float64` AABB `M x 8` or OBB `M x 9`
rows. Convert external results to canonical structures when composing a
pipeline or attaching masks, embeddings, and explicit sample metadata:

```python
import torch

from boxmot import create_tracker
from boxmot.pipelines import TrackingPipeline
from boxmot.structures import Boxes, Detections, Frame
from boxmot.trackers import TrackerSpec

frame = Frame(
    image=torch.zeros((3, 480, 640), dtype=torch.uint8),
    sample_id="stream-a:15",
    sequence_id="stream-a",
    frame_index=15,
)
detections = Detections(
    geometry=Boxes(torch.tensor([[100, 150, 180, 300]], dtype=torch.float32)),
    scores=torch.tensor([0.9], dtype=torch.float32),
    class_ids=torch.tensor([0], dtype=torch.int64),
    sample_id=frame.sample_id,
)

tracker = create_tracker(TrackerSpec(name="bytetrack", geometry="aabb"))
pipeline = TrackingPipeline(detector=None, tracker=tracker)
result = pipeline.step_detections(frame, detections)
print(result.tracks.track_ids)
```

Replace the synthetic frame and detection with your own values. `Frame`
requires RGB `uint8[3,H,W]`; geometry and scores require `float32`; class IDs
require `int64`; all tensors must be CPU-contiguous. Call this pipeline once
per frame, including empty detections, and reset it before starting another
sequence. The [Python guide](../python/index.md) covers empty batches and
input conversion.

## Add perception components

Use `create_detector`, `create_segmentor`, and `create_reid_encoder` with their
immutable specs, then pass the instances to `PerceptionPipeline` or
`TrackingPipeline`. The pipeline inspects the resolved tracker's requirements
and executes only missing enrichment. Runtime payloads are still validated;
capability declarations are only an early check.

An external appearance encoder is optional when embeddings are needed only by
a ReID-enabled tracker adapter: `TrackingPipeline` forwards the frame and the
tracker extracts missing embeddings internally. Supplying an encoder preserves
the reusable upstream path and is required when `PipelineOutputs` requests
embeddings. ReID-enabled native adapters support the same fallback before
passing features into their model-free C++ tracker libraries.

## HTTP service

The `/v1` endpoints accept AABB or OBB detection rows and optional image pixels;
they do not accept precomputed embeddings, masks, or 3D observations. The engine decodes images and
detection rows into canonical structures, enriches them with shared engine-owned
models when necessary, and advances one decoupled tracker per stream. Requests
for one stream must remain ordered. The GPU service deliberately keeps one
external encoder shared by all stream trackers; its attached embeddings bypass
the private fallback and avoid loading one model per stream. See
[Deployment](../guides/deployment.md).

## Sensor fusion

Use [EagerMOT](../trackers/eagermot.md#sensor-fusion-example) directly for
independent 2D/3D detections and calibration, or its saved-sensor evaluation
workflow. `TrackingPipeline` handles image detections and cannot supply the
camera and 3D observations required by EagerMOT.

## Docker

Run tracking and batch workflows in a CLI container, with datasets, builds,
and models mounted from the host. See [Docker](docker.md) for CPU images,
local GPU builds, and persistent evaluation jobs.

## Native C++

Native tracker libraries use the typed v2 C ABI: geometry, scores, and
embeddings use floating-point buffers; IDs, classes, and detection indices use
`int64`. The library allocates each output object and callers release it with
the matching free call. Cached evaluation streams keyed Parquet data through
this live API; there is no positional NumPy replay format. See
[Native C++](../native/index.md).
