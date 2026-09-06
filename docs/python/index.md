# Python API

BoxMOT v24 separates values, components, composition, and orchestration:

```text
structures -> detector / segmentor / ReID / tracker -> pipelines -> engine
```

The package root deliberately exports only `__version__`, `create_tracker`, and
the ten lazily loaded tracker classes. Import every other public contract from
its domain package.

## Canonical values

Component boundaries use CPU-contiguous Torch tensors. Constructors validate
their inputs without casting, clipping, filtering, copying, or moving them.

```python
import torch

from boxmot.structures import Boxes, Detections, Frame

frame = Frame(
    image=torch.zeros((3, 480, 640), dtype=torch.uint8),  # RGB CHW
    sample_id="camera-1:000042",
    sequence_id="camera-1",
    frame_index=42,
    timestamp_s=1.4,
    source_uri="camera:1",
)
detections = Detections(
    geometry=Boxes(
        torch.tensor([[100, 120, 260, 420]], dtype=torch.float32)
    ),
    scores=torch.tensor([0.94], dtype=torch.float32),
    class_ids=torch.tensor([0], dtype=torch.int64),
    sample_id=frame.sample_id,
)
```

Use `OrientedBoxes` for `float32[N,5]` `cxcywha` geometry. Angles are radians
and may remain unwrapped across frames. `MaskBatch` stores full-frame
`bool[N,H,W]` masks. `Detections` can carry row-aligned instance IDs, masks,
and `float32[N,D]` embeddings.

`select`, `with_masks`, `with_embeddings`, and `with_instance_ids` return new
immutable values. Use `to_aabb_rows()` or `to_obb_rows()` only when a file or
wire boundary requires the legacy 6/7- or 8/9-column representation.

## Tracker factory

Tracker specifications contain algorithm configuration only. A tracker never
owns or invokes an appearance model or segmentor.

```python
from boxmot import create_tracker
from boxmot.trackers import TrackerSpec

tracker = create_tracker(
    TrackerSpec(
        name="bytetrack",
        backend="python",
        geometry="aabb",
        per_class=False,
        options=(("track_thresh", 0.55),),
    )
)

tracks = tracker.update(detections)
print(tracks.geometry.values, tracks.track_ids)
tracker.reset()
```

The public method preserves the input representation:

```text
update(detections: Detections, frame: Frame | None = None) -> Tracks
update(detections: np.ndarray, frame: Frame | None = None) -> np.ndarray
```

A tracker configured for AABB expects exactly `N x 6`
`(x1, y1, x2, y2, confidence, class_id)` rows; OBB expects exactly `N x 7`
`(cx, cy, w, h, angle, confidence, class_id)` rows. Real numeric arrays are
normalized to canonical dtypes. Geometry is still fixed by `TrackerSpec`, not
inferred from the first matrix.

Class IDs retain only the integer precision present in the packed array. Use
`Detections.class_ids` with `int64` storage when large IDs must remain exact.

Packed NumPy input returns a C-contiguous `float64` matrix. AABB output is
`M x 8` in `(x1, y1, x2, y2, track_id, confidence, class_id,
detection_index)` order. OBB output is `M x 9` with
`(cx, cy, w, h, angle)` replacing the first four coordinates. Empty results
retain the corresponding `(0, 8)` or `(0, 9)` shape. Integer columns must fit
the exact `float64` integer range (`-2**53` through `2**53`); use structured
`Detections` input and `Tracks` output when unrestricted `int64` values or
track-aligned masks are required.

The NumPy form is a convenience for simple box-only tracker calls. Use
`Detections` when providing embeddings, masks, sample metadata, or composing a
pipeline. A supplied `Frame` does not change the NumPy return type or add sample
metadata. A `detection_index == -1` value identifies a propagated track without
a current detection.

Read `tracker.requirements` after construction. When `embeddings`, `masks`, or
`frame` is true, attach/provide that value before calling `update`.

## Component factories

Each factory accepts one frozen, immutable specification:

```python
from boxmot.detectors import DetectorSpec, create_detector
from boxmot.reid import ReIDEncoderSpec, create_reid_encoder
from boxmot.segmentors import SegmentorSpec, create_segmentor

# Values produced by your artifact resolver before component construction.
detector_sha256 = "..."
segmentor_sha256 = "..."
encoder_sha256 = "..."

detector = create_detector(
    DetectorSpec(
        backend="ultralytics",
        artifact="/models/yolo11n.pt",
        artifact_sha256=detector_sha256,
        device="cuda:0",
        precision="fp16",
        geometry_mode="aabb",
    )
)

segmentor = create_segmentor(
    SegmentorSpec(
        backend="sam",
        artifact="/models/sam2_b.pt",
        artifact_sha256=segmentor_sha256,
        device="cuda:0",
        precision="fp16",
    )
)

encoder = create_reid_encoder(
    ReIDEncoderSpec(
        backend="pytorch",
        artifact="/models/osnet_x0_25_msmt17.pt",
        artifact_sha256=encoder_sha256,
        device="cuda:0",
        precision="fp16",
    )
)
```

Resolve real artifact paths and hashes before creating a materialization plan.
Backend `options` are sorted tuples of key/value pairs so specs remain
canonical-JSON serializable.

The encoder derives each crop from the supplied detection geometry: AABBs use
clipped axis-aligned crops and OBBs use the canonical rectified transform.
Built-in ReID encoders ignore detection masks. A custom mask-dependent encoder
can declare `EncoderRequirements(masks=True)` instead.

## Pipelines

`PerceptionPipeline` batches detection and enrichment. `TrackingPipeline` owns
one tracker state and exactly one sequence at a time.

```python
from boxmot.pipelines import PipelineOutputs, TrackingPipeline

pipeline = TrackingPipeline(
    detector=detector,
    segmentor=None,
    reid=encoder,
    tracker=tracker,
    outputs=PipelineOutputs(embeddings=True),
)

result = pipeline.step(frame)
assert result.detections.sample_id == result.tracks.sample_id
pipeline.reset()
```

For service or cached inputs, construct a pipeline with `detector=None` and call
`step_detections(frame, detections)`. Both entry points use the same enrichment
and runtime-validation path. A `PipelineResult` has exactly two fields:
`detections` and `tracks`.

## Materialized datasets

```python
from boxmot.datasets import CachedVisionDataset

dataset = CachedVisionDataset(
    "runs/materializations/BUILD_ID",
    split="ablation",
    load_images=False,
    load_masks=False,
    load_embeddings=True,
)
for sample in dataset:
    print(sample.sample_id, sample.detections.instance_ids)
```

The loader validates requested artifacts at construction and joins every table
by `sample_id` and `instance_id`; Parquet row position has no meaning.

For files, cameras, retry policy, rendering, metrics, persistence, and CLI
workflows, use `boxmot.engine` or the CLI rather than adding those concerns to a
component or pipeline.
