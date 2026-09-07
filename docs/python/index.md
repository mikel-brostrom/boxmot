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

Tracker specifications contain algorithm configuration; pipelines can keep
appearance models and segmentors as separate reusable components. Every
high-level ReID-enabled tracker adapter supports live appearance extraction: it
can lazily invoke ReID when embeddings are absent and a `Frame` is supplied.
For a native backend, the adapter sends those features to the model-free C++
tracker library.

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

The NumPy form is a convenience for tracker calls without masks or precomputed
embeddings. Use `Detections` when providing either enrichment, retaining sample
metadata, or composing a pipeline. Any high-level ReID-enabled tracker adapter
may pair NumPy rows with a supplied `Frame` to generate missing embeddings
lazily; the return value remains a NumPy matrix without sample metadata. A
`detection_index == -1` value identifies a propagated track without a current
detection.

Read `tracker.requirements` after construction. When `embeddings`, `masks`, or
`frame` is true, attach/provide that value before calling `update`. For a
ReID-enabled tracker adapter, `requirements.embeddings` means appearance is
required by the algorithm; the direct update boundary can satisfy it from
either attached embeddings or a `Frame`.

### Live embeddings in ReID-enabled trackers

`BotSort`, `StrongSort`, `DeepOcSort`, `HybridSort`, `BoostTrack`, and
`OccluBoost` share the same Python direct-construction options. The native
BotSort and OccluBoost adapters expose the same live fallback:

- `reid_model` injects a pre-built backend exposing `get_features(boxes, image)`.
- `reid_weights` selects the weights for a lazily built backend; omitting it
  selects the default ReID model.
- `device`, `half`, and `reid_preprocess` configure that lazy backend.

When embeddings are already attached, the tracker uses them without invoking
its backend. A non-empty batch without embeddings requires a `Frame`, then
extracts one embedding per detection. An empty batch bypasses ReID extraction
and does not initialize the model; independent frame requirements such as CMC
still apply. For trackers with a `use_embeddings` option, disabling it also
disables extraction. The resolved `tracker.generates_embeddings` property
reports whether this fallback is active for either backend.

These are direct class-construction options for real-time tracking loops. A
resolved `ReIDEncoderSpec` can instead be installed before the first update of
a sequence with `tracker.configure_reid(spec)`; the tracker keeps the full
backend, artifact hash, preprocessing, and encoder options and still constructs
the encoder lazily. A composed pipeline can also share one `AppearanceEncoder`
and attach its output before the tracker runs. Model settings do not belong in
`TrackerSpec` for either backend. A native adapter owns the optional encoder;
the underlying C++ tracker library accepts only the resulting typed embedding
buffer and never loads a model.

Factory results use the general `Tracker` type because not every tracker can
own ReID. Narrow their configuration surface with the runtime-checkable
optional protocol, then confirm that embedding generation is enabled before
installing a complete encoder specification:

```python
from boxmot.reid import ReIDEncoderSpec
from boxmot.trackers import ReIDConfigurableTracker, TrackerSpec, create_tracker

tracker = create_tracker(TrackerSpec(name="botsort", backend="cpp"))
spec = ReIDEncoderSpec(backend="onnx", artifact="/models/reid.onnx")

if not isinstance(tracker, ReIDConfigurableTracker) or not tracker.generates_embeddings:
    raise TypeError("This tracker cannot own ReID inference.")
tracker.configure_reid(spec)
```

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

In this example, `outputs.embeddings=True` makes embeddings part of the public
`PipelineResult`, so an external `reid` encoder is required unless the detector
already supplies them. If embeddings are needed only inside a ReID-enabled
tracker adapter, omit both `reid` and the embeddings output request; the
pipeline forwards the `Frame` and the tracker extracts them privately. This is
the same for Python implementations and ReID-enabled native adapters.

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
