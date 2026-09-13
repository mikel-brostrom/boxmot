# Python API

BoxMOT v24 separates values, components, composition, and orchestration:

```text
structures -> detector / segmentor / ReID / tracker -> pipelines -> engine
```

The package root exports `__version__`, `create_tracker`, and the lazily loaded
tracker algorithm classes. Import every other public contract from its domain
package.

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

For calibrated 2D/3D sensor fusion, see [EagerMot](../trackers/eagermot.md).
Its extended update interface consumes independent `Detections3D` and a
`CameraModel` and returns `MultimodalTracks` with separate image and spatial
collections. The image-only interfaces below apply to the other trackers.

Tracker specifications contain algorithm configuration; pipelines can keep
appearance models and segmentors as separate reusable components. Every
high-level ReID-enabled tracker adapter supports live appearance extraction: it
can lazily invoke ReID when embeddings are absent and a `Frame` or NumPy image
is supplied.
For a native backend, the adapter sends those features to the model-free C++
tracker library.

```python
from boxmot import create_tracker

tracker = create_tracker(
    "bytetrack",
    backend="python",
    geometry="aabb",
    per_class=False,
    track_thresh=0.55,
)

tracks = tracker.update(detections)
print(tracks.geometry.values, tracks.track_ids)
tracker.reset()
```

Pass algorithm settings as keywords. An explicit `TrackerSpec` is also
accepted; keyword overrides take precedence without changing the original
spec. Configure model selection and inference on the detector and ReID
factories separately.

### Tracker classes and autocomplete

Import a tracker class directly when you want its constructor arguments in your
editor's completion menu:

```python
from boxmot import BotSort

tracker = BotSort(
    track_high_thresh=0.5,
    per_class=True,
    max_age=45,
    class_ids=(0,),
    class_names={0: "person"},
    use_embeddings=False,
    use_cmc=False,
)
```

Invoke completion inside `BotSort(...)` to see argument names and types.
All public tracker classes expose their own arguments and supported shared
settings, including class metadata. `OccluBoost(...)` also suggests its inherited
BoostTrack options, such as `use_cmc`, `cmc_method`, and `lambda_iou`.
Editors that support typed keyword arguments can flag misspelled names and
incorrect value types before you run the code.

Suggestions follow each tracker's supported controls. For example, `ByteTrack`
uses `track_thresh` and `SFSORT` uses `high_th` for detection thresholds; ReID
options appear on trackers that accept embeddings.

### Update inputs and outputs

The public method preserves the input representation:

```text
update(detections: Detections, frame: Frame | np.ndarray | None = None, *, timestamp_s: float | None = None) -> Tracks
update(detections: np.ndarray, frame: Frame | np.ndarray | None = None, *, timestamp_s: float | None = None) -> np.ndarray
```

The optional `frame` accepts a canonical `Frame` or a NumPy image. NumPy images
must have dtype `uint8` and shape `(height, width, 3)` in BGR order; strided
views are made contiguous when needed. A canonical `Frame` carries sample
metadata and stores an RGB CHW tensor in `Frame.image`.

A tracker configured for AABB expects exactly `N x 6`
`(x1, y1, x2, y2, confidence, class_id)` rows; OBB expects exactly `N x 7`
`(cx, cy, w, h, angle, confidence, class_id)` rows. Real numeric arrays are
normalized to canonical dtypes. Geometry is fixed at construction, not
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
may pair NumPy rows with a supplied `Frame` or NumPy image to generate missing
embeddings lazily; the return value remains a NumPy matrix without sample
metadata. A `detection_index == -1` value identifies a propagated track without
a current detection.

Use the [tracker input matrix](../trackers/index.md#input-support) to compare
geometry, embeddings, masks, frames, and sensor inputs across implementations.
Read `tracker.requirements` after construction. When `embeddings`, `masks`, or
`frame` is true, attach/provide that value before calling `update`. For a
ReID-enabled tracker adapter, `requirements.embeddings` means appearance is
required by the algorithm; the direct update boundary can satisfy it from
either attached embeddings or a supplied `Frame` or NumPy image.

### Kalman noise configuration

Python trackers that use Kalman filters accept an immutable configuration with
typed fields, defaults, and editor autocomplete:

```python
from boxmot import KalmanNoiseConfig, OcSort, create_tracker

noise = KalmanNoiseConfig(
    process_position_scale=1.0,
    process_velocity_scale=1.0,
    measurement_noise_scale=1.0,
    initial_position_scale=1.0,
    initial_velocity_scale=1.0,
    reference_dt_s=1 / 30,
)
tracker = OcSort(kalman_noise=noise)
tracker_from_factory = create_tracker("ocsort", kalman_noise=noise)
```

`1.0` preserves each filter's own covariance priors. These values multiply
covariance, not standard deviation. Omitting `kalman_noise` uses the defaults.
The tracker resolves `time_unit=None` from `variable_dt` without mutating the
provided object. A saved explicit unit must match the selected timing mode.

Use `by_class` for complete class-specific settings. Unlisted class IDs use the
global settings. Box trackers require `per_class=True` when these profiles are
present:

```python
noise = KalmanNoiseConfig(
    measurement_noise_scale=1.5,
    by_class={
        0: KalmanNoiseConfig(measurement_noise_scale=0.8),
        1: KalmanNoiseConfig(measurement_noise_scale=2.0),
    },
)
tracker = OcSort(per_class=True, kalman_noise=noise)
```

Each track keeps independent state and covariance. Class profiles share the
same timing units and reference interval. EagerMot accepts the same configuration
for its 3D filter with frame-based timing. SFSORT and MafHda do not accept Kalman
configuration. See [tracker YAMLs](../config/trackers.md#kalman-noise) and
[calibration and tuning](../modes/tune.md#kalman-noise-and-timing) for saved profiles.

### Elapsed time

Trackers default to fixed-step prediction (`variable_dt=False`), preserving the
motion behavior used by established benchmarks and tuning. Timestamps remain
metadata in this mode; supplying them does not enable variable timing.

Variable timing and online noise adaptation are independent settings. See
[choosing Kalman timing and adaptation](../modes/track.md#choose-kalman-timing-and-adaptation)
for scenarios, recommended starting points, and configuration examples.

Python ByteTrack, BotSort, StrongSort, OcSort, DeepOcSort, HybridSort, BoostTrack,
and OccluBoost offer an experimental seconds-based mode. Enable it explicitly
when constructing the tracker:

```python
tracker = create_tracker(
    "bytetrack",
    variable_dt=True,
)
tracks = tracker.update(detections, frame=frame)
```

The tracker then derives elapsed seconds internally from `Frame.timestamp_s`.
For detections-only calls or NumPy images, supply the capture timestamp in
seconds with `tracker.update(detections, timestamp_s=12.04)`. Use one timestamp
source: passing this keyword together with a timestamp-bearing `Frame` raises
an error.

The first timestamp anchors the clock; later updates use the difference from
the preceding timestamp. Timestamps must be finite and increase strictly,
including on updates with no detections. Variable timing requires timestamps
on every update, starting with the first. `tracker.reset()` clears the clock
for a new sequence. The tracker does not infer capture intervals from
wall-clock time, since processing and network delays do not describe object
motion.

The `track`, `eval`, and `tune` commands accept `--variable-dt` to enable the
experimental mode, or `--fixed-dt` to select fixed steps explicitly. Omitting
both flags preserves the tracker YAML setting, which defaults to fixed steps.
Tuning holds this mode constant, records it with the tuned configuration, and
requires the same timing settings when resuming a run. Saved configurations
declare `time_unit: frames` or `time_unit: seconds` under `kalman_noise`; an
override that conflicts with those units is rejected. Untuned defaults use
`time_unit: null` to resolve the units from the chosen mode. Video sources provide media
timestamps, falling back to the
nominal frame rate when timestamps are unavailable or stop advancing and that
rate is known.

The five Python Kalman filters (`xyah`, `xywh`, `xysr`, `xyscr`, and `xyhr`)
retain the optional `dt` in their low-level `predict`, `multi_predict`, and
`predict_state` methods. The tracker supplies this interval internally, updates
the transition matrix, and integrates process covariance over the entire
capture interval. A larger capture gap therefore changes both predicted
motion and uncertainty without using processing time. A filter configured
for seconds requires an explicit measured interval in these low-level methods;
the reference interval never substitutes for a missing capture interval.

The seconds-based mode converts historic per-frame priors using the fixed
reference interval `h = kalman_noise.reference_dt_s`, which defaults to `1/30` second.
This is the basis of the original noise priors, not a measured source frame
interval. The conversion is:

| Reference prior | Seconds-based value before tuning multipliers |
| --- | --- |
| Position process covariance `Q_position` per reference frame | Position noise density `Q_position / h` |
| Velocity process covariance `Q_velocity` per reference frame | Velocity noise density `Q_velocity / h³` |
| Initial velocity covariance `P_velocity` | `P_velocity / h²` |
| Initial position covariance and measurement covariance `R` | Unchanged |

The converted process values define continuous noise densities. Integrating
velocity noise also contributes to position covariance and position–velocity
cross-covariance, so prediction at the reference interval does not reproduce
the legacy discrete process covariance exactly.

Five independent, dimensionless multipliers then calibrate position and
velocity process noise, measurement noise, and initial position and velocity
covariance. They default to `1.0` and are estimated by
[Kalman calibration](../modes/eval.md#kalman-calibration), with timing mode,
units, and reference interval held fixed. New tracker tuning runs preserve the
default or loaded Kalman settings; they have no YAML search ranges.
Unit conversion provides coherent priors; it does not guarantee that existing
benchmark accuracy transfers without calibration and held-out evaluation.

SFSORT, SAM2, and native C++ tracker adapters reject `variable_dt=True` and
retain fixed-step behavior. Native C++ elapsed-time prediction is not
implemented. Track expiration and confirmation settings such as `max_age`,
`track_buffer`, and `min_hits`
remain counts of updates; elapsed-time prediction does not turn them into
durations.

### Live embeddings in ReID-enabled trackers

`BotSort`, `StrongSort`, `DeepOcSort`, `HybridSort`, `BoostTrack`, and
`OccluBoost` share the same Python direct-construction options. The native
BotSort and OccluBoost adapters expose the same live fallback:

- `reid_model` injects a pre-built backend exposing `get_features(boxes, image)`.
- `reid_weights` selects the weights for a lazily built backend; omitting it
  selects the default ReID model.
- `device`, `half`, and `reid_preprocess` configure that lazy backend.

When embeddings are already attached, the tracker uses them without invoking
its backend. A non-empty batch without embeddings requires a `Frame` or NumPy
image, then extracts one embedding per detection. An empty batch bypasses ReID
extraction and does not initialize the model; independent frame requirements
such as CMC still apply. For trackers with a `use_embeddings` option, disabling
it also disables extraction. The resolved `tracker.generates_embeddings`
property reports whether this fallback is active for either backend.

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
from boxmot.trackers import ReIDConfigurableTracker, create_tracker

tracker = create_tracker("botsort", backend="cpp")
spec = ReIDEncoderSpec(backend="onnx", artifact="/models/reid.onnx")

if not isinstance(tracker, ReIDConfigurableTracker) or not tracker.generates_embeddings:
    raise TypeError("This tracker cannot own ReID inference.")
tracker.configure_reid(spec)
```

## Component factories

Create detectors, ReID encoders, and trackers by name. Detector and ReID
factories resolve the model config and weights internally, downloading missing
weights when a download source is configured. Use `allow_download=False` to
require local weights. Detector geometry comes from its config; an explicit
`geometry="aabb"` or `geometry="obb"` must agree with that config.

```python
from boxmot import create_tracker
from boxmot.detectors import create_detector
from boxmot.reid import create_reid_encoder

detector = create_detector("yolo26n", device="cpu")
encoder = create_reid_encoder("osnet-x0-25-msmt17", device="cpu")
tracker = create_tracker("occluboost", per_class=True, use_embeddings=True)
```

Editors that support Python literal completions can suggest model names inside
the first argument's quotes. Detector suggestions combine BoxMOT profiles with
the official box-producing checkpoints from Ultralytics. ReID suggestions combine
runtime profiles with BoxMOT's pretrained checkpoint catalog. Invoke your editor's completion menu
while typing `create_detector("...")`, `create_reid_encoder("...")`, or
`create_tracker("...")`. Detector suggestions include checkpoint selections such
as `"yolox/n"` when a profile has multiple checkpoints.

Suggestions ship with BoxMOT; detector suggestions record the Ultralytics version
used to generate them. Custom paths, config mappings, string variables, and
explicit specs remain accepted. Adding a local model file does not automatically
add an editor suggestion.

ReID checkpoints cover OSNet (including IBN and AIN), ResNet50, MobileNetV2,
MLFN, HACNN, and LMBN. Checkpoint suggestions retain their filenames' underscores
and omit `.pt`; profile IDs such as `"osnet-x0-25-msmt17"` retain their hyphens.
Use either kind directly:

```python
encoder = create_reid_encoder("osnet_x1_0_msmt17", device="cpu")
# Other suggestions: "mobilenetv2_x1_0_market1501", "lmbn_n_market", etc.
```

Backbones without a cataloged ReID checkpoint require your own trained weights
and are not suggested by this inference factory.

Ultralytics models include YOLO detection, instance segmentation, pose, and OBB
variants, YOLO-World, YOLOE, RT-DETR, FastSAM, and YOLO-NAS. For example,
`create_detector("yolov8n-seg")` returns boxes and masks, while
`create_detector("rtdetr-l")` uses the Ultralytics RT-DETR checkpoint. Hugging
Face `rtdetr_v2_*` selectors continue to use the separate RT-DETR backend.
YOLO-NAS requires its upstream `super_gradients` dependency.

Pose models contribute detection boxes; keypoints are not part of `Detections`.
World and YOLOE use the checkpoint's vocabulary. Classification and semantic
segmentation produce different outputs and are rejected by the detector API.
SAM models that require their own prompt workflow are not detector suggestions.

Detectors and ReID encoders also accept a local model path, YAML path, or config
mapping. Set inference options with an `options` mapping:

```python
detector = create_detector(
    "yolo26n",
    device="cuda:0",
    precision="fp16",
    options={"confidence": 0.3, "image_size": [640, 640]},
)
```

Keyword overrides take precedence over the selected config or spec. Options
merge by key, preserving settings you did not override. Explicit specs remain
immutable, and their resolved artifact identity is still validated.

### Independent calls

Using a canonical `Frame` as shown above, call any component independently:

```python
detections = detector.predict([frame])[0]
embeddings = encoder.encode([frame], [detections])[0]
tracks = tracker.update(detections.with_embeddings(embeddings), frame)
```

The encoder accepts boxes from any source, including saved detections. The
tracker consumes attached embeddings without invoking another ReID encoder.
Keep the same tracker through a sequence, update it even on empty detection
frames, and call `tracker.reset()` before the next sequence. Image and
embedding requirements depend on the tracker configuration.

### Explicit specifications

For advanced configuration, pass a frozen specification. These calls use
already resolved local artifact paths and SHA-256 hashes. Segmentor
construction uses this form as well:

```python
from boxmot.detectors import DetectorSpec, create_detector
from boxmot.reid import ReIDEncoderSpec, create_reid_encoder
from boxmot.segmentors import SegmentorSpec, create_segmentor
from boxmot.trackers import TrackerSpec

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

tracker = create_tracker(
    TrackerSpec(name="occluboost", per_class=True),
    max_age=60,
)
```

Resolve real artifact paths and hashes before creating a materialization plan.
Backend `options` are sorted tuples of key/value pairs so specs remain
canonical-JSON serializable.

ReID inference uses the shared [device selectors](../modes/track.md#device-selection):
for example, `device="0"` and `device="cuda:0"` select the first visible CUDA GPU.
Select one device supported by the backend; GPU lists are rejected, and device
selection preserves the process's `CUDA_VISIBLE_DEVICES` setting.

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

### Capture timestamps

Enable `variable_dt` on the tracker and pass timestamp-bearing frames to the
pipeline normally. The tracker derives prediction intervals internally.
Here, `samples` contains consecutive `(Frame, Detections)` pairs with capture
timestamps:

```python
pipeline = TrackingPipeline(
    detector=None,
    tracker=create_tracker("bytetrack", variable_dt=True),
)
for frame, detections in samples:
    result = pipeline.step_detections(frame, detections)
pipeline.reset()
```

Every frame must have a finite timestamp, strictly increasing within the
sequence, including frames without detections. The first frame establishes
the clock; later frames use the difference from the preceding timestamp.
`reset()` clears the clock along with tracker state. Use capture or media
timestamps rather than processing or network arrival times.

The same behavior applies to `step(frame)` when the pipeline owns a detector.
No pipeline timing option or interval argument is needed. With the default
`variable_dt=False`, timestamps are metadata and prediction uses fixed steps.
Variable timing requires one of the [supported Python trackers](#elapsed-time)
and timestamps on every frame.

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
