# Tracker Overview

Choose a tracker by its supported [task](../tasks/index.md) and association
variant. One algorithm can support several tasks and variants.

## Tasks and variants

The table lists common task variants and the shipped Python algorithm defaults.
Choose AABB or OBB geometry independently of the association variant; a custom
tracker config or preset can change the enabled cues. Individual tracker pages
describe further association settings.

| Tracker | Main tasks | Default variant | Other task variants |
| --- | --- | --- | --- |
| [ByteTrack](bytetrack.md) | [2D boxes (AABB)](../tasks/aabb.md), [Oriented boxes (OBB)](../tasks/obb.md) | Motion-only | — |
| [OcSort](ocsort.md) | [2D boxes (AABB)](../tasks/aabb.md), [Oriented boxes (OBB)](../tasks/obb.md) | Motion-only | — |
| [SFSORT](sfsort.md) | [2D boxes (AABB)](../tasks/aabb.md), [Oriented boxes (OBB)](../tasks/obb.md) | Motion-only | — |
| [BotSort](botsort.md) | [2D boxes (AABB)](../tasks/aabb.md), [Oriented boxes (OBB)](../tasks/obb.md) | Appearance-assisted (ReID) | Motion-only |
| [StrongSort](strongsort.md) | [2D boxes (AABB)](../tasks/aabb.md), [Oriented boxes (OBB)](../tasks/obb.md) | Appearance-assisted (ReID) | — |
| [DeepOcSort](deepocsort.md) | [2D boxes (AABB)](../tasks/aabb.md), [Oriented boxes (OBB)](../tasks/obb.md) | Appearance-assisted (ReID) | Motion-only |
| [HybridSort](hybridsort.md) | [2D boxes (AABB)](../tasks/aabb.md), [Oriented boxes (OBB)](../tasks/obb.md) | Appearance-assisted (ReID) | Motion-only |
| [BoostTrack](boosttrack.md) | [2D boxes (AABB)](../tasks/aabb.md), [Oriented boxes (OBB)](../tasks/obb.md) | Appearance-assisted (ReID) | Motion-only |
| [OccluBoost](occluboost.md) | [2D boxes (AABB)](../tasks/aabb.md), [Oriented boxes (OBB)](../tasks/obb.md) | Appearance-assisted (ReID) | Motion-only |
| [MafHda](maf_hda.md) | [Masks](../tasks/masks.md) (AABB + instance masks) | Motion and masked appearance | Motion-only association |
| [EdgeTAM](edgetam.md) | [Masks](../tasks/masks.md) from manual boxes or points | Prompted video propagation | — |
| [EagerMot](eagermot.md) | [2D/3D fusion](../tasks/fusion.md) | Calibrated fusion without ego poses | Fusion with ego-motion compensation |

For the five configurable ReID trackers, set `use_embeddings=False` in the
typed config, or `use_embeddings: false` in a tracker YAML, to select the
motion-only variant. StrongSort always requires appearance embeddings.
Motion-only means ReID is disabled; camera-motion compensation and other
enabled features can still require image frames.

MafHda keeps masks in both variants. Select its motion-only variant with
`s2ta_mode="motion"` and `t2ta_mode="motion"`; the default masked appearance
cue uses a correlation filter. EagerMot's fusion variant depends on whether
camera-to-world poses are supplied consistently throughout the sequence.

The box trackers also offer [temporal mask guidance](../tasks/masks.md#use-temporal-masks-in-association)
under its supported Python AABB configuration. Carrying masks for output or
enabling this extra cue does not change their primary task classification.
See [input support](#input-support) for the complete requirements and
[native differences](#native-input-differences) when choosing C++.

## Input support

The matrix describes **Python tracker** inputs. **Required** applies across
supported configurations; **Configurable** depends on enabled features;
**Optional** is consumed when supplied. **Unused** fields are rejected at the
tracking boundary unless a configured appearance encoder consumes them.
Ground truth used for scoring or calibration is separate from these tracking
inputs.

Rows are ordered from fewest to most supported input categories, counting
each non-`Unused` column once, including optional and configurable inputs.

| Tracker | 2D geometry | ReID embeddings | Instance masks | Image input | 3D boxes | Calibration | Ego poses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [ByteTrack](bytetrack.md) | AABB, OBB | Unused | Unused | Optional CMC / centroid dimensions | Unused | Unused | Unused |
| [OcSort](ocsort.md) | AABB, OBB | Unused | Unused | Optional CMC / centroid dimensions | Unused | Unused | Unused |
| [SFSORT](sfsort.md) | AABB, OBB | Unused | Unused | Dimensions or configured size | Unused | Unused | Unused |
| [BotSort](botsort.md) | AABB, OBB | Configurable | Unused | CMC / live ReID / centroid | Unused | Unused | Unused |
| [StrongSort](strongsort.md) | AABB, OBB | Required | Unused | CMC / live ReID / centroid | Unused | Unused | Unused |
| [DeepOcSort](deepocsort.md) | AABB, OBB | Configurable | Unused | CMC / live ReID / centroid | Unused | Unused | Unused |
| [HybridSort](hybridsort.md) | AABB, OBB | Configurable | Unused | CMC / live ReID / centroid | Unused | Unused | Unused |
| [BoostTrack](boosttrack.md) | AABB, OBB | Configurable | Unused | CMC / live ReID / centroid | Unused | Unused | Unused |
| [OccluBoost](occluboost.md) | AABB, OBB | Configurable | Unused | CMC / live ReID / centroid | Unused | Unused | Unused |
| [MafHda](maf_hda.md) | AABB | Unused | Required | Configurable: appearance / centroid | Unused | Unused | Unused |
| [EagerMot](eagermot.md) | AABB | Unused | Optional | Optional metadata; pixels unused | Required | Required | Optional |

- **ReID:** `use_embeddings=True` enables appearance on configurable trackers.
  Supply embeddings or let the tracker generate them from image pixels.
  Cached embeddings remove this pixel requirement but do not disable CMC.
  StrongSort always uses appearance and enables ECC by default; set
  `use_cmc=False` to disable its pixel requirement when embeddings are supplied.
  With appearance disabled on configurable trackers, supplied embeddings are unused.
- **Pixels and dimensions:** CMC and live ReID need actual pixels. Centroid
  association needs dimensions only, including when CMC and ReID are disabled.
  Python ByteTrack and OcSort can enable CMC with `use_cmc=True`; it is off by
  default. SFSORT needs dimensions for its
  regions and optional centroid association; provide a frame or configure both
  `frame_width` and `frame_height`.
- **MafHda:** Its default appearance stages need pixels, including empty
  detection updates. Setting both `s2ta_mode: motion` and `t2ta_mode: motion`
  removes that requirement. Centroid association still needs dimensions.
- **EdgeTAM:** Requires current image pixels on every update and box or
  point prompts for selected objects. Pass empty detections; its model creates
  masks and enclosing AABBs. It has no detection-association or Kalman stage.
  [Prompted evaluation](../modes/eval.md#edgetam-with-ground-truth-prompts)
  can initialize pedestrians from their first eligible ground-truth boxes.
- **Masks:** MafHda requires full-frame boolean masks aligned with its AABB
  detections, with foreground in every nonempty detection row. Supply an
  empty mask batch with empty detections. EagerMot accepts masks and preserves
  them on matched image-track outputs; its association uses box geometry.
  A ReID encoder that requires masks may consume them when generating missing
  embeddings for a box tracker; this does not enable mask association.
  All nine Python box trackers also support optional
  [temporal mask guidance](../tasks/masks.md#use-temporal-masks-in-association).
  That extension needs current pixels and derives its own masks; the table's
  instance-mask column describes supplied detection masks.
- **3D and calibration:** EagerMot requires independent 2D and 3D detections
  and camera calibration on every update. These can be NumPy arrays or
  canonical `Detections`, `Detections3D`, and `CameraModel`. Either detection batch
  can be explicitly empty. A 3D observation initializes a track; later 2D
  observations can sustain it. A NumPy projection requires a frame to supply
  image dimensions; use `CameraModel.image_size` when no frame is supplied.
  Saved-sensor `eval --eval-3d` can omit `detections_2d` and
  supplies empty image detections; mask scoring still needs predicted masks.
- **Ego motion:** EagerMot optionally consumes absolute camera-to-world poses
  through `camera_to_world=` with a NumPy projection, or through
  `CameraModel.camera_to_world`. With poses it tracks motion in world
  coordinates; without them it uses camera coordinates. Keep pose availability
  consistent throughout a sequence. Calibration projection remains required.
- **Timing:** Python trackers except SFSORT and EdgeTAM, plus native ByteTrack, BotSort,
  OcSort, and OccluBoost, support `kalman.variable_dt=True`. This then
  requires a capture timestamp on every update through `Frame.timestamp_s`
  or `timestamp_s=`. Otherwise timestamps are metadata and prediction uses
  fixed frame intervals. SFSORT uses fixed steps on both backends and rejects
  variable timing. Track lifetimes remain counts of updates.
  See [choosing Kalman timing and
  adaptation](../modes/track.md#choose-kalman-timing-and-adaptation).

### Input representations

All nine box-state trackers accept canonical `Detections` or packed NumPy
matrices: AABB uses `(x1, y1, x2, y2, confidence, class_id)` and OBB uses
`(cx, cy, w, h, angle, confidence, class_id)`. Set the geometry when creating
the tracker. NumPy input returns packed tracks; canonical input returns
`Tracks`.

Use `Detections.embeddings` and `Detections.masks` for precomputed enrichments;
packed rows cannot carry them. MafHda therefore needs canonical `Detections`
with masks. EagerMot returns `(image_tracks, spatial_tracks)` arrays for NumPy
2D detections, or `MultimodalTracks` for canonical 2D detections. See its
[sensor fusion example](eagermot.md#sensor-fusion-example) for input and output
layouts. A `Frame` contains an RGB CHW Torch tensor, while NumPy
images use BGR HWC layout. See the [Python API](../python/index.md#canonical-values)
for dtypes, shapes, and validation.

Every ReID-enabled high-level adapter can consume attached embeddings or
generate missing embeddings from a supplied frame. Direct Python class
construction exposes the same `reid=ReIDConfig(...)` or prebuilt
`AppearanceEncoder` input across all six trackers. The configuration groups
model, device, precision, preprocessing, and batching. Attached embeddings bypass inference, and empty batches do not load
the model. Direct callers may also install a complete `ReIDEncoderSpec` with
`tracker.configure_reid(spec)` before the first update of a sequence. Native
BotSort and OccluBoost adapters support that same dual path, then pass features
to their model-free C++ libraries. See
[Live embeddings in ReID-enabled trackers](../python/index.md#live-embeddings-in-reid-enabled-trackers).

Inspect the exact inputs required after resolving configuration:

```python
from boxmot import create_tracker
from boxmot.trackers import TrackerSpec

tracker = create_tracker(TrackerSpec(name="botsort"))
print(tracker.capabilities)  # Supported geometry and input types.
print(tracker.requirements)  # Requirements for this configuration.
```

### Native input differences

ByteTrack, BotSort, OcSort, OccluBoost, and SFSORT have C++ adapters supporting
both AABB and OBB, canonical detections, and
packed NumPy rows. BotSort and OccluBoost accept embeddings or generate them
through their Python adapter. Native trackers do not support masks, 3D sensor
inputs, or mask outputs. ByteTrack, BotSort, OcSort, and OccluBoost support
variable timing; native SFSORT uses fixed steps.

Native frame requirements are fixed at construction: CMC needs pixels;
centroid association without CMC needs dimensions only. SFSORT likewise
requires a frame for dimensions unless both dimensions are configured.
Live ReID additionally needs real pixels when embeddings are missing. See
[native capabilities and requirements](../native/index.md#capabilities-and-requirements)
for backend details.

## How to choose

See [Compare trackers and ReID models](../guides/comparison.md) for selection
guidance and a workflow that evaluates trackers on the same detections.

- Start with `bytetrack` when you want a fast motion-only baseline.
- Use `botsort`, `strongsort`, `deepocsort`, `hybridsort`, `boosttrack`, or `occluboost` when appearance cues matter.
- Use `maf_hda` for AABB detections with full-frame instance masks; its default appearance stages also use current image pixels.
- Use `edgetam` to select objects with boxes or points in the UI and propagate their masks through a video.
- Use `eagermot` through Python when 3D detections and camera calibration are available alongside image detections.
- OBB support is listed in the table above; MafHda, EdgeTAM, and EagerMot accept AABB image geometry only.
- Detection-based image trackers expose the same selectable `asso_func`; see
  [tracker configuration](../config/trackers.md#association-function) for the
  supported AABB and OBB choices. EagerMot uses IoU for image association and
  exposes a separate 3D association metric.
- Use `--tracker-backend cpp` for native C++ implementations when the selected tracker has a native backend.

## Implementation taxonomy

Each algorithm lives under `boxmot/trackers/<name>/`. Shared tracker
infrastructure lives under `boxmot/trackers/common/`, including the base class,
configuration, factory, protocols, registry, and specifications. The shared
`BoxTracker` base and box-geometry adapters live under `common/box/`.

`TrackerFamily` classifies the representation maintained as tracker state:

- `box`: AABB or OBB state. Appearance embeddings, camera motion, masks, or
  frames may still be optional association inputs.
- `mask`: an instance mask is the primary state.
- `multimodal`: multiple primary representations or model memory are
  fundamental to the method, as in MafHda, EdgeTAM, and EagerMot.

The family is capability metadata; algorithm directories are flat. Each
registered implementation also declares immutable geometry and input
capabilities, which factories and pipelines use for validation. Selecting
`backend="cpp"` chooses a registered native implementation of the same family.

## Config and factory

- Each tracker has a frozen Python algorithm config, such as `BotSortConfig`, that owns its runtime defaults. YAML files under `boxmot/configs/trackers/` define tuning ranges and component profiles; reusable runtime presets remain under `boxmot/configs/trackers/presets`.
- The runtime factory lives in `boxmot/trackers/common/factory.py`; public
  contracts and factories remain available from `boxmot.trackers`, and public
  tracker classes from `boxmot`.
- Native C++ sources and low-level ctypes bindings live under `boxmot/native/`.
  Domain adapters live beside their algorithms at
  `boxmot/trackers/<name>/native.py`.

Use [Native C++ Integration](../native/index.md) when you want to compile and embed a tracker directly in a C++ program.

Use the pages below for each tracker's API reference.
