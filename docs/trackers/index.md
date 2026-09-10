# Tracker Overview

BoxMOT ships multiple tracker backends behind one interface.

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
  fundamental to the method, as in MafHda and EagerMot.

The family is capability metadata; algorithm directories are flat. Each
registered implementation also declares immutable geometry and input
capabilities, which factories and pipelines use for validation. Selecting
`backend="cpp"` chooses a registered native implementation of the same family.

## Input support

The matrix describes the registered **Python tracker** inputs. **Required**
means the input must be available; **Configurable** means it is required when
the corresponding feature is enabled. **Unused** means the tracker does not
use that field for tracking, even if an adapter accepts it on `Detections`.
The last column indicates whether a separate native backend is available.

| Tracker | 2D geometry | ReID embeddings | Instance masks | Frame requirement | 3D detections + camera | Variable timing | Native C++ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [ByteTrack](bytetrack.md) | AABB, OBB | Unused | Unused | Optional | No | Yes | Yes |
| [BotSort](botsort.md) | AABB, OBB | Configurable | Unused | CMC / live ReID | No | Yes | Yes |
| [StrongSort](strongsort.md) | AABB, OBB | Required | Unused | Required | No | Yes | No |
| [OcSort](ocsort.md) | AABB, OBB | Unused | Unused | Optional | No | Yes | Yes |
| [DeepOcSort](deepocsort.md) | AABB, OBB | Configurable | Unused | CMC / live ReID | No | Yes | No |
| [HybridSort](hybridsort.md) | AABB, OBB | Configurable | Unused | CMC / live ReID | No | Yes | No |
| [BoostTrack](boosttrack.md) | AABB, OBB | Configurable | Unused | CMC / live ReID | No | Yes | No |
| [OccluBoost](occluboost.md) | AABB, OBB | Configurable | Unused | CMC / live ReID | No | Yes | Yes |
| [SFSORT](sfsort.md) | AABB, OBB | Unused | Unused | Dimensions | No | No | Yes |
| [MafHda](maf_hda.md) | AABB | Unused | Required | Required | No | No | No |
| [EagerMot](eagermot.md) | AABB | Unused | Optional | Optional | Required | No | No |

- **ReID:** Configurable trackers use embeddings when `use_embeddings=True`.
  Required embeddings can be attached to detections or generated from a frame
  by the tracker; they do not have to be precomputed by the caller.
- **Frames:** All listed trackers accept a canonical `Frame` or a `uint8` BGR
  NumPy image. CMC / live ReID requires pixels when camera-motion compensation
  is enabled or missing embeddings must be generated. StrongSort and MafHda
  require pixels on every update, including empty detection batches.
- **Dimensions:** SFSORT needs image dimensions but does not use image pixels.
  Supply a frame on each update or configure both `frame_width` and
  `frame_height`. Selecting centroid association on other box trackers can
  also make a frame required, even when it is optional in the table.
- **Masks:** MafHda requires full-frame boolean masks aligned with its AABB
  detections, with foreground in every nonempty detection row. Supply an
  empty mask batch with empty detections. EagerMot accepts masks and preserves
  them on matched image-track outputs; its association uses box geometry.
- **3D and calibration:** EagerMot requires `Detections3D` and `CameraModel`
  on every update alongside canonical 2D `Detections`. Either detection batch
  can be explicitly empty. A 3D observation initializes a track; later 2D
  observations can sustain it. Camera image size supplies dimensions when no
  frame is passed.
- **Timing:** Yes means `variable_dt=True` is supported. It remains opt-in and
  then requires a capture timestamp on every update through `Frame.timestamp_s`
  or `timestamp_s=`. See [choosing Kalman timing and
  adaptation](../modes/track.md#choose-kalman-timing-and-adaptation).

### Input representations

All nine box-state trackers accept canonical `Detections` or packed NumPy
matrices: AABB uses `(x1, y1, x2, y2, confidence, class_id)` and OBB uses
`(cx, cy, w, h, angle, confidence, class_id)`. Set the geometry when creating
the tracker. NumPy input returns packed tracks; canonical input returns
`Tracks`.

Use `Detections.embeddings` and `Detections.masks` for precomputed enrichments;
packed rows cannot carry them. MafHda therefore needs canonical `Detections`
with masks. EagerMot also requires canonical structures and returns
`MultimodalTracks`. A `Frame` contains an RGB CHW Torch tensor, while NumPy
images use BGR HWC layout. See the [Python API](../python/index.md#canonical-values)
for dtypes, shapes, and validation.

Every ReID-enabled high-level adapter can consume attached embeddings or
generate missing embeddings from a supplied frame. Direct Python class
construction exposes the same `reid_model`,
`reid_weights`, `device`, `half`, and `reid_preprocess` options across all six
trackers. Attached embeddings bypass inference, and empty batches do not load
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

The five C++ adapters support both AABB and OBB, canonical detections, and
packed NumPy rows. BotSort and OccluBoost accept embeddings or generate them
through their Python adapter. Native trackers do not support masks, 3D sensor
inputs, or variable timing, and their outputs do not preserve masks.

Native frame requirements are fixed at construction: a frame is required on
every update when CMC or centroid association is enabled. SFSORT likewise
requires a frame for dimensions unless both dimensions are configured.
Live ReID additionally needs real pixels when embeddings are missing. See
[native capabilities and requirements](../native/index.md#capabilities-and-requirements)
for backend details.

## How to choose

- Start with `bytetrack` when you want a fast motion-only baseline.
- Use `botsort`, `strongsort`, `deepocsort`, `hybridsort`, `boosttrack`, or `occluboost` when appearance cues matter.
- Use `maf_hda` for AABB detections with nonempty full-frame instance masks and current image frames, combining motion with masked correlation-filter appearance.
- Use `eagermot` through Python when 3D detections and camera calibration are available alongside image detections.
- OBB support is listed in the table above; MafHda and EagerMot accept AABB image geometry only.
- Image trackers expose the same selectable `asso_func`; see
  [tracker configuration](../config/trackers.md#association-function) for the
  supported AABB and OBB choices. EagerMot uses IoU for image association and
  exposes a separate 3D association metric.
- Use `--tracker-backend cpp` for native C++ implementations when the selected tracker has a native backend.

## Config and factory

- Tracker runtime defaults and tuning search spaces share `boxmot/configs/trackers/<tracker>.yaml`; reusable scalar presets remain under `boxmot/configs/trackers/presets`.
- The runtime factory lives in `boxmot/trackers/common/factory.py`; public
  contracts and factories remain available from `boxmot.trackers`, and public
  tracker classes from `boxmot`.
- Native C++ sources and low-level ctypes bindings live under `boxmot/native/`.
  Domain adapters live beside their algorithms at
  `boxmot/trackers/<name>/native.py`.

Use [Native C++ Integration](../native/index.md) when you want to compile and embed a tracker directly in a C++ program.

Use the pages below for each tracker's API reference.
