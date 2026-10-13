# Tracker YAMLs

Algorithm defaults live in immutable Python config classes such as
`BotSortConfig`. Each `boxmot/configs/trackers/<tracker>.yaml` file defines the
corresponding tuning search space and shared component settings. Every searchable
parameter includes an explicit `default`; algorithm defaults must match the
value and type in the corresponding Python config. Tuned presets live under
`boxmot/configs/trackers/presets`. Single-profile presets are runtime
overlays and declare their owning tracker with a top-level `tracker` field.
The EagerMOT `eagermot-kitti-mots-val.yaml` preset contains both `car` and
`pedestrian` profiles and loads through `--class-config`; see
[the evaluation example](../trackers/eagermot.md#saved-kitti-mots-validation-preset).

## Role

The filename matches the tracker name used from the CLI:

- `--tracker bytetrack` loads `boxmot/configs/trackers/bytetrack.yaml`
- `--tracker boosttrack` loads `boxmot/configs/trackers/boosttrack.yaml`

## Runtime vs tuning

Runtime defaults and optimization policy are resolved together:

- `track`, `eval`, the Python factory, and direct tracker classes use the same
  typed algorithm defaults
- YAML presets overlay those defaults
- `tune` reads `type`, `range`, `options`, `values`, `activates`, `depends_on`,
  `cases`, `geometry`, and `association`

Search entries can declare `geometry: aabb` or `geometry: obb`. Image tuning
keeps entries for the other geometry fixed at their runtime values, including
any conditional descendants. OccluBoost uses this to exclude OBB thresholds
from AABB searches and AABB-only settings from OBB searches.

Select the backend for the run with `--tracker-backend`; it is not a search
dimension. BoT-SORT, ByteTrack, and OccluBoost share their new algorithm controls
and search ranges across the Python and C++ implementations.

Algorithm fields omitted from a tuning profile retain their typed defaults and
are not searched. Built-in YAMLs omit unused inherited fields and display-only
controls. Empty mappings (`{}`) keep effective setup parameters fixed, such as
camera-motion settings, input frame dimensions, and recovery-history capacity.
For example, OC-SORT searches `iou_threshold` over `[0.1, 0.7]`, while its recovery
history `max_obs` stays fixed; StrongSORT searches
`nn_budget` over `[25, 50, 100, 200]` appearance features per track.
ByteTrack and BotSort search `max_lost_frames` directly over integers 1 through
199 (`range: [1, 200]`). This lifetime counts tracker updates and is independent
of input FPS in both geometries and backends.
For shared component settings, an entry with only `default` is also fixed. If that entry controls an `activates`
block, a true value enables the children's search ranges and a false value
keeps the children fixed. A searchable parent's default does not fix its
children: the selected trial value controls Optuna and HyperOpt branches.
Random search samples variable branches as a flat space.

Use `association` to declare which metrics a search parameter applies to.
An overlap threshold is searched only for the listed overlap metrics, while
the centroid radius is searched only when `asso_func` is `centroid`:

```yaml
iou_threshold:
  association: [iou, giou, diou, ciou, hmiou]
  type: uniform
  default: 0.3
  range: [0.1, 0.9]

centroid_max_distance:
  association: [centroid]
  type: uniform
  default: 0.05
  range: [0.005, 0.1]
```

Parameters outside their applicable metrics are not searched; their typed or
authored runtime values are retained. `association` works together with
`geometry` and `activates`, and applies to random, Optuna, and HyperOpt search,
whether `asso_func` is fixed or searched. Centroid ignores overlap thresholds
at runtime and uses `centroid_max_distance` for distance eligibility. Optional
Kalman motion fusion adds its own uncertainty gate in the first association pass.

Use `depends_on` and `cases` when an applicable parameter needs different numeric
ranges for different metrics, as ByteTrack does for GIoU, DIoU, and CIoU.
Searchable parameters inside `activates` blocks and parameters searched only in
a case also declare their defaults on the parameter entry. Case fragments inherit
that default unless they override it. Cases override the base distribution;
a case containing only `default` fixes that value, including when a runtime
profile supplies a different starting
value. Without a matching case, the base distribution applies, or the runtime
default if no distribution exists.

## Python algorithm configs

Pass algorithm settings through a matching config class. Component settings and
geometry/class selection stay on the constructor:

```python
from boxmot import BotSort, BotSortConfig, ReIDConfig

tracker = BotSort(
    config=BotSortConfig(match_thresh=0.8, max_lost_frames=40),
    reid=ReIDConfig(device="cpu"),
    per_class=True,
)
```

All eleven trackers expose a `<TrackerName>Config` from `boxmot`. Configs are
frozen dataclasses with validation, `to_dict()`, and `from_mapping()`.
`tracker.config` contains the resolved algorithm settings. See
[tracker classes and autocomplete](../python/index.md#tracker-classes-and-autocomplete)
for the complete list and factory override precedence.

Useful algorithm controls include:

| Tracker | Controls |
| --- | --- |
| BoT-SORT | `alpha_fixed_emb` controls appearance smoothing; `fuse_unconfirmed_associate` controls tentative-track score fusion. |
| ByteTrack | `fuse_first_associate` and `fuse_unconfirmed_associate` control score fusion in the two high-confidence association stages. |
| BoT-SORT / ByteTrack | `duplicate_distance_thresh` controls active/lost duplicate suppression using IoU distance, independently of the association metric. Zero disables suppression. |
| BoostTrack / OccluBoost | `alpha_fixed_emb`, `lambda_emb_multiplier`, `appearance_thresh`, and `appearance_iou_fraction` configure first-pass appearance matching; `dlo_soft_alpha`, `dlo_soft_power`, `dlo_threshold_start`, and `dlo_threshold_end` configure DLO boosting; `duo_mahalanobis_thresh` and `duo_iou_thresh` configure DUO boosting. `track_confidence_decay` and `track_confidence_warmup` control track confidence. |
| StrongSORT | `gating_only_position` selects position-only motion gating; `geometric_recovery_max_age` controls how long unmatched tracks can use geometric recovery. |
| HybridSORT | `confidence_floor` sets the lower bound on predicted confidence in AABB mode. |
| MAF-HDA | `kcf_padding`, `kcf_sigma`, `kcf_regularization`, and `kcf_output_sigma_factor` configure its appearance filter. `kcf_update_mode` chooses template reinitialization or interpolation; `kcf_interp_factor` applies to interpolation. `max_area_ratio`, `min_intersection_ratio`, and `recovery_overlap_thresh` configure non-centroid geometric gates. |

Their defaults preserve the original algorithms. For example, this runtime preset
adjusts BoT-SORT's appearance smoothing and duplicate suppression:

```yaml title="botsort-appearance.yaml"
tracker: botsort
alpha_fixed_emb: 0.8
duplicate_distance_thresh: 0.1
fuse_unconfirmed_associate: false
```

These BoT-SORT overrides and the new ByteTrack and OccluBoost algorithm controls
work with both tracker backends. Rebuild native libraries after upgrading to
the expanded configuration ABI.

## Kalman noise

Runtime YAML groups all configurable Kalman settings under `kalman`. Covariance
scales and their unit metadata belong to its `noise` group:

```yaml title="ocsort-noise.yaml"
tracker: ocsort
kalman:
  variable_dt: false
  noise:
    process_position_scale: 1.0
    process_velocity_scale: 1.0
    measurement_noise_scale: 1.0
    initial_position_scale: 1.0
    initial_velocity_scale: 1.0
    reference_dt_s: 0.03333333333333333
    time_unit: frames
```

Pass the file through `--tracker-config`. Partial profiles override individual
fields; omitted fields retain the tracker defaults. Python uses the corresponding
`KalmanConfig` object through the `kalman` constructor argument. Its `noise`
field accepts a `KalmanNoiseConfig`.

`kalman.variable_dt` selects capture timing. Fresh noise settings can use
`time_unit: null` to derive units from it; calibrated profiles save the resolved
unit. Timing and the reference interval remain fixed during tuning.

`kalman.adaptive_kf` selects innovation-based adaptation in BoostTrack and
OccluBoost. `kalman.is_angular` selects EagerMOT's object yaw-velocity state.
OccluBoost's `kalman.ams` group holds `enabled`, `alpha0`, `threshold`,
`buffer_size`, and `shrink_ratio` for its AABB gain-suppression policy. Other
trackers reject these policy groups. `kalman.type` selects a compatible Python
2D filter (`xyah`, `xywh`, `xysr`, `xyhr`, or `xyscr`). Built-in 2D Kalman
profiles search these five named types and declare explicit defaults:

| Tracker | Default `kalman.type` |
| --- | --- |
| ByteTrack, StrongSORT | `xyah` |
| BoT-SORT | `xywh` |
| OC-SORT, DeepOC-SORT | `xysr` |
| HybridSORT | `xyscr` |
| BoostTrack, OccluBoost | `xyhr` |

The Python factory and CLI use these YAML defaults for both AABB and OBB.
Direct constructors without a Kalman configuration use their geometry-dependent
filter defaults; explicitly supplying `kalman.type: null` also requests that
fallback. For OBB, that fallback is `xywh` for ByteTrack and StrongSORT and
`xysr` for HybridSORT. An explicit runtime
`kalman.type`, calibration, covariance vectors, or adaptive KF policy fixes
the filter; C++ tuning keeps its native filter. Use
[`tune --select-kf`](../modes/tune.md#select-the-kf-before-tracker-tuning) to
compare types and noise profiles before tuning tracker thresholds.

For 2D calibration, the five covariance groups also support coordinate vectors:
`process_position_diagonal`, `process_velocity_diagonal`,
`measurement_noise_diagonal`, `initial_position_diagonal`, and
`initial_velocity_diagonal`. Their lengths match the selected filter's
measurement coordinates. Calibration fits these vectors and saves the global
`*_scale` multipliers as `1.0`; explicit `--tune-kf` options refine those
multipliers while preserving the fitted coordinate ratios. EagerMOT's 3D
calibration continues to use five scalar scales.

With per-class tracking, `kalman.noise.by_class` maps detector class IDs to
complete noise profiles. Unlisted classes use the global profile. Calibration
with `--per-class` writes these profiles and records when a class used pooled
estimates because it lacked sufficient evidence.

Calibrated files also carry a `calibration` mapping describing the tracker,
backend, geometry, filter dimensions, and timing basis. Loading validates that
signature before creating a tracker. A single-class profile also binds the
factory's class selection. The YAML is portable without its report sidecar;
dataset and split identify the calibration evidence and allow reuse on held-out
data.

Built-in search schemas use the same group, with a `default` entry for each
field. Search backends address scalar leaves such as
`kalman.noise.measurement_noise_scale`. Noise scales stay fixed unless selected
with `--tune-kf`; see [Kalman tuning](../modes/tune.md#kalman-noise-and-timing).

## EagerMOT class profiles

EagerMOT sensor evaluation loads both class profiles from one YAML through
`--class-config`. The saved `eagermot-kitti-mots-val.yaml` preset contains
`car` and `pedestrian` mappings; tuning writes the same format to `best.yaml`.

## Association function

The 2D box trackers and MAF-HDA expose `asso_func` through the same config
entry. AABB and OBB box tracking both support `iou`, `giou`, `diou`, `ciou`,
`hmiou`, and `centroid`; EagerMOT uses IoU for its image associations.
Select a metric without editing the built-in file by
passing a runtime override:

```python
from boxmot import create_tracker
from boxmot.trackers import TrackerSpec

tracker = create_tracker(
    TrackerSpec(
        name="ocsort",
        geometry="aabb",
        options=(("asso_func", "centroid"), ("centroid_max_distance", 0.05)),
    )
)
```

For live tracking or evaluation from the CLI, use the same selector directly:

```bash
boxmot track --tracker bytetrack --asso-func giou --source video.mp4
boxmot eval --dataset mot17 --build BUILD_ID --tracker ocsort --asso-func centroid
```

For OBB detections, BoxMOT uses these exact definitions:

| `asso_func` | Status | OBB similarity |
| --- | --- | --- |
| `iou` | Supported | IoU from the intersection and union of the two oriented rectangles. |
| `giou` | Supported | Oriented IoU with the unused area in the convex hull of both rectangles as the GIoU penalty. |
| `diou` | Supported | Oriented IoU with squared center distance normalized by the squared diagonal of the rotation-invariant, minimum-area oriented rectangle enclosing both boxes. |
| `ciou` | **Experimental** | The OBB DIoU construction plus a custom aspect-ratio penalty computed from each box's ordered long and short sides. |
| `hmiou` | **Experimental** | Oriented IoU multiplied by the interval IoU of the boxes' projections onto the image's global y-axis. |
| `centroid` | Supported | `1 - center_distance / (centroid_max_distance * frame_diagonal)`, with pairs outside the radius forbidden. |

GIoU, DIoU, and CIoU are transformed with `(score + 1) / 2` before
association. OBB results are defensively clipped to `[0, 1]`; AABB CIoU uses
the same clipping.

OBB `ciou` is an experimental, representation-invariant adaptation of the AABB
aspect-ratio term; it is not a canonical rotated-CIoU definition. OBB `hmiou`
is also experimental. Its global-y projection follows the screen-space height
cue from [Hybrid-SORT](https://arxiv.org/abs/2308.00783), so use it only when
image vertical has a meaningful relationship to object height or depth, such as
upright subjects under a stable camera. It is not rotation-invariant and is
usually unsuitable for arbitrary-heading aerial objects. Retune association
thresholds when changing metrics.

Centroid association needs the frame dimensions for normalization, so trackers
using it need an image until those dimensions have been initialized. SFSORT can
instead use the explicit `frame_width` and `frame_height` fields in `SFSORTConfig`.

`centroid_max_distance` is a positive fraction of the frame diagonal, at most
`1.0`, with a runtime default of `0.05`. On a 1920×1080 frame, `0.05` gives a
maximum center displacement of approximately 110 pixels. This hard distance
limit applies to every matching and recovery pass; appearance similarity cannot
override it. Overlap thresholds do not add another centroid distance cutoff:
settings such as `iou_threshold`, `match_thresh`, and `proximity_thresh` do not
tighten the radius, even when supplied through a runtime profile. The tracker
still uses its confidence, appearance, and lifecycle rules, and competing pairs
are ranked by its association scores. A pair inside the radius is eligible for
geometric matching, rather than guaranteed a match. With optional motion fusion,
the first pass also requires the pair to pass the Kalman uncertainty gate.
IoU and the other metrics ignore `centroid_max_distance`.

ByteTrack exposes `second_match_thresh` and `unconfirmed_match_thresh` alongside
`match_thresh` for other metrics. Their runtime defaults remain `0.5` and `0.7`.
Centroid uses the same radius in the high-confidence, low-confidence, and
unconfirmed passes. A centroid runtime profile needs only the metric and radius:

```yaml title="bytetrack-centroid.yaml"
tracker: bytetrack
asso_func: centroid
centroid_max_distance: 0.03
```

```bash
boxmot track --tracker bytetrack --tracker-config bytetrack-centroid.yaml --source video.mp4
```

The native backends expose the same choices for both box layouts; see the
native tracker documentation for C++ integration details.

## Association fusion

Python trackers expose fusion controls where their association cues support
them. New controls default to disabled; StrongSORT retains its existing motion
blend and ByteTrack/BotSort retain their existing score-fusion defaults.
Built-in tuning profiles search the relevant switches automatically.

| Tracker | Motion fusion | Detection-confidence fusion | ReID–IoU fusion |
| --- | --- | --- | --- |
| ByteTrack | Optional, first pass | Existing first/unconfirmed switches | Unavailable; no ReID |
| BotSort | Optional, first pass | Existing first/unconfirmed switches | Optional, first pass with embeddings |
| OC-SORT | Optional, first pass | Optional, first pass | Unavailable; no ReID |
| DeepOCSORT, HybridSORT, BoostTrack, OccluBoost | Optional, first pass | Optional, first pass | Optional, first pass with embeddings |
| StrongSORT | Existing blend; selectable off/gate/blend | Optional, appearance pass | Optional, appearance pass |
| SFSORT | Unavailable; no Kalman filter | Optional, first pass | Unavailable; no ReID |
| MAF-HDA | Uses its own motion-affinity model | Optional, segment-to-track pass | Unavailable; uses KCF appearance |

EagerMOT retains its separate 3D/sensor association workflow.

`motion_fusion` accepts `off`, `gate`, or `blend`. `gate` rejects pairs outside
the 95% chi-square region without changing other costs. `blend` also computes
`lambda * cost + (1 - lambda) * squared_mahalanobis_distance`. The distance is
not normalized, so changing the blend can require different matching thresholds.
`motion_fusion_lambda` defaults to `0.98` and is searched only for `blend`;
`motion_fusion_only_position` defaults to false and is searched only when motion
fusion is enabled. StrongSORT uses its existing `mc_lambda` and
`gating_only_position` names for these two settings.

Motion fusion uses each track's selected KF, covariance, and predicted state
after CMC. Full geometry uses four measurement coordinates for AABB or five for
OBB; position-only gating uses two. Confidence in an XYSCR state is excluded
from the geometry gate. OBB measurements are aligned to equivalent rectangles
before angular residuals are computed. Motion gates remain hard constraints
through mask conditioning and assignment. They apply to the first pass only;
later recovery passes keep their existing rules.
HybridSORT applies first-pass fusion in AABB mode only when `tcm_first_step`
is enabled; its OBB first pass runs independently of that setting.

`fuse_iou` multiplies appearance similarity by `(1 + IoU) / 2`, using oriented
IoU for OBB. It is searched only when appearance embeddings are enabled.
`fuse_detection_score` multiplies the first-pass similarity by detection
confidence. ByteTrack and BotSort keep their existing score-fusion option names.
MAF-HDA applies equivalent confidence weighting in its log-affinity cost scale;
it does not apply this weighting to tracklet recovery.

For example, this runtime profile enables ByteTrack motion gating without
requiring a ReID model:

```yaml title="bytetrack-motion.yaml"
tracker: bytetrack
motion_fusion: gate
motion_fusion_only_position: false
```

Use it with `--tracker-config bytetrack-motion.yaml` in `track` or `eval`.
For `tune`, runtime values set the baseline and the shipped search profile can
still vary the fusion settings. Start a new tuning run to search new dimensions;
an already running study retains its original search space. Resuming requires
the saved and requested search spaces to match.

The new fusion switches currently require the Python backend. Native C++ tuning
keeps them disabled and native constructors reject explicitly enabled options.
Existing native score fusion remains supported.

## Example schema

```yaml title="trackers/bytetrack.yaml"
track_thresh:
  type: uniform
  default: 0.6
  range: [0.4, 0.7]

max_lost_frames:
  type: randint
  default: 30
  range: [1, 200]
```

The tracker name selects its combined built-in file. `track` and `eval` accept
`--tracker-config` to overlay a runtime YAML file or a built-in preset. Explicit
runtime flags override the loaded values. The Python factory accepts a canonical
`TrackerSpec`; place scalar overrides in its sorted `options` tuple. Tuning
writes resolved runtime YAML that can be reused with `--tracker-config`.
