# Tracker YAMLs

Algorithm defaults live in immutable Python config classes such as
`BotSortConfig`. Each `boxmot/configs/trackers/<tracker>.yaml` file defines the
corresponding tuning search space and shared component settings. Tuned presets live under
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
- `tune` reads `type`, `range`, `options`, `values`, `activates`, and `geometry`

Search entries can declare `geometry: aabb` or `geometry: obb`. Image tuning
keeps entries for the other geometry fixed at their runtime values, including
any conditional descendants. OccluBoost uses this to exclude OBB thresholds
from AABB searches and AABB-only settings from OBB searches.

An algorithm entry without search metadata is fixed at its config default.
Built-in YAMLs list every algorithm parameter. Settings used only for history,
display, or input frame dimensions have explicit empty mappings (`{}`) where
searching them would not improve tracking. For example, OC-SORT searches
`iou_threshold` over `[0.1, 0.7]`, while `max_obs` stays fixed; StrongSORT searches
`nn_budget` over `[25, 50, 100, 200]` appearance features per track.
For shared component settings, an entry with only `default` is also fixed. If that entry controls an `activates`
block, a true value enables the children's search ranges and a false value
keeps the children fixed. A searchable parent's default does not fix its
children: the selected trial value controls Optuna and HyperOpt branches.
Random search samples variable branches as a flat space.

## Python algorithm configs

Pass algorithm settings through a matching config class. Component settings and
geometry/class selection stay on the constructor:

```python
from boxmot import BotSort, BotSortConfig, ReIDConfig

tracker = BotSort(
    config=BotSortConfig(match_thresh=0.8, track_buffer=40),
    reid=ReIDConfig(device="cpu"),
    per_class=True,
)
```

All eleven trackers expose a `<TrackerName>Config` from `boxmot`. Configs are
frozen dataclasses with validation, `to_dict()`, and `from_mapping()`.
`tracker.config` contains the resolved algorithm settings. See
[tracker classes and autocomplete](../python/index.md#tracker-classes-and-autocomplete)
for the complete list and factory override precedence.

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
trackers reject these policy groups. Filter implementation and dimensions follow
the tracker and box geometry automatically.

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

Every registered Python tracker exposes `asso_func` through the same config
entry. AABB and OBB tracking both support `iou`, `giou`, `diou`, `ciou`,
`hmiou`, and `centroid`. Select one without editing the built-in file by
passing a runtime override:

```python
from boxmot import create_tracker
from boxmot.trackers import TrackerSpec

tracker = create_tracker(
    TrackerSpec(
        name="ocsort",
        geometry="aabb",
        options=(("asso_func", "centroid"),),
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
| `centroid` | Supported | One minus center distance normalized by the frame diagonal. |

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

The native backends expose the same choices for both box layouts; see the
native tracker documentation for C++ integration details.

## Example schema

```yaml title="trackers/bytetrack.yaml"
track_thresh:
  type: uniform
  default: 0.6
  range: [0.4, 0.7]

track_buffer:
  type: qrandint
  default: 30
  range: [10, 61, 10]
```

The tracker name selects its combined built-in file. `track` and `eval` accept
`--tracker-config` to overlay a runtime YAML file or a built-in preset. Explicit
runtime flags override the loaded values. The Python factory accepts a canonical
`TrackerSpec`; place scalar overrides in its sorted `options` tuple. Tuning
writes resolved runtime YAML that can be reused with `--tracker-config`.
