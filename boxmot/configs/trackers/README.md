# Tracker configuration assets

Each tracker owns a typed algorithm config in `boxmot/trackers/<tracker>/config.py`.
That class defines its runtime fields, defaults, and validation. The matching
`<tracker>.yaml` supplies tuning metadata and Kalman/mask-guidance component profiles.
Its filename matches the name used by the CLI and registry.

## Schema

Every searchable parameter declares an explicit `default`, including parameters
inside `activates` blocks. Algorithm defaults must match the value and type in
the tracker's typed config:

```yaml
track_thresh:
  type: uniform
  default: 0.6
  range: [0.4, 0.7]

max_lost_frames:
  type: randint
  default: 30
  range: [1, 200]
```

The loader checks authored algorithm defaults against the typed config and adds
canonical defaults for fixed or omitted fields. The tuning engine reads
`type`, `range`, `options`, `values`, applicability metadata `geometry` and
`association`, and conditional `activates`, `depends_on`, and `cases` metadata.
An empty mapping (`parameter: {}`) marks an explicitly fixed parameter that affects
the tracker's setup, such as camera-motion settings or input frame dimensions.
Omit unused inherited fields and display-only controls from tuning profiles.
Keep observation-history settings when they affect motion estimation or recovery.
Omitted algorithm fields retain their typed defaults and are not searched.
Component entries under `kalman` and `edgetam` retain their profile defaults.
The eight 2D Kalman profiles search five named filter types. Each tracker declares
its default explicitly; for example, DeepOC-SORT uses `xysr`:

```yaml
kalman:
  type:
    type: choice
    default: xysr
    options: [xyah, xywh, xysr, xyhr, xyscr]
```

ByteTrack and StrongSORT default to `xyah`, BoT-SORT to `xywh`, OC-SORT and
DeepOC-SORT to `xysr`, HybridSORT to `xyscr`, and BoostTrack and OccluBoost to
`xyhr`. These YAML defaults apply to Python AABB and OBB runs. Python tuning
searches the named choices unless an authored filter, calibration, covariance vectors, or adaptive
KF policy fixes the model. `--select-kf` selects and fixes a filter before
tracker tuning; C++ runs keep their native filter fixed. Optional covariance
vectors and class-specific noise belong in calibrated or authored runtime profiles.

The backend is fixed for the run with `--tracker-backend`. BoT-SORT, ByteTrack,
and OccluBoost expose the same new algorithm controls in Python and C++, including
score fusion, duplicate suppression, appearance smoothing, and boosting policies.

The 2D box trackers and MAF-HDA declare the canonical association selector:

```yaml
asso_func:
  type: choice
  default: iou
  options: [iou, giou, diou, ciou, hmiou, centroid]
```

The full list applies to both AABB and OBB detections. EagerMOT uses IoU-only
image association. OBB `iou` uses
oriented-rectangle overlap; `giou` uses the joint convex hull; `diou` and
`ciou` normalize center distance with the rotation-invariant minimum-area joint
oriented enclosure; and `centroid` uses center distance divided by
`centroid_max_distance * frame_diagonal`. Pairs beyond that distance are
ineligible in every association stage, including appearance recovery. The
default distance is `0.05` of the frame diagonal and is searched only for centroid
association, over `[0.005, 0.1]`. This is centroid's sole geometric cutoff;
other geometric match thresholds are inactive in centroid mode. OBB `ciou` is an experimental custom long/short-side aspect
adaptation. OBB `hmiou` is an experimental product of oriented IoU and global-y
projection IoU and should be used only where image vertical is a meaningful
height/depth cue.
See the [association function guide](../../../docs/config/trackers.md#association-function)
for the complete formulas and limitations.

Declare applicable metrics with `association`. Overlap thresholds and centroid
distance have separate search dimensions:

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

Only parameters applicable to a trial's `asso_func` are searched. Inactive
parameters retain their typed or authored runtime values. This applies across
all search backends, together with `geometry` and `activates` conditions.
Centroid uses the full radius at runtime and ignores overlap thresholds.

For different numeric ranges within the applicable metrics, use `depends_on`
and `cases`, as in ByteTrack's GIoU, DIoU, and CIoU thresholds. A case overrides
the base distribution; a case containing only `default` fixes the value and
takes precedence over scalar starting values from a runtime profile. Without
a matching case, the base entry applies. Case fragments inherit the parameter's
default unless they explicitly override it. Runtime profiles use scalar values
rather than search metadata.

## Presets

`presets/` contains named parameter profiles for a particular dataset, split,
or published result. Single-profile presets contain scalar runtime values
that overlay the resolved algorithm/component defaults and load with `--tracker-config`.

`presets/eagermot-kitti-mots-val.yaml` stores the tuned KITTI MOTS validation
profiles together under `car` and `pedestrian`. It uses the same class mapping
as EagerMOT tuning's `best.yaml` and loads through evaluation's `--class-config`:

```bash
boxmot eval --dataset ./kitti-mots --tracker eagermot \
  --class-config boxmot/configs/trackers/presets/eagermot-kitti-mots-val.yaml
```
