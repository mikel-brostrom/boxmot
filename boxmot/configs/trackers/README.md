# Tracker configuration assets

Each tracker owns a typed algorithm config in `boxmot/trackers/<tracker>/config.py`.
That class defines its runtime fields, defaults, and validation. The matching
`<tracker>.yaml` supplies tuning metadata and Kalman/mask-guidance component profiles.
Its filename matches the name used by the CLI and registry.

## Schema

Algorithm entries contain tuning metadata. Runtime defaults come from the typed config:

```yaml
track_thresh:
  type: uniform
  range: [0.4, 0.7]

track_buffer:
  type: qrandint
  range: [10, 61, 10]
```

The loader adds canonical defaults to the resolved schema. The tuning engine reads
`type`, `range`, `options`, `values`, and conditional `activates` metadata from
these entries. An empty mapping (`parameter: {}`) marks a fixed parameter.
Built-in profiles explicitly list every algorithm field, including fixed fields,
so missing search metadata is visible during review. Keep observation history,
display-only settings, and input frame dimensions fixed when they do not affect
association or track lifecycle. Omitted fields in custom profiles still use their
typed defaults.
Component entries under `kalman` and `edgetam` retain their profile defaults.

Every registered Python tracker declares the canonical association selector:

```yaml
asso_func:
  type: choice
  options: [iou, giou, diou, ciou, hmiou, centroid]
```

The full list applies to both AABB and OBB detections. OBB `iou` uses
oriented-rectangle overlap; `giou` uses the joint convex hull; `diou` and
`ciou` normalize center distance with the rotation-invariant minimum-area joint
oriented enclosure; and `centroid` uses frame-diagonal-normalized center
distance. OBB `ciou` is an experimental custom long/short-side aspect
adaptation. OBB `hmiou` is an experimental product of oriented IoU and global-y
projection IoU and should be used only where image vertical is a meaningful
height/depth cue.
See the [association function guide](../../../docs/config/trackers.md#association-function)
for the complete formulas and limitations.

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
