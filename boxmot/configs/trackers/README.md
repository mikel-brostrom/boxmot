# Tracker configuration assets

Each `<tracker>.yaml` is the single source of truth for that tracker's runtime
defaults and tuning search space. The filename matches the tracker name used by
the CLI and registry.

## Schema

Each parameter colocates its scalar runtime `default` with its tuning metadata:

```yaml
track_thresh:
  type: uniform
  default: 0.6
  range: [0.4, 0.7]

track_buffer:
  type: qrandint
  default: 30
  range: [10, 61, 10]
```

Normal tracker construction extracts only `default`. The tuning engine reads
`type`, `range`, `options`, `values`, and conditional `activates` metadata from
the same entries.

Every registered Python tracker declares the canonical association selector:

```yaml
asso_func:
  type: choice
  default: iou
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

`presets/` contains named scalar parameter profiles for a particular dataset,
split, or published result. Presets and generated tuning results contain
resolved runtime values only and overlay the defaults in `<tracker>.yaml`.
