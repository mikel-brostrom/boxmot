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

The full list applies to both AABB and OBB detections. In OBB mode, overlap
terms use oriented intersections while enclosure and support terms use the
corresponding enclosing bounds required by each metric.

## Presets

`presets/` contains named scalar parameter profiles for a particular dataset,
split, or published result. Presets and generated tuning results contain
resolved runtime values only and overlay the defaults in `<tracker>.yaml`.
