# Tracker YAMLs

Each `boxmot/configs/trackers/<tracker>.yaml` file contains both runtime
defaults and the corresponding tuning search space. Tuned presets remain plain
scalar overlays under `boxmot/configs/trackers/presets` and declare their owning
tracker with a top-level `tracker` field.

## Role

The filename matches the tracker name used from the CLI:

- `--tracker bytetrack` loads `boxmot/configs/trackers/bytetrack.yaml`
- `--tracker boosttrack` loads `boxmot/configs/trackers/boosttrack.yaml`

## Runtime vs tuning

Runtime values and optimization policy share a file but remain separate in
code:

- `track` and `eval` extract each parameter's scalar `default`
- a preset overlays those defaults
- `tune` reads `type`, `range`, `options`, `values`, and `activates`

## Association function

Every registered Python tracker exposes `asso_func` through the same config
entry. AABB and OBB tracking both support `iou`, `giou`, `diou`, `ciou`,
`hmiou`, and `centroid`. Select one without editing the built-in file by
passing a runtime override:

```python
from boxmot import BoxMOT
from boxmot.trackers.registry import create_tracker

model = BoxMOT(
    tracker="bytetrack",
    tracker_kwargs={"asso_func": "giou"},
)

tracker = create_tracker(
    "ocsort",
    tracker_kwargs={"asso_func": "centroid"},
)
```

For live tracking or evaluation from the CLI, use the same selector directly:

```bash
boxmot track --tracker bytetrack --asso-func giou --source video.mp4
boxmot eval --dataset mot17 --tracker ocsort --asso-func centroid
```

In OBB mode, overlap terms use oriented intersections while enclosure and
support terms use the corresponding enclosing bounds required by each metric.
Centroid association uses the frame diagonal for normalization, so trackers
using it need an image until the frame dimensions have been initialized. SFSORT
can instead use its explicit `frame_width` and `frame_height` constructor
settings.

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

There is no separate `--tracker-config` CLI flag. The tracker name selects its
combined built-in file. The low-level `create_tracker(...)` factory accepts a
scalar YAML path or built-in preset name through `tracker_config`; mapping
overrides use `tracker_kwargs` or `evolve_param_dict`. Tuning writes fully
resolved scalar YAML that can be passed back through `tracker_config`.
