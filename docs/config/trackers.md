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
instead use its explicit `frame_width` and `frame_height` constructor settings.

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
