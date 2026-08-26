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
