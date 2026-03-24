# Tracker Configs

This directory contains one YAML file per tracker. The filename must match the
tracker name used from the CLI.

Examples:

- `--tracker bytetrack` loads `boxmot/configs/trackers/bytetrack.yaml`
- `--tracker boosttrack` loads `boxmot/configs/trackers/boosttrack.yaml`

These YAMLs are used in two ways:

- `track` and `eval` read each parameter's `default` value
- `tune` reads the search space from each parameter's `type`, `range`, or
  `options`

Each parameter entry follows this shape:

```yaml
parameter_name:
  type: uniform | randint | qrandint | choice
  default: <runtime default>
  range: [min, max]
  options: [a, b, c]
```

Only the keys required by that parameter type need to be present.

Example:

```yaml
track_thresh:
  type: uniform
  default: 0.6
  range: [0.4, 0.7]

track_buffer:
  type: qrandint
  default: 30
  range: [10, 61, 10]

frame_rate:
  type: choice
  default: 30
  options: [25, 30]
```

Use a tracker config by selecting the tracker:

```bash
boxmot eval --benchmark mot17-ablation --tracker bytetrack
```

There is no separate `--tracker-config` flag at the moment. The tracker name is
the config selector.
