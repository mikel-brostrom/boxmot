# Tracker YAMLs

Tracker configs live under `boxmot/configs/trackers`.

## Role

The filename matches the tracker name used from the CLI:

- `--tracker bytetrack` loads `boxmot/configs/trackers/bytetrack.yaml`
- `--tracker boosttrack` loads `boxmot/configs/trackers/boosttrack.yaml`

## Runtime vs tuning

Tracker YAMLs are used in two ways:

- `track` and `eval` read each parameter's `default`
- `tune` reads search-space metadata such as `type`, `range`, and `options`

## Example schema

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

There is no separate `--tracker-config` flag. The tracker name is the selector.
