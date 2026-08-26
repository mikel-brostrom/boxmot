# Mode Defaults

Shared tracking-workflow defaults live in `boxmot/configs/runtime.yaml`.

## What it controls

`runtime.yaml` centralizes defaults for:

- shared detector and ReID selections
- runtime options such as `imgsz`, `device`, `batch_size`, and `postprocessing`
- command-specific defaults for `track`, `generate`, `eval`, `tune`, and `research`

## Current shape

```yaml
shared:
  detector: yolov8n
  reid: osnet_x0_25_msmt17

runtime:
  tracker: bytetrack
  tracker_backend: python
  postprocessing: none
  save: false

research:
  proposal_model: openai/gpt-5.4
  max_metric_calls: 24
  eval_timeout: 900.0

```

The CLI and high-level `BoxMOT` facade both resolve tracking defaults through
this file. ReID training defaults remain in
`boxmot/reid/training/configs/defaults.yaml`, while export defaults remain in
`boxmot/reid/exporters/defaults.yaml`.
