# Mode Defaults

Shared runtime defaults live in `boxmot/configs/modes.yaml`.

## What it controls

`modes.yaml` centralizes defaults for:

- shared detector and ReID selections
- runtime options such as `imgsz`, `device`, `batch_size`, and `postprocessing`
- command-specific defaults for `track`, `generate`, `eval`, `tune`, `research`, and `export`

## Current shape

```yaml
shared:
  detector: yolov8n
  reid: osnet_x0_25_msmt17

runtime:
  tracker: bytetrack
  postprocessing: none
  save: false

research:
  proposal_model: openai/gpt-5.4
  max_metric_calls: 24
  eval_timeout: 900.0

export:
  weights: osnet_x0_25_msmt17
  include: [onnx]
```

The CLI and the high-level `BoxMOT` facade both resolve defaults through this file.
