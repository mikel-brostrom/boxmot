# Mode Defaults

Shared tracking-workflow defaults live in `boxmot/configs/runtime.yaml`.

## What it controls

`runtime.yaml` centralizes defaults for:

- shared detector and ReID selections
- runtime options such as `imgsz`, `device`, `batch_size`, and `sequence_workers`
- command-specific defaults for `track`, `materialize`, `eval`, `tune`, and `research`

## Current shape

```yaml
shared:
  detector: yolov8n
  reid: osnet_x0_25_msmt17

runtime:
  tracker: bytetrack
  tracker_backend: python
  sequence_workers: auto
  save: false

research:
  proposal_model: openai/gpt-5.4
  max_metric_calls: 24
  eval_timeout: 900.0

```

The engine CLI resolves tracking defaults through this file. Reusable Python
components instead receive explicit immutable specs. ReID training defaults remain in
`boxmot/reid/training/configs/defaults.yaml`, while export defaults remain in
`boxmot/reid/exporters/defaults.yaml`.

`sequence_workers: auto` selects automatic replay and CMC analysis parallelism
for evaluation and tuning: `min(selected_sequences, max(1, logical_cpus - 2))` workers.
A positive integer supplies an explicit cap, bounded by the selected sequence
count. Automatic CMC uses it when generating missing transforms; compatible
evidence in `<build>/cmc/<analysis-key>/` is validated and reused across runs.
In tuning, CMC setup precedes trials; replay uses this count separately for each trial. See
[sequence parallelism](../modes/eval.md#sequence-parallelism) for visualization
constraints.
