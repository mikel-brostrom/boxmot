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

Defaults are merged in order: `shared`, then `runtime`, then the command's
section. Explicit CLI values override that result. The file is a packaged
default catalog, not a file accepted by `--tracker-config` or `--plan`; those
options load tracker settings and materialization executor settings respectively.

The engine CLI resolves tracking defaults through this file. Detector-based
experiments supply perception settings for materialization, evaluation, and
tuning, so the shared live detector/ReID defaults do not replace an experiment's
components. Reusable Python components instead receive immutable specs or typed
tracker configs. ReID training defaults remain in
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

Materialization batching and retries have their own defaults in
`boxmot.engine.materialization.settings`: detect batches of 16, segment batches
of 8, and embedding batches of 64. Use
[`materialize --plan` or `--set`](../modes/materialize.md#executor-settings)
to change them; the shared runtime `batch_size` is not a materialization override.
