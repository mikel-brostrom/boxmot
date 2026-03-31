---
description: Overview of BoxMOT modes for live tracking, cache generation, evaluation, tuning, and export.
---

# Modes

BoxMOT groups its workflows into five modes. They share a common component model but serve different stages of the tracking pipeline.

<div class="grid cards" markdown>

- **Track**

  ---

  Run detector + ReID + tracker pipelines on webcam, videos, image folders, or streams.

  [Open track mode](track.md)

- **Generate**

  ---

  Precompute detections and embeddings into `runs/dets_n_embs/` for reuse by `eval` and `tune`.

  [Open generate mode](generate.md)

- **Evaluate**

  ---

  Build MOT-format tracker outputs and score them with TrackEval on benchmark bundles.

  [Open evaluate mode](eval.md)

- **Tune**

  ---

  Search tracker YAML parameter spaces with Ray Tune using one or more MOT metrics as objectives.

  [Open tune mode](tune.md)

- **Export**

  ---

  Export ReID backbones to formats such as TorchScript, ONNX, OpenVINO, TensorRT, and TFLite.

  [Open export mode](export.md)

</div>

## Shared Conventions

All runtime modes use the same component vocabulary:

- `--detector` selects the detector weights or detector alias
- `--reid` selects the ReID weights or ReID alias
- `--tracker` selects the tracker and tracker YAML

The main difference is how each mode gets its input:

- `track` consumes a direct `--source`
- `generate` consumes either `--source` or `--benchmark`
- `eval` and `tune` require `--benchmark`
- `export` works only from `--weights`

## Recommended Flow

For interactive tracking, start with [Track](track.md).

For reproducible benchmark work, the usual order is:

1. `generate`
2. `eval`
3. `tune`

`eval` and `tune` can invoke generation automatically, but running `generate` first makes the cache step explicit and easier to inspect.
