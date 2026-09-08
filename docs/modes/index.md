# Modes Overview

The engine exposes one command group. Domain components remain usable on their
own and pipelines contain no source, output, persistence, retry, or display
logic.

| Mode | Purpose | Required input |
| --- | --- | --- |
| `track` | Run a detector and stateful tracker on a source | `--source` plus component selectors |
| `materialize` | Publish keyed detections and optional masks/embeddings | experiment |
| `time-variant` | Derive a timestamped frame-loss dataset using cached perception | dataset, sequence, and `--build` |
| `eval` | Materialize/replay a build and calculate MOT metrics | experiment (filename or component shorthand), or dataset plus `--build` |
| `tune` | Prepare/replay a build and optimize tracker parameters | experiment (filename or component shorthand), or dataset plus `--build` |
| `research` | Score proposed tracker changes against a build | experiment plus `--build` |
| `train-reid` | Train a reusable appearance backbone | ReID dataset/config |
| `eval-reid` | Evaluate query/gallery retrieval | checkpoint and ReID dataset |
| `compare-reid` | Compare checkpoints across ReID datasets | checkpoints and targets |
| `export` | Export a ReID backbone | checkpoint and formats |
| `build` | Compile native tracker libraries | tracker selector/toolchain |

## Live tracking

```bash
boxmot track \
  --source video.mp4 \
  --detector yolov8n \
  --reid osnet_x0_25_msmt17 \
  --tracker botsort \
  --save
```

The engine creates canonical RGB `Frame` values, runs a `TrackingPipeline`, and
sends each `(frame, result)` pair to configured sinks.

## Reproducible benchmark workflow

Perception runs once during materialization. Eval and tune can perform that
deterministic step automatically; research consumes an explicit build:

```bash
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml
boxmot tune --experiment mot17/ablation-yolox-lmbn.yaml
boxmot research --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID
```

These workflows never select a latest build. They verify source, split,
taxonomy, geometry, and component fingerprints before replay.

See [Materialize](materialize.md), [Evaluate](eval.md), [Tune](tune.md), and
[Research](research.md).

Use [Time-variant dataset](time-variant.md) to compare timing modes under
reproducible frame loss without rerunning perception.

## Python composition

Python callers use factories and canonical structures directly. See the
[Python API](../python/index.md) for detector, segmentor, appearance encoder,
tracker, and pipeline examples.
