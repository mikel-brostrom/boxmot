# Modes Overview

The engine exposes one command group. Domain components remain usable on their
own and pipelines contain no source, output, persistence, retry, or display
logic.

| Mode | Purpose | Required input |
| --- | --- | --- |
| `track` | Run a detector and stateful tracker on a source | `--source` plus component selectors |
| `materialize` | Publish keyed detections and optional masks/embeddings | experiment |
| `eval` | Replay a build and calculate MOT metrics | experiment or dataset, plus `--build` |
| `tune` | Optimize tracker parameters against a build | experiment plus `--build` |
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

Perception runs once during materialization. Eval, tune, and research then
stream stable keyed Parquet rows through a fresh live tracker:

```bash
boxmot materialize --experiment mot17-ablation-yolox-lmbn
boxmot eval --experiment mot17-ablation-yolox-lmbn --build BUILD_ID
boxmot tune --experiment mot17-ablation-yolox-lmbn --build BUILD_ID
boxmot research --experiment mot17-ablation-yolox-lmbn --build BUILD_ID
```

These workflows never select a latest build and never run a detector,
segmentor, or encoder implicitly. They verify source, split, taxonomy,
geometry, and component fingerprints before replay.

See [Materialize](materialize.md), [Evaluate](eval.md), [Tune](tune.md), and
[Research](research.md).

## Python composition

Python callers use factories and canonical structures directly. See the
[Python API](../python/index.md) for detector, segmentor, appearance encoder,
tracker, and pipeline examples.
