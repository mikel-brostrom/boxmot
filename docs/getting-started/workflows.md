# Choose a Mode

BoxMOT has two main usage styles:

- direct source tracking for videos, cameras, image folders, and streams
- config-driven benchmark modes for repeatable experiments

## Mode Guide

| Mode | Use it when | Main command | Guide |
| --- | --- | --- | --- |
| Track | You want detector + tracker output on a live or saved source | `boxmot track` | [Track](../modes/track.md) |
| Generate | You want reusable detections and embeddings | `boxmot generate` | [Generate](../modes/generate.md) |
| Evaluate | You want TrackEval metrics on a benchmark | `boxmot eval` | [Evaluate](../modes/eval.md) |
| Tune | You want to search tracker hyperparameters | `boxmot tune` | [Tune](../modes/tune.md) |
| Research | You want GEPA to propose tracker code changes | `boxmot research` | [Research](../modes/research.md) |
| Export | You want to convert a ReID model to deployment formats | `boxmot export` | [Export](../modes/export.md) |

## Two common paths

### Direct source path

Use this when you already have a source and want outputs quickly:

```bash
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --source video.mp4 --save
```

### Benchmark path

Use this when you want repeatable experiments on MOT-style datasets:

```bash
boxmot generate --benchmark mot17-ablation
boxmot eval --benchmark mot17-ablation --tracker boosttrack
boxmot tune --benchmark mot17-ablation --tracker bytetrack
```

In this path, the benchmark config resolves the dataset, detector profile, and ReID profile automatically.

## Python API path

If you want the same modes in Python, start with [Python API Overview](../python/index.md). The public facade is `Boxmot`, re-exported from `boxmot`.
