# Modes Overview

BoxMOT organizes its main workflows into six modes exposed through one CLI and one high-level Python facade.

| Mode | Use it when | Main command | Install notes | Start here |
| --- | --- | --- | --- | --- |
| `track` | You want detector + tracker output on a live or saved source | `boxmot track` | Core install. `yolo` extra preinstalls common YOLO backends. | [Track](track.md) |
| `generate` | You want reusable detections and embeddings | `boxmot generate` | Same as `track`. | [Generate](generate.md) |
| `eval` | You want TrackEval metrics on a benchmark | `boxmot eval` | Same as `generate`; reuses cached detections and embeddings. | [Evaluate](eval.md) |
| `tune` | You want to optimize tracker hyperparameters | `boxmot tune` | Add the `evolve` extra. | [Tune](tune.md) |
| `research` | You want GEPA to propose and score tracker code changes | `boxmot research` | Add the `research` extra. | [Research](research.md) |
| `export` | You want to convert a ReID model to deployment formats | `boxmot export` | Add format-specific extras (`onnx`, `openvino`, `tflite`). | [Export](export.md) |

See [Installation](../getting-started/installation.md#mode-specific-extras) for exact extras commands.

## Two workflow families

### Direct-source execution

Use `track` when you already have a webcam, video, image folder, or stream and want annotated output immediately.

```bash
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --source video.mp4 --save
```

### Benchmark-driven execution

Use `generate`, `eval`, `tune`, and `research` when you want repeatable experiments backed by YAML configs in `boxmot/configs`.

```bash
boxmot generate --benchmark mot17-ablation
boxmot eval --benchmark mot17-ablation --tracker boosttrack
boxmot tune --benchmark mot17-ablation --tracker bytetrack
```

## Shared CLI shape

All BoxMOT modes start from the same command group:

```bash
boxmot MODE [OPTIONS] [DETECTOR] [REID] [TRACKER]
```

See [CLI](../usage/index.md) for the high-level syntax. Each mode page below includes its own examples and a generated CLI argument table.

## Python API path

If you want the same modes from Python, start with the [Python API Overview](../python/index.md). The public facade is `boxmot.Boxmot`.
