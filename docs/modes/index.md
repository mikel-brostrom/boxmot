# Modes Overview

BoxMOT organizes its main workflows into six modes exposed through one CLI and one high-level Python facade.

| Mode | Use it when | Start here |
| --- | --- | --- |
| `track` | You want detector + tracker output on a live or saved source | [Track](track.md) |
| `generate` | You want reusable detections and embeddings | [Generate](generate.md) |
| `eval` | You want benchmark metrics on MOT-style datasets | [Evaluate](eval.md) |
| `tune` | You want to optimize tracker hyperparameters | [Tune](tune.md) |
| `research` | You want GEPA to propose and score tracker code changes | [Research](research.md) |
| `export` | You want to convert a ReID model to deployment formats | [Export](export.md) |

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

Use [CLI](../usage/cli.md) for the high-level syntax. Each mode page includes its own examples and generated CLI argument table.
