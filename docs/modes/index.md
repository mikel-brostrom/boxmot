# Modes Overview

BoxMOT organizes its workflows into one CLI command group plus a high-level Python facade for the tracking and benchmark paths.

| Mode | Use it when | Main command | Install notes | Start here |
| --- | --- | --- | --- | --- |
| `track` | You want detector + tracker output on a live or saved source | `boxmot track` | Core install. `yolo` extra preinstalls common YOLO backends. | [Track](track.md) |
| `generate` | You want reusable detections and embeddings | `boxmot generate` | Same as `track`. | [Generate](generate.md) |
| `eval` | You want MOT metrics on a benchmark | `boxmot eval` | Same as `generate`; reuses cached detections and embeddings. | [Evaluate](eval.md) |
| `tune` | You want to optimize tracker hyperparameters | `boxmot tune` | Add the `evolve` extra. | [Tune](tune.md) |
| `research` | You want GEPA to propose and score tracker code changes | `boxmot research` | Add the `research` extra. | [Research](research.md) |
| `train` | You want to train a ReID backbone on a ReID dataset | `boxmot train` | Core install. | [Train ReID](train.md) |
| `eval-reid` | You want `mAP` and CMC metrics for a trained ReID checkpoint | `boxmot eval-reid` | Core install. | [Evaluate ReID](eval-reid.md) |
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
boxmot generate --benchmark mot17 --split ablation
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack
boxmot tune --benchmark mot17 --split ablation --tracker bytetrack
```

These modes share several workflow flags:

- `--benchmark` selects the benchmark YAML config (e.g., `mot17`, `sportsmot`, `mmot`).
- `--split` overrides the dataset split (e.g., `train`, `val`, `test`, `ablation`).
- `--detection-source` switches between the configured detector (`private`) and public MOTChallenge detections (`frcnn`, `sdp`, `dpm`).
- `--postprocessing` applies post-tracking processing steps such as `gsi`, `gbrc`, or `gta`, chained with commas.
- `--tune-kf` estimates Kalman filter noise (Q/R) from ground truth before tracking (`eval` and `tune` only).

See [Evaluation and Postprocessing](../guides/evaluation.md) and [Benchmark Workflows](../guides/benchmarks.md) for details.

### ReID model lifecycle

Use `train`, `eval-reid`, and `export` when you are working on the appearance model itself rather than the full tracking loop.

```bash
boxmot train --model osnet_x0_25 --dataset market1501 --data-dir /data/reid
boxmot eval-reid --weights runs/reid_train/exp/best.pt --dataset market1501 --data-dir /data/reid
boxmot export --weights runs/reid_train/exp/best.pt --include onnx
```

## Shared CLI shape

All BoxMOT modes start from the same command group:

```bash
boxmot MODE [OPTIONS] [DETECTOR] [REID] [TRACKER]
```

See [CLI](../usage/index.md) for the high-level syntax. Each mode page below includes its own examples and a generated CLI argument table.

## Python API path

If you want the same tracking and benchmark modes from Python, start with the [Python API Overview](../python/index.md). The public facade is `boxmot.BoxMOT`.
