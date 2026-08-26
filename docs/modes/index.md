# Modes Overview

BoxMOT organizes its workflows into one CLI command group plus a high-level
Python facade for tracking, benchmark, and ReID paths.

| Mode | Use it when | Main command | Install notes | Start here |
| --- | --- | --- | --- | --- |
| `track` | You want detector + tracker output on a live or saved source | `boxmot track` | Core install. `yolo` extra preinstalls common YOLO backends. | [Track](track.md) |
| `generate` | You want reusable detections and embeddings | `boxmot generate` | Same as `track`. | [Generate](generate.md) |
| `eval` | You want MOT metrics on a benchmark | `boxmot eval` | Same as `generate`; reuses cached detections and embeddings. | [Evaluate](eval.md) |
| `tune` | You want to optimize tracker hyperparameters | `boxmot tune` | Add the `evolve` extra. | [Tune](tune.md) |
| `research` | You want GEPA to propose and score tracker code changes | `boxmot research` | Add the `research` extra. | [Research](research.md) |
| `train-reid` | You want to train a ReID backbone on a ReID dataset | `boxmot train-reid` | Core install. | [Train ReID](train.md) |
| `eval-reid` | You want `mAP` and CMC metrics for a trained ReID checkpoint | `boxmot eval-reid` | Core install. | [Evaluate ReID](eval-reid.md) |
| `compare-reid` | You want a cross-domain matrix for several ReID checkpoints and datasets | `boxmot compare-reid` | Core install. | [Compare ReID](compare-reid.md) |
| `export` | You want to convert a ReID model to deployment formats | `boxmot export` | Add the relevant format extra (`onnx`, `coreml`, `openvino`, or `tflite`); TensorRT needs CUDA. | [Export](export.md) |
| `build` | You want to compile native tracker libraries | `boxmot build` | Requires a supported C++ toolchain. | [Native C++ Integration](../native/index.md) |

See [Installation](../getting-started/installation.md#mode-specific-extras) for exact extras commands.

## Two workflow families

### Direct-source execution

Use `track` when you already have a webcam, video, image folder, or stream and want annotated output immediately.

```bash
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --source video.mp4 --save
```

### Experiment-driven execution

Use `generate`, `eval`, `tune`, and `research` when you want repeatable experiments backed by YAML configs in `boxmot/configs`.

```bash
boxmot generate --experiment mot17-ablation-yolox-lmbn
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack
boxmot tune --experiment mot17-ablation-yolox-lmbn --tracker bytetrack
```

The benchmark modes share several workflow flags, with mode-specific scope:

- `--experiment` selects an experiment ID or explicit experiment YAML (for
  example, `mot17-ablation-yolox-lmbn`). It is required by `tune` and
  `research`, and is one of the input choices for `generate` and `eval`.
- `--split` overrides the dataset split for `generate`, `eval`, and `tune`.
  Research uses the split resolved by its experiment.
- `--detection-source` selects `private` model detections or `public` sequence
  detections for `generate`, `eval`, and `tune`; choose a source-specific
  experiment when the exact public producer matters.
- `--postprocessing` applies steps such as `gsi`, `gbrc`, or `gta` during
  `eval` and `tune`; comma-separated steps run in order.
- `--tune-kf` estimates Kalman filter noise (Q/R) from ground truth before tracking (`eval` and `tune` only).

See [Evaluation and Postprocessing](../guides/evaluation.md) and [Experiment Workflows](../guides/experiments.md) for details.

### ReID model lifecycle

Use `train-reid`, `eval-reid`, `compare-reid`, and `export` when you are working on
the appearance model itself rather than the full tracking loop.

```bash
boxmot train-reid --model osnet_x0_25 --dataset market1501 --data-dir /data/reid
boxmot eval-reid --weights runs/reid_train/exp/best.pt --dataset market1501 --data-dir /data/reid
boxmot compare-reid --weights runs/reid_train/exp/best.pt --target msmt17=/data/reid
boxmot export --weights runs/reid_train/exp/best.pt --include onnx
```

## Shared CLI shape

All BoxMOT modes start from the same command group:

```bash
boxmot MODE [OPTIONS]
```

Commands that select runtime components take them as options, for example
`--detector`, `--reid`, and `--tracker`; they are not positional arguments.

See [CLI](../usage/index.md) for the high-level syntax. Each mode page below includes its own examples and a generated CLI argument table.

## Python API path

If you want the same workflows from Python, start with the [Python API Overview](../python/index.md). The public facade is `boxmot.BoxMOT`.
