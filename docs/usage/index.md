# CLI

BoxMOT exposes one command group for all supported workflows:

```bash
boxmot MODE [OPTIONS]
```

## Core idea

- `MODE` selects the workflow such as `track`, `generate`, or `eval`.
- `--detector` selects detector weights or a model identifier such as `yolov8n`.
- `--reid` selects ReID weights or a model identifier such as
  `osnet_x0_25_msmt17`.
- `--tracker` selects the tracker implementation and its YAML config.
- `--tracker-backend cpp` selects a native C++ tracker implementation when one is registered.
- ReID model lifecycle commands are available through `boxmot train-reid`,
  `boxmot eval-reid`, `boxmot compare-reid`, and `boxmot export`.
- `boxmot build` prebuilds native ReID and live tracker libraries.

Legacy aliases such as `--yolo-model`, `--reid-model`, and `--tracking-method` are not part of the current CLI.

## Common examples

Track a video:

```bash
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --source video.mp4 --save
```

Evaluate a tracker on a benchmark:

```bash
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack --verbose
```

Run a native C++ tracker backend:

```bash
boxmot track --detector yolov8n --tracker bytetrack --tracker-backend cpp --source video.mp4
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker bytetrack --tracker-backend cpp
```

Export a ReID model:

```bash
boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --include engine --dynamic
```

Train a ReID model:

```bash
boxmot train-reid --model osnet_x0_25 --dataset market1501 --data-dir /data/reid
```

Evaluate a trained ReID model:

```bash
boxmot eval-reid --weights runs/reid_train/exp/best.pt --dataset market1501 --data-dir /data/reid
```

Compare ReID checkpoints across target datasets:

```bash
boxmot compare-reid --weights runs/reid_train/exp/best.pt --target msmt17=/data/reid
```

Run GEPA-based research:

```bash
boxmot research --experiment mot17-ablation-yolox-lmbn --tracker bytetrack --proposal-model openai/gpt-5.4 --max-metric-calls 24
```

## Source, dataset, and experiment inputs

The tracking workflows accept these mutually exclusive input forms:

| Mode | Input contract |
| --- | --- |
| `track` | `--source <input>` (webcam `0` by default) or `--dataset <id-or-yaml>` |
| `generate` | exactly one of `--source <dataset-path>` or `--experiment <id-or-yaml>` |
| `eval` | exactly one of `--dataset <id-or-yaml>` or `--experiment <id-or-yaml>` |
| `tune` | `--experiment <id-or-yaml>` |
| `research` | `--experiment <id-or-yaml>` |

For `track`, a concrete source can be `0`, `video.mp4`, `path/`, or
`rtsp://...`. It can also resolve a model-free dataset config and optional split:

```bash
boxmot track --dataset mot17 --split ablation --tracker botsort
```

Use experiment-driven modes when you want BoxMOT to resolve dataset, detector,
ReID, and artifact profiles automatically from the central catalog. The CLI
uses `--experiment` with an experiment ID or explicit experiment YAML:

```bash
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack
```

For `track` and `eval`, `--dataset` selects a model-free dataset profile while
retaining the selected or default detector and ReID models. `--split` overrides
the dataset's `default_split`:

```bash
boxmot eval --dataset mot17 --split ablation --tracker boosttrack
```

## Full argument tables

Each mode page includes its own generated CLI argument table. Direct links:

- [Track](../modes/track.md)
- [Generate](../modes/generate.md)
- [Eval](../modes/eval.md)
- [Tune](../modes/tune.md)
- [Research](../modes/research.md)
- [Train ReID](../modes/train.md)
- [Evaluate ReID](../modes/eval-reid.md)
- [Compare ReID](../modes/compare-reid.md)
- [Export](../modes/export.md)
- [Native C++ Integration and build](../native/index.md)
