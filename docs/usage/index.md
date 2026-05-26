# CLI

BoxMOT exposes one command group for all supported workflows:

```bash
boxmot MODE [OPTIONS]
```

## Core idea

- `MODE` selects the workflow such as `track`, `generate`, or `eval`.
- `--detector` selects the detector backend or profile.
- `--reid` selects the appearance model or profile.
- `--tracker` selects the tracker implementation and its YAML config.
- `--tracker-backend cpp` selects a native C++ tracker implementation when one is registered.

Legacy aliases such as `--yolo-model`, `--reid-model`, and `--tracking-method` are not part of the current CLI.

## Common examples

Track a video:

```bash
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --source video.mp4 --save
```

Evaluate a tracker on a benchmark:

```bash
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --verbose
```

Run a native C++ tracker backend:

```bash
boxmot track --detector yolov8n --tracker bytetrack --tracker-backend cpp --source video.mp4
boxmot eval --benchmark mot17 --split ablation --tracker bytetrack:cpp
```

Export a ReID model:

```bash
boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --include engine --dynamic
```

Run GEPA-based research:

```bash
boxmot research --benchmark mot17 --split ablation --tracker bytetrack --proposal-model openai/gpt-5.4 --max-metric-calls 24
```

## Direct source vs benchmark configs

Use `track` when you already have a concrete source such as `0`, `video.mp4`, `path/`, or `rtsp://...`.

Use benchmark-driven modes when you want BoxMOT to resolve dataset, detector, and ReID profiles automatically from config files:

```bash
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack
```

## Full argument tables

Each mode page includes its own generated CLI argument table. Direct links:

- [Track](../modes/track.md)
- [Generate](../modes/generate.md)
- [Eval](../modes/eval.md)
- [Tune](../modes/tune.md)
- [Research](../modes/research.md)
- [Export](../modes/export.md)
