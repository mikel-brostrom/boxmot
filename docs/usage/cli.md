---
description: Learn BoxMOT's CLI syntax, component selection, benchmark config flow, and common output paths.
---

# CLI Usage

BoxMOT exposes one command surface whether you install it from PyPI or run it from a source checkout.

```bash
# installed package
boxmot MODE [OPTIONS] [DETECTOR] [REID] [TRACKER]

# source checkout
uv run python -m boxmot.engine.cli MODE [OPTIONS] [DETECTOR] [REID] [TRACKER]
```

`MODE` is one of:

- `track`
- `generate`
- `eval`
- `tune`
- `export`

## Canonical Flags

BoxMOT treats the detector, ReID model, and tracker as separate runtime components.

Use these canonical selectors:

- `--detector` for the detector weights or detector name
- `--reid` for the ReID weights or ReID name
- `--tracker` for the tracker name
- `--benchmark` for benchmark bundles; `--data` is kept as an alias

Examples:

```bash
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --source video.mp4
boxmot eval --benchmark mot17-ablation --tracker boosttrack
boxmot tune --benchmark mot17-ablation --detector yolo11s_obb --reid lmbn_n_duke --tracker sfsort
```

Legacy aliases such as `--yolo-model`, `--reid-model`, and `--tracking-method` are intentionally not supported.

## Modes At A Glance

| Mode | When to use it | Required input | Main output |
| --- | --- | --- | --- |
| `track` | run detector + ReID + tracker on live or file-based inputs | `--source` | annotated media and optional MOT text output |
| `generate` | precompute detections and embeddings for later reuse | `--benchmark` or dataset `--source` | `.npy` caches under `project/dets_n_embs/` |
| `eval` | generate MOT results and score them with TrackEval | `--benchmark` | MOT text files and benchmark metrics |
| `tune` | search tracker hyperparameters from the tracker YAML | `--benchmark` | Ray Tune runs, best YAML, and summary files |
| `export` | export ReID backbones to deployment formats | `--weights` | TorchScript / ONNX / OpenVINO / TensorRT / TFLite artifacts |

## Two Input Styles

### Direct-Source Workflow

Use a concrete `--source` when you want to track files, streams, or already-local datasets.

Typical sources include:

| Source type | Example |
| --- | --- |
| Webcam | `--source 0` |
| Image | `--source image.jpg` |
| Video | `--source video.mp4` |
| Directory | `--source path/to/images` |
| Glob | `--source 'path/*.jpg'` |
| Stream | `--source rtsp://example.com/live` |
| URL / YouTube | `--source https://youtu.be/...` |

`track` always uses a direct source. `generate` can also use a direct dataset root.

### Benchmark-Driven Workflow

Use `--benchmark <name>` for config-driven `generate`, `eval`, and `tune` runs. `--data` is accepted as a legacy alias for the same setting.

Example:

```bash
boxmot eval --benchmark mot17-ablation --tracker boosttrack
```

The benchmark config chain looks like this:

1. `boxmot/configs/benchmarks/mot17-ablation.yaml`
2. linked dataset config from `boxmot/configs/datasets/`
3. linked detector profile from `boxmot/configs/detectors/`
4. linked ReID profile from `boxmot/configs/reid/`
5. tracker defaults or search space from `boxmot/configs/trackers/<tracker>.yaml`

This is the preferred workflow for reproducible experiments.

## Positional Shorthand

`track`, `generate`, `eval`, and `tune` also accept positional component arguments:

```bash
boxmot track yolov8n osnet_x0_25_msmt17 botsort --source video.mp4
boxmot generate yolov8n osnet_x0_25_msmt17 --source ./assets/MOT17-mini/train
boxmot eval boosttrack --benchmark mot17-ablation
boxmot tune deepocsort --benchmark mot17-ablation --n-trials 10
```

For benchmark-driven `eval` and `tune` runs, a single positional tracker name is interpreted as `[TRACKER]` so you can keep detector and ReID on their defaults.

Use flags in docs, scripts, and CI because they are more explicit and easier to maintain.

## Workflow Rules

- `track` requires `--source`.
- `generate` accepts either `--benchmark` or `--source`, but not both.
- `eval` and `tune` require `--benchmark` and do not accept `--source`.
- `generate` can auto-resolve `--source mot17-ablation` to the matching benchmark config when that value is not a real local path, but `--benchmark mot17-ablation` is clearer.
- `export` is ReID-only. It does not use detector or tracker selectors.

## Common Examples

### Track

```bash
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker deepocsort --source 0 --show
boxmot track --detector yolo11s_obb --reid lmbn_n_duke --tracker sfsort --source video.mp4 --save
```

### Generate

```bash
boxmot generate --benchmark mot17-ablation
boxmot generate --source ./assets/MOT17-mini/train --detector yolov8n --reid osnet_x0_25_msmt17
```

### Evaluate

```bash
boxmot eval --benchmark mot20-ablation --tracker boosttrack --verbose
boxmot eval --benchmark mot17-ablation --tracker boosttrack --postprocessing gsi
```

### Tune

```bash
boxmot tune --benchmark mot17-ablation --tracker ocsort --n-trials 10
boxmot tune --benchmark visdrone-ablation --tracker botsort --objectives HOTA --objectives MOTA --objectives IDF1
```

### Export

```bash
boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --device cpu
boxmot export --weights osnet_x0_25_msmt17.pt --include engine --device 0 --dynamic --half
```

## Frequently Used Options

- Runtime: `--imgsz`, `--conf`, `--iou`, `--device`, `--batch-size`, `--auto-batch`, `--resume`
- Filtering: `--classes`, `--per-class`, `--agnostic-nms`
- Visualization: `--show`, `--show-labels`, `--show-conf`, `--show-trajectories`, `--show-kf-preds`
- Outputs: `--save`, `--save-txt`, `--save-crop`, `--project`, `--name`, `--exist-ok`
- Postprocessing: `--postprocessing none|gsi|gbrc`

The mode pages under [Modes](../modes/index.md) include the generated option tables from the live Click command definitions.

## Output Layout

By default, BoxMOT writes under `runs/`.

- `track` writes annotated outputs under `runs/<name>/` or an incremented variant
- `generate` writes caches under `runs/dets_n_embs/`
- `eval` writes tracker result files under `runs/mot/`
- `tune` stores Ray Tune runs under `runs/ray/`
- `export` writes artifacts next to the resolved weights file under the BoxMOT weights directory

You can override the root with `--project` and the experiment name with `--name`.

## Class Filtering

Use `--classes` to keep only selected classes from the detector output:

```bash
boxmot track --detector yolov8s --source video.mp4 --classes 0,2
```

Use `--per-class` when you want BoxMOT to track each class independently.

## Postprocessing

`track`, `eval`, and `tune` expose `--postprocessing` with these options:

- `none`
- `gsi`
- `gbrc`

`eval` is the most common place to apply postprocessing because it benchmarks the tracker output after cache generation and MOT result writing.

## Next Steps

- Open [Modes Overview](../modes/index.md) for per-mode references.
- Open [Track Mode](../modes/track.md) for source-specific examples.
- Open [Evaluate Mode](../modes/eval.md) for benchmark config details.
- Open [Python API](python.md) if you want to integrate BoxMOT into your own code.
