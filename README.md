<div align="center" markdown="1">

  <img width="640"
       src="https://github.com/mikel-brostrom/boxmot/releases/download/v12.0.0/output_640.gif"
       alt="BoxMOT demo">
  <br>

  <a href="https://trendshift.io/repositories/13239" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13239" alt="mikel-brostrom%2Fboxmot | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"></a>

  [![CI](https://github.com/mikel-brostrom/boxmot/actions/workflows/ci.yml/badge.svg)](https://github.com/mikel-brostrom/boxmot/actions/workflows/ci.yml)
  [![PyPI version](https://badge.fury.io/py/boxmot.svg)](https://badge.fury.io/py/boxmot)
  [![downloads](https://static.pepy.tech/badge/boxmot)](https://pepy.tech/project/boxmot)
  [![license](https://img.shields.io/badge/license-AGPL%203.0-blue)](https://github.com/mikel-brostrom/boxmot/blob/master/LICENSE)
  [![python-version](https://img.shields.io/pypi/pyversions/boxmot)](https://badge.fury.io/py/boxmot)
  [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18nIqkBr68TkK8dHdarxTco6svHUJGggY?usp=sharing)
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8132989.svg)](https://doi.org/10.5281/zenodo.8132989)
  [![docker pulls](https://img.shields.io/docker/pulls/boxmot/boxmot?logo=docker)](https://hub.docker.com/r/boxmot/boxmot)
  [![discord](https://img.shields.io/discord/1377565354326495283?logo=discord&label=discord&labelColor=fff&color=5865f2)](https://discord.gg/tUmFEcYU4q)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mikel-brostrom/boxmot)

</div>

BoxMOT gives you one CLI and one Python API for running, evaluating, tuning, and exporting modern multi-object tracking pipelines. Swap trackers without rewriting your detector stack, reuse cached detections and embeddings across experiments, and benchmark locally on MOT-style datasets.

<div align="center" markdown="1">

[Installation](#installation) • [Metrics](#benchmark-results-mot17-ablation-split) • [CLI](#cli) • [Python API](#python-api) • [Detection Layouts](#detection-layouts) • [Examples](#examples) • [Contributing](#contributing)

</div>

## Why BoxMOT

- One interface for `track`, `generate`, `eval`, `tune`, and `export`.
- Works with detection, segmentation, and pose models as long as they emit boxes.
- Supports both motion-only trackers and motion + appearance trackers.
- Reuses saved detections and embeddings to speed up repeated evaluation and tuning.
- Handles both AABB and OBB detection layouts natively.
- Includes local benchmarking workflows for MOT17, MOT20, and DanceTrack ablation splits.

## Installation

BoxMOT supports Python `3.9` through `3.12`.

```bash
pip install boxmot
boxmot --help
```

## Benchmark Results (MOT17 ablation split)

<div align="center" markdown="1">

<!-- START TRACKER TABLE -->
| Tracker | Status  | OBB | HOTA↑ | MOTA↑ | IDF1↑ | FPS |
| :-----: | :-----: | :-: | :---: | :---: | :---: | :---: |
| [botsort](https://arxiv.org/abs/2206.14651) | ✅ | ✅ | 69.418 | 78.232 | 81.812 | 12 |
| [boosttrack](https://arxiv.org/abs/2408.13003) | ✅ | ❌ | 69.253 | 75.914 | 83.206 | 13 |
| [strongsort](https://arxiv.org/abs/2202.13514) | ✅ | ❌ | 68.05 | 76.185 | 80.763 | 11 |
| [deepocsort](https://arxiv.org/abs/2302.11813) | ✅ | ❌ | 67.796 | 75.868 | 80.514 | 12 |
| [bytetrack](https://arxiv.org/abs/2110.06864) | ✅ | ✅ | 67.68 | 78.039 | 79.157 | 720 |
| [hybridsort](https://arxiv.org/abs/2308.00783) | ✅ | ❌ | 67.39 | 74.127 | 79.105 | 25 |
| [ocsort](https://arxiv.org/abs/2203.14360) | ✅ | ✅ | 66.441 | 74.548 | 77.899 | 890 |
| [sfsort](https://arxiv.org/pdf/2404.07553) | ✅ | ✅ | 62.653 | 76.87 | 69.184 | 6000 |
<!-- END TRACKER TABLE -->

<sub>Evaluation was run on the second half of the MOT17 training set because the validation split is not public and the ablation detector was trained on the first half. Results used [pre-generated detections and embeddings](https://github.com/mikel-brostrom/boxmot/releases/download/v11.0.9/runs2.zip) with each tracker configured from its default repository settings.</sub>

</div>

## CLI

BoxMOT provides a unified CLI with a simple syntax:

```bash
boxmot MODE [OPTIONS] [DETECTOR] [REID] [TRACKER]
```

Modes:

```text
track      run detector + tracker on webcam, images, videos, directories, or streams
generate   precompute detections and embeddings for later reuse
eval       benchmark on MOT-style datasets and apply optional postprocessing
tune       optimize tracker hyperparameters with multi-objective search
export     export ReID models to deployment formats
```

Use `boxmot MODE --help` for mode-specific flags.

Use `--detector`, `--reid`, and `--tracker` for explicit component selection. Legacy aliases such as `--yolo-model`, `--reid-model`, and `--tracking-method` are not supported.

Quick examples:

```bash
# Track a webcam feed
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker deepocsort --source 0 --show

# Track a video, draw trajectories, and save the result
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --source video.mp4 --show-trajectories --save

# Evaluate on the MOT17 ablation split with GBRC postprocessing
boxmot eval --benchmark mot17-ablation --tracker boosttrack --postprocessing gbrc --verbose

# Generate reusable detections and embeddings for a benchmark
boxmot generate --benchmark mot17-ablation

# Tune tracker hyperparameters on a benchmark
boxmot tune --benchmark mot17-ablation --tracker ocsort --n-trials 10

# Export a ReID model to ONNX and TensorRT with dynamic input
boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --include engine --dynamic
```

Common `--source` values for `track` and direct-source `generate` runs include `0`, `img.jpg`, `video.mp4`, `path/`, `path/*.jpg`, YouTube URLs, and RTSP / RTMP / HTTP streams.

For config-driven `generate`, `eval`, and `tune` runs:

- `--benchmark <benchmark>` selects a benchmark config from `boxmot/configs/benchmarks/`
- the benchmark config selects its associated dataset config from `boxmot/configs/datasets/`
- the benchmark config selects its associated detector profile from `boxmot/configs/detectors/`
- the benchmark config selects its associated ReID profile from `boxmot/configs/reid/`
- `--tracker <name>` selects the tracker and loads `boxmot/configs/trackers/<name>.yaml`

Example:

```bash
boxmot eval --benchmark mot17-ablation --tracker boosttrack
```

The benchmark config's associated dataset, detector, and ReID profiles are used automatically.

To override the benchmark's detector and ReID defaults explicitly:

```bash
boxmot eval --benchmark mot17-ablation --detector yolo11s_obb --reid lmbn_n_duke --tracker boosttrack
```

If you want to track only selected classes, pass a comma-separated list:

```bash
boxmot track --detector yolov8s --source 0 --classes 16,17
```

## Python API

If you already have detections from your own model, call `tracker.update(...)` once per frame inside your video loop:

```python
from pathlib import Path

import cv2
import numpy as np
from boxmot import BotSort

tracker = BotSort(
    reid_weights=Path("osnet_x0_25_msmt17.pt"),
    device="cpu",
    half=False,
)

cap = cv2.VideoCapture("video.mp4")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Replace this with your detector output for the current frame.
    # AABB input: (N, 6) = (x1, y1, x2, y2, conf, cls)
    # OBB input: (N, 7) = (cx, cy, w, h, angle, conf, cls)
    detections = np.empty((0, 6), dtype=np.float32)
    # detections = your_detector(frame)

    tracks = tracker.update(detections, frame)
    tracker.plot_results(frame, show_trajectories=True)

    print(tracks)
    # AABB output: (N, 8) = (x1, y1, x2, y2, id, conf, cls, det_ind)
    # OBB output: (N, 9) = (cx, cy, w, h, angle, id, conf, cls, det_ind)
    # Use det_ind to map a track back to the detector output

    cv2.imshow("BoxMOT", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

For end-to-end detector integrations, see the notebooks in [examples](examples).

## Detection Layouts

BoxMOT switches tracking mode from the detection tensor shape:

| Geometry | Input detections | Output tracks |
| --- | --- | --- |
| AABB | `(N, 6)` = `(x1, y1, x2, y2, conf, cls)` | `(N, 8)` = `(x1, y1, x2, y2, id, conf, cls, det_ind)` |
| OBB | `(N, 7)` = `(cx, cy, w, h, angle, conf, cls)` | `(N, 9)` = `(cx, cy, w, h, angle, id, conf, cls, det_ind)` |

OBB-specific tracking paths are enabled automatically when OBB detections are provided. Current OBB-capable trackers: `bytetrack`, `botsort`, `ocsort`, and `sfsort`.

## Examples

The short commands above are enough to get started. The sections below keep the longer recipe list available without turning the README into a wall of commands.

<details>
<summary><strong>Tracking recipes</strong></summary>

Track from common sources:

```bash
# Webcam
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker deepocsort --source 0 --show

# Video file
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --source video.mp4 --save

# Image directory
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker bytetrack --source path/to/images --save

# Stream or URL
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker ocsort --source 'rtsp://example.com/media.mp4'

# YouTube
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker boosttrack --source 'https://youtu.be/Zgi9g1ksQHc'
```

</details>

<details>
<summary><strong>Detector backends</strong></summary>

Swap detectors without changing the overall CLI:

```bash
# Ultralytics detection
boxmot track --detector yolov8n
boxmot track --detector yolo11n

# Segmentation and pose variants
boxmot track --detector yolov8n-seg
boxmot track --detector yolov8n-pose

# YOLOX
boxmot track --detector yolox_s

# RF-DETR
boxmot track --detector rf-detr-base
```

</details>

<details>
<summary><strong>Tracker swaps</strong></summary>

Use the same detector and ReID model while changing only the tracker:

```bash
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker deepocsort
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker strongsort
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker boosttrack
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker hybridsort

# Motion-only trackers
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker bytetrack
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker ocsort
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker sfsort
```

</details>

<details>
<summary><strong>Filtering and visualization</strong></summary>

Useful flags for inspection and debugging:

```bash
# Draw trajectories and show lost tracks
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --source video.mp4 --show-trajectories --show-lost --save

# Track only selected classes
boxmot track --detector yolov8s --source 0 --classes 16,17

# Track each class independently
boxmot track --detector yolov8n --source video.mp4 --per-class --save

# Highlight one target ID
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker deepocsort --source video.mp4 --target-id 7 --show
```

</details>

<details>
<summary><strong>Evaluation and tuning</strong></summary>

Benchmark on built-in MOT-style dataset shortcuts:

```bash
# Reproduce README-style MOT17 results
boxmot eval --benchmark mot17-ablation --tracker boosttrack --verbose

# MOT20 ablation split
boxmot eval --benchmark mot20-ablation --tracker boosttrack --verbose

# DanceTrack ablation split
boxmot eval --benchmark dancetrack-ablation --tracker boosttrack --verbose

# VisDrone ablation split
boxmot eval --benchmark visdrone-ablation --tracker botsort --verbose

# Apply postprocessing
boxmot eval --benchmark mot17-ablation --tracker boosttrack --postprocessing gsi
boxmot eval --benchmark mot17-ablation --tracker boosttrack --postprocessing gbrc

# Generate detections and embeddings once for a benchmark
boxmot generate --benchmark mot17-ablation

# Generate detections and embeddings for a direct dataset path
boxmot generate --detector yolov8n --reid osnet_x0_25_msmt17 --source ./assets/MOT17-mini/train

# Tune on a built-in benchmark config
boxmot tune --benchmark mot17-ablation --tracker boosttrack --n-trials 9

# Tune a tracker with explicit detector/ReID overrides
boxmot tune --benchmark mot17-ablation --detector yolo11s_obb --reid lmbn_n_duke --tracker botsort --n-trials 9
```

</details>

<details>
<summary><strong>Export and OBB</strong></summary>

Deployment and oriented-box examples:

```bash
# Export to ONNX
boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --device cpu

# Export to OpenVINO
boxmot export --weights osnet_x0_25_msmt17.pt --include openvino --device cpu

# Export to TensorRT with dynamic input
boxmot export --weights osnet_x0_25_msmt17.pt --include engine --device 0 --dynamic
```

OBB references:

- Notebook: [examples/det/obb.ipynb](examples/det/obb.ipynb)
- OBB-capable trackers: `bytetrack`, `botsort`, `ocsort`, `sfsort`

</details>

## Contributing

If you want to contribute, start with [CONTRIBUTING.md](CONTRIBUTING.md).

## Contributors

<a href="https://github.com/mikel-brostrom/boxmot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mikel-brostrom/boxmot" alt="BoxMOT contributors">
</a>

## Support and Citation

- Bugs and feature requests: [GitHub Issues](https://github.com/mikel-brostrom/boxmot/issues)
- Questions and discussion: [GitHub Discussions](https://github.com/mikel-brostrom/boxmot/discussions) or [Discord](https://discord.gg/tUmFEcYU4q)
- Citation metadata: [CITATION.cff](CITATION.cff)
- Commercial support: `box-mot@outlook.com`
