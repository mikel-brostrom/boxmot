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

Where:

```text
MODE      (required) one of [track, eval, tune, generate, export]
DETECTOR  (optional) model like yolov8n, yolov9c, yolo11m, yolox_x, rf-detr-base
REID      (optional) model like osnet_x0_25_msmt17, mobilenetv2_x1_4, lmbn_n_duke
TRACKER   (optional) one of [deepocsort, botsort, bytetrack, strongsort, ocsort, hybridsort, boosttrack, sfsort]
OPTIONS   (optional) flags like --source 0, --imgsz 640, --postprocessing gbrc
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

Quick examples:

```bash
# Track a webcam feed
boxmot track yolov8n osnet_x0_25_msmt17 deepocsort --source 0 --show

# Track a video, draw trajectories, and save the result
boxmot track yolov8n osnet_x0_25_msmt17 botsort --source video.mp4 --show-trajectories --save

# Evaluate on the MOT17 ablation split with GBRC postprocessing
boxmot eval yolox_x_MOT17_ablation lmbn_n_duke boosttrack --source MOT17-ablation --postprocessing gbrc --verbose

# Generate reusable detections and embeddings
boxmot generate yolov8n osnet_x0_25_msmt17 --source ./assets/MOT17-mini/train

# Tune tracker hyperparameters on a MOT-style dataset
boxmot tune yolov8n osnet_x0_25_msmt17 ocsort --source ./assets/MOT17-mini/train --n-trials 10

# Export a ReID model to ONNX and TensorRT with dynamic input
boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --include engine --dynamic
```

Common `--source` values include `0`, `img.jpg`, `video.mp4`, `path/`, `path/*.jpg`, YouTube URLs, and RTSP / RTMP / HTTP streams.

If you want to track only selected classes, pass a comma-separated list:

```bash
boxmot track yolov8s --source 0 --classes 16,17
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
    # Expected AABB shape: (N, 6) = (x1, y1, x2, y2, conf, cls)
    detections = np.empty((0, 6), dtype=np.float32)
    # detections = your_detector(frame)

    tracks = tracker.update(detections, frame)
    tracker.plot_results(frame, show_trajectories=True)

    print(tracks)
    # AABB output: (N, 8) = (x1, y1, x2, y2, id, conf, cls, det_ind)

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
boxmot track yolov8n osnet_x0_25_msmt17 deepocsort --source 0 --show

# Video file
boxmot track yolov8n osnet_x0_25_msmt17 botsort --source video.mp4 --save

# Image directory
boxmot track yolov8n osnet_x0_25_msmt17 bytetrack --source path/to/images --save-txt

# Stream or URL
boxmot track yolov8n osnet_x0_25_msmt17 ocsort --source 'rtsp://example.com/media.mp4'

# YouTube
boxmot track yolov8n osnet_x0_25_msmt17 boosttrack --source 'https://youtu.be/Zgi9g1ksQHc'
```

</details>

<details>
<summary><strong>Detector backends</strong></summary>

Swap detectors without changing the overall CLI:

```bash
# Ultralytics detection
boxmot track yolov8n
boxmot track yolo11n

# Segmentation and pose variants
boxmot track yolov8n-seg
boxmot track yolov8n-pose

# YOLOX
boxmot track yolox_s

# RF-DETR
boxmot track rf-detr-base
```

</details>

<details>
<summary><strong>Tracker swaps</strong></summary>

Use the same detector and ReID model while changing only the tracker:

```bash
boxmot track yolov8n osnet_x0_25_msmt17 deepocsort
boxmot track yolov8n osnet_x0_25_msmt17 strongsort
boxmot track yolov8n osnet_x0_25_msmt17 botsort
boxmot track yolov8n osnet_x0_25_msmt17 boosttrack
boxmot track yolov8n osnet_x0_25_msmt17 hybridsort

# Motion-only trackers
boxmot track yolov8n osnet_x0_25_msmt17 bytetrack
boxmot track yolov8n osnet_x0_25_msmt17 ocsort
boxmot track yolov8n osnet_x0_25_msmt17 sfsort
```

</details>

<details>
<summary><strong>Filtering and visualization</strong></summary>

Useful flags for inspection and debugging:

```bash
# Draw trajectories and show lost tracks
boxmot track yolov8n osnet_x0_25_msmt17 botsort --source video.mp4 --show-trajectories --show-lost --save

# Track only selected classes
boxmot track yolov8s --source 0 --classes 16,17

# Track each class independently
boxmot track yolov8n --source video.mp4 --per-class --save-txt

# Highlight one target ID
boxmot track yolov8n osnet_x0_25_msmt17 deepocsort --source video.mp4 --target-id 7 --show
```

</details>

<details>
<summary><strong>Evaluation and tuning</strong></summary>

Benchmark on built-in MOT-style dataset shortcuts or your own data:

```bash
# Reproduce README-style MOT17 results
boxmot eval yolox_x_MOT17_ablation lmbn_n_duke boosttrack --source MOT17-ablation --verbose

# MOT20 ablation split
boxmot eval yolox_x_MOT20_ablation lmbn_n_duke boosttrack --source MOT20-ablation --verbose

# DanceTrack ablation split
boxmot eval yolox_x_dancetrack_ablation lmbn_n_duke boosttrack --source dancetrack-ablation --verbose

# VisDrone ablation split
boxmot eval yolox_x_visdrone lmbn_n_duke botsort --source visdrone-ablation --verbose

# Apply postprocessing
boxmot eval yolox_x_MOT17_ablation lmbn_n_duke boosttrack --source MOT17-ablation --postprocessing gsi
boxmot eval yolox_x_MOT17_ablation lmbn_n_duke boosttrack --source MOT17-ablation --postprocessing gbrc

# Generate detections and embeddings once
boxmot generate yolov8n osnet_x0_25_msmt17 --source ./assets/MOT17-mini/train

# Tune a tracker on a custom MOT-style dataset
boxmot tune yolov8n osnet_x0_25_msmt17 botsort --source ./assets/MOT17-mini/train --n-trials 9
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
- Script: [examples/det/run_obb_kalman.py](examples/det/run_obb_kalman.py)
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
