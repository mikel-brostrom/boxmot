# **BoxMOT**: Pluggable SOTA multi-object tracking modules for segmentation, object detection and pose estimation models

<div align="center" markdown="1">

  <img width="640"
       src="https://github.com/mikel-brostrom/boxmot/releases/download/v12.0.0/output_640.gif"
       alt="BoxMot demo">
  <br> <!-- one blank line -->

  <a href="https://trendshift.io/repositories/13239" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13239" alt="mikel-brostrom%2Fboxmot | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

  [![CI](https://github.com/mikel-brostrom/yolov8_tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/mikel-brostrom/yolov8_tracking/actions/workflows/ci.yml)
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


## 🚀 Key Features

- **Pluggable Architecture**  
  Easily swap in/out SOTA multi-object trackers.

- **Universal Model Support**  
  Integrate with any segmentation, object-detection and pose-estimation models that outputs bounding boxes

- **Benchmark-Ready**  
  Local evaluation pipelines for MOT17, MOT20, and DanceTrack ablation datasets with "official" ablation detectors

- **Performance Modes**
  - **Motion-only**: for lightweight, CPU-efficient, high-FPS performance 
  - **Motion + Appearance**: Combines motion cues with appearance embeddings ([CLIPReID](https://arxiv.org/pdf/2211.13977.pdf), [LightMBN](https://arxiv.org/pdf/2101.10774.pdf), [OSNet](https://arxiv.org/pdf/1905.00953.pdf)) to maximize identity consistency and accuracy at a higher computational cost

- **Reusable Detections & Embeddings**  
  Save once, run evaluations with no redundant preprocessing lightning fast.


## 📊 Benchmark Results (MOT17 ablation split)

<div align="center" markdown="1">

<!-- START TRACKER TABLE -->
| Tracker | Status  | HOTA↑ | MOTA↑ | IDF1↑ | FPS |
| :-----: | :-----: | :---: | :---: | :---: | :---: |
| [botsort](https://arxiv.org/abs/2206.14651) | ✅ | 69.418 | 78.232 | 81.812 | 46 |
| [boosttrack](https://arxiv.org/abs/2408.13003) | ✅ | 69.254 | 75.921 | 83.205 | 25 |
| [strongsort](https://arxiv.org/abs/2202.13514) | ✅ | 68.05 | 76.185 | 80.763 | 17 |
| [deepocsort](https://arxiv.org/abs/2302.11813) | ✅ | 67.796 | 75.868 | 80.514 | 12 |
| [bytetrack](https://arxiv.org/abs/2110.06864) | ✅ | 67.68 | 78.039 | 79.157 | 1265 |
| [hybridsort](https://arxiv.org/abs/2308.00783) | ✅ | 67.39 | 74.127 | 79.105 | 25 |
| [ocsort](https://arxiv.org/abs/2203.14360) | ✅ | 66.441 | 74.548 | 77.899 | 1483 |

<!-- END TRACKER TABLE -->

<sub> NOTES: Evaluation was conducted on the second half of the MOT17 training set, as the validation set is not publicly available and the ablation detector was trained on the first half. We employed [pre-generated detections and embeddings](https://github.com/mikel-brostrom/boxmot/releases/download/v11.0.9/runs2.zip). Each tracker was configured using the default parameters from their official repositories. </sub>

</div>

</details>


## 🔧 Installation

Install the `boxmot` package, including all requirements, in a Python>=3.9 environment:

```bash
pip install boxmot
```

If you want to contribute to this package check how to contribute [here](https://github.com/mikel-brostrom/boxmot/blob/master/CONTRIBUTING.md)

## 💻 CLI

BoxMOT provides a unified CLI with a simple syntax:

```bash
boxmot MODE DETECTOR REID TRACKER ARGS

Where:
  MODE      (required) one of [track, eval, tune, generate, export]
  DETECTOR  (optional) YOLO model like yolov8n, yolov9c, yolo11m, yolox_x
  REID      (optional) ReID model like osnet_x0_25_msmt17, mobilenetv2_x1_4
  TRACKER   (optional) one of [deepocsort, botsort, bytetrack, strongsort, ocsort, hybridsort, boosttrack]
  ARGS      (optional) 'arg=value' pairs that override defaults
```

**Quick Examples:**
```bash
# Track with webcam, save results, show basic results
boxmot track yolov8n osnet_x0_25_msmt17 deepocsort --source 0 --show --save

# Track a video file, save results, show trajectories + lost tracks
boxmot track yolov8n osnet_x0_25_msmt17 botsort --source video.mp4 --save --show-trajectories --show-lost

# Evaluate on MOT dataset
boxmot eval yolox_x_MOT17_ablation lmbn_n_duke botsort --source MOT17-ablation

# Tune ocsort's hyperparameters for dancetrack
boxmot tune yolox_x_dancetrack_ablation lmbn_n_duke ocsort --source dancetrack-ablation --n-trials 10

# Export ReID model with dynamic sized input
boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --include engine dynamic
```

## 🐍 PYTHON

Seamlessly integrate BoxMOT directly into your Python MOT applications with your custom model.

```python
import cv2
import torch
import numpy as np
from pathlib import Path
from boxmot import BoostTrack
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights as Weights
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load detector with pretrained weights and preprocessing transforms
weights = Weights.DEFAULT
detector = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
detector.to(device).eval()
transform = weights.transforms()

# Initialize tracker
tracker = BoostTrack(reid_weights=Path('osnet_x0_25_msmt17.pt'), device=device, half=False)

# Start video capture
cap = cv2.VideoCapture(0)

with torch.inference_mode():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB and prepare for detector
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.uint8)
        input_tensor = transform(tensor).to(device)

        # Run detection
        output = detector([input_tensor])[0]
        scores = output['scores'].cpu().numpy()
        keep = scores >= 0.5

        # Prepare detections for tracking
        boxes = output['boxes'][keep].cpu().numpy()
        labels = output['labels'][keep].cpu().numpy()
        filtered_scores = scores[keep]
        detections = np.concatenate([boxes, filtered_scores[:, None], labels[:, None]], axis=1)

        # Update tracker and draw results
        #   INPUT:  M X (x, y, x, y, conf, cls)
        #   OUTPUT: M X (x, y, x, y, id, conf, cls, ind)
        res = tracker.update(detections, frame)
        tracker.plot_results(frame, show_trajectories=True)

        # Show output
        cv2.imshow('BoXMOT + Torchvision', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()
```


## 📝 Code Examples & Tutorials

<details>
<summary>Tracking</summary>

```bash
# Different detector models
boxmot track rf-detr-base                        # RF-DETR
boxmot track yolox_s                             # YOLOX  
boxmot track yolo12n                             # YOLO12
boxmot track yolo11n                             # YOLO11
boxmot track yolov10n                            # YOLOv10
boxmot track yolov9c                             # YOLOv9
boxmot track yolov8n                             # YOLOv8 bboxes only
boxmot track yolov8n-seg                         # YOLOv8 + segmentation masks
boxmot track yolov8n-pose                        # YOLOv8 + pose estimation
```

  </details>

<details>
<summary>Tracking methods</summary>

```bash
boxmot track yolov8n osnet_x0_25_msmt17 deepocsort
boxmot track yolov8n osnet_x0_25_msmt17 strongsort
boxmot track yolov8n osnet_x0_25_msmt17 ocsort
boxmot track yolov8n osnet_x0_25_msmt17 bytetrack
boxmot track yolov8n osnet_x0_25_msmt17 botsort
boxmot track yolov8n osnet_x0_25_msmt17 boosttrack
boxmot track yolov8n osnet_x0_25_msmt17 hybridsort
```

</details>

<details>
<summary>Tracking sources</summary>

Tracking can be run on most video formats

```bash
boxmot track yolov8n --source 0                               # webcam
boxmot track yolov8n --source img.jpg                         # image
boxmot track yolov8n --source vid.mp4                         # video
boxmot track yolov8n --source path/                           # directory
boxmot track yolov8n --source path/*.jpg                      # glob
boxmot track yolov8n --source 'https://youtu.be/Zgi9g1ksQHc'  # YouTube
boxmot track yolov8n --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Select ReID model</summary>

Some tracking methods combine appearance description and motion in the process of tracking. For those which use appearance, you can choose a ReID model based on your needs from this [ReID model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO). These models can be further optimized for your needs by the export command.

```bash
boxmot track yolov8n lmbn_n_cuhk03_d botsort --source 0           # lightweight
boxmot track yolov8n osnet_x0_25_market1501 botsort --source 0
boxmot track yolov8n mobilenetv2_x1_4_msmt17 botsort --source 0
boxmot track yolov8n resnet50_msmt17 botsort --source 0
boxmot track yolov8n osnet_x1_0_msmt17 botsort --source 0
boxmot track yolov8n clip_market1501 botsort --source 0           # heavy
boxmot track yolov8n clip_vehicleid botsort --source 0
```

</details>

<details>
<summary>Filter tracked classes</summary>

By default the tracker tracks all MS COCO classes.

If you want to track a subset of the classes that your model predicts, add their corresponding index after the classes flag:

```bash
boxmot track yolov8s --source 0 --classes 16 17  # Track cats and dogs only
```

[Here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) is a list of all the possible objects that a YOLOv8 model trained on MS COCO can detect. Notice that the indexing for the classes in this repo starts at zero

</details>

<details>
<summary>Evaluation</summary>

Evaluate a combination of detector, tracking method and ReID model on standard MOT dataset or your custom one:

```bash
# reproduce MOT17 README results
boxmot eval yolox_x_MOT17_ablation lmbn_n_duke boosttrack --source MOT17-ablation --verbose 
# MOT20 results
boxmot eval yolox_x_MOT20_ablation lmbn_n_duke boosttrack --source MOT20-ablation --verbose 
# DanceTrack results
boxmot eval yolox_x_dancetrack_ablation lmbn_n_duke boosttrack --source dancetrack-ablation --verbose 
# metrics on custom dataset
boxmot eval yolov8n osnet_x0_25_msmt17 deepocsort --source ./assets/MOT17-mini/train --verbose
```

Add `--gsi` to your command for postprocessing the MOT results by Gaussian smoothed interpolation. Detections and embeddings are stored for the selected YOLO and ReID model respectively. They can then be loaded into any tracking algorithm, avoiding the overhead of repeatedly generating this data.
</details>


<details>
<summary>Hyperparameter Tuning</summary>

We use a fast and elitist multiobjective genetic algorithm for tracker hyperparameter tuning. By default the objectives are: HOTA, MOTA, IDF1.

```bash
# Generate detections and embeddings (saves under ./runs/dets_n_embs)
boxmot generate yolov8n osnet_x0_25_msmt17 --source ./assets/MOT17-mini/train

# Tune parameters for specified tracking method
boxmot tune --yolo-model yolov8n.pt --reid-model osnet_x0_25_msmt17.pt --n-trials 9 --tracking-method botsort --source ./assets/MOT17-mini/train
```

The set of hyperparameters leading to the best HOTA result are written to the tracker's config file.

</details>

<details>
<summary>Export</summary>

We support ReID model export to ONNX, OpenVINO, TorchScript and TensorRT:

```bash
# export to ONNX
boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --device cpu
# export to OpenVINO
boxmot export --weights osnet_x0_25_msmt17.pt --include openvino --device cpu
# export to TensorRT with dynamic input
boxmot export --weights osnet_x0_25_msmt17.pt --include engine --device 0 --dynamic
```

</details>


<div align="center" markdown="1">

| Example Description | Notebook |
|---------------------|----------|
| Torchvision bounding box tracking with BoxMOT | [![Notebook](https://img.shields.io/badge/Notebook-torchvision_det_boxmot.ipynb-blue)](examples/det/torchvision_boxmot.ipynb) |
| Torchvision pose tracking with BoxMOT | [![Notebook](https://img.shields.io/badge/Notebook-torchvision_pose_boxmot.ipynb-blue)](examples/pose/torchvision_boxmot.ipynb) |
| Torchvision segmentation tracking with BoxMOT | [![Notebook](https://img.shields.io/badge/Notebook-torchvision_seg_boxmot.ipynb-blue)](examples/seg/torchvision_boxmot.ipynb) |

</div>

## Contributors

<a href="https://github.com/mikel-brostrom/boxmot/graphs/contributors ">
  <img src="https://contrib.rocks/image?repo=mikel-brostrom/boxmot" />
</a>

## Contact

For BoxMOT bugs and feature requests please visit [GitHub Issues](https://github.com/mikel-brostrom/boxmot/issues).
For business inquiries or professional support requests please send an email to: box-mot@outlook.com
