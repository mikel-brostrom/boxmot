# BoxMOT: pluggable SOTA tracking modules for segmentation, object detection and pose estimation models

<div align="center">

  <img width="640"
       src="https://github.com/mikel-brostrom/boxmot/releases/download/v12.0.0/output_640.gif"
       alt="BoxMot demo">
  <br> <!-- one blank line -->

  [![CI](https://github.com/mikel-brostrom/yolov8_tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/mikel-brostrom/yolov8_tracking/actions/workflows/ci.yml)
  [![PyPI version](https://badge.fury.io/py/boxmot.svg)](https://badge.fury.io/py/boxmot)
  [![downloads](https://static.pepy.tech/badge/boxmot)](https://pepy.tech/project/boxmot)
  [![license](https://img.shields.io/badge/license-AGPL%203.0-blue)](https://github.com/mikel-brostrom/boxmot/blob/master/LICENSE)
  [![python-version](https://img.shields.io/pypi/pyversions/boxmot)](https://badge.fury.io/py/boxmot)
  [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18nIqkBr68TkK8dHdarxTco6svHUJGggY?usp=sharing)
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8132989.svg)](https://doi.org/10.5281/zenodo.8132989)
  [![docker pulls](https://img.shields.io/docker/pulls/boxmot/boxmot?logo=docker)](https://hub.docker.com/r/boxmot/boxmot)
  [![discord](https://img.shields.io/discord/1377565354326495283?logo=discord&label=discord&labelColor=fff&color=5865f2)](https://discord.gg/3w4aYGbU)
</div>



## Introduction

This repository addresses the fragmented nature of the multi-object tracking (MOT) field by providing a standardized collection of pluggable, state-of-the-art trackers. Designed to seamlessly integrate with segmentation, object detection, and pose estimation models, the repository streamlines the adoption and comparison of MOT methods. For trackers employing appearance-based techniques, we offer a range of automatically downloadable state-of-the-art re-identification (ReID) models, from heavyweight ([CLIPReID](https://arxiv.org/pdf/2211.13977.pdf)) to lightweight options ([LightMBN](https://arxiv.org/pdf/2101.10774.pdf), [OSNet](https://arxiv.org/pdf/1905.00953.pdf)). Additionally, clear and practical examples demonstrate how to effectively integrate these trackers with various popular models, enabling versatility across diverse vision tasks.

<div align="center">

<!-- START TRACKER TABLE -->
| Tracker | Status  | HOTA↑ | MOTA↑ | IDF1↑ | FPS |
| :-----: | :-----: | :---: | :---: | :---: | :---: |
| [boosttrack](https://arxiv.org/abs/2408.13003) | ✅ | 69.253 | 75.914 | 83.206 | 25 |
| [botsort](https://arxiv.org/abs/2206.14651) | ✅ | 68.885 | 78.222 | 81.344 | 46 |
| [strongsort](https://arxiv.org/abs/2202.13514) | ✅ | 68.05 | 76.185 | 80.763 | 17 |
| [deepocsort](https://arxiv.org/abs/2302.11813) | ✅ | 67.796 | 75.868 | 80.514 | 12 |
| [bytetrack](https://arxiv.org/abs/2110.06864) | ✅ | 67.68 | 78.039 | 79.157 | 1265 |
| [ocsort](https://arxiv.org/abs/2203.14360) | ✅ | 66.441 | 74.548 | 77.899 | 1483 |

<!-- END TRACKER TABLE -->

<sub> NOTES: Evaluation was conducted on the second half of the MOT17 training set, as the validation set is not publicly available and the ablation detector was trained on the first half. We employed [pre-generated detections and embeddings](https://github.com/mikel-brostrom/boxmot/releases/download/v11.0.9/runs2.zip). Each tracker was configured using the default parameters from their official repositories. </sub>

</div>

</details>



## Why BOXMOT?

Multi-object tracking solutions today depend heavily on the computational capabilities of the underlying hardware. BoxMOT addresses this by offering a wide array of tracking methods tailored to accommodate diverse hardware constraints, ranging from CPU-only setups to high-end GPUs. Furthermore, we provide scripts designed for rapid experimentation, enabling users to save detections and embeddings once and subsequently reuse them with any tracking algorithm. This approach eliminates redundant computations, significantly speeding up the evaluation and comparison of multiple trackers.

## Installation

Install the `boxmot` package, including all requirements, in a Python>=3.9 environment:

```bash
pip install boxmot
```

BoxMOT provides a unified CLI `boxmot` with the following subcommands:

```bash
Usage: boxmot COMMAND [ARGS]...

Commands:
  track                  Run tracking only
  generate               Generate detections and embeddings
  eval                   Evaluate tracking performance using the official trackeval repository
  tune                   Tune tracker hyperparameters based on selected detections and embeddings
```

## YOLOv12 | YOLOv11 | YOLOv10 | YOLOv9 | YOLOv8 | RFDETR | YOLOX examples

<details>
<summary>Tracking</summary>

```bash
$ boxmot track --yolo-model rf-detr-base.pt     # bboxes only
  boxmot track --yolo-model yolox_s.pt          # bboxes only
  boxmot track --yolo-model yolo12n.pt         # bboxes only
  boxmot track --yolo-model yolo11n.pt         # bboxes only
  boxmot track --yolo-model yolov10n.pt         # bboxes only
  boxmot track --yolo-model yolov9c.pt          # bboxes only
  boxmot track --yolo-model yolov8n.pt          # bboxes only
                            yolov8n-seg.pt      # bboxes + segmentation masks
                            yolov8n-pose.pt     # bboxes + pose estimation
```

  </details>

<details>
<summary>Tracking methods</summary>

```bash
$ boxmot track --tracking-method deepocsort
                                 strongsort
                                 ocsort
                                 bytetrack
                                 botsort
                                 boosttrack
```

</details>

<details>
<summary>Tracking sources</summary>

Tracking can be run on most video formats

```bash
$ boxmot track --source 0                               # webcam
                        img.jpg                         # image
                        vid.mp4                         # video
                        path/                           # directory
                        path/*.jpg                      # glob
                        'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                        'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Select ReID model</summary>

Some tracking methods combine appearance description and motion in the process of tracking. For those which use appearance, you can choose a ReID model based on your needs from this [ReID model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO). These model can be further optimized for you needs by the [reid_export.py](https://github.com/mikel-brostrom/yolo_tracking/blob/master/boxmot/appearance/reid_export.py) script

```bash
$ boxmot track --source 0 --reid-model lmbn_n_cuhk03_d.pt               # lightweight
                                       osnet_x0_25_market1501.pt
                                       mobilenetv2_x1_4_msmt17.engine
                                       resnet50_msmt17.onnx
                                       osnet_x1_0_msmt17.pt
                                       clip_market1501.pt               # heavy
                                       clip_vehicleid.pt
                                      ...
```

</details>

<details>
<summary>Filter tracked classes</summary>

By default the tracker tracks all MS COCO classes.

If you want to track a subset of the classes that you model predicts, add their corresponding index after the classes flag,

```bash
boxmot track --source 0 --yolo-model yolov8s.pt --classes 16 17  # COCO yolov8 model. Track cats and dogs, only
```

[Here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) is a list of all the possible objects that a Yolov8 model trained on MS COCO can detect. Notice that the indexing for the classes in this repo starts at zero

</details>


</details>

<details>
<summary>Evaluation</summary>

Evaluate a combination of detector, tracking method and ReID model on standard MOT dataset or you custom one by

```bash
# reproduce MOT17 README results
$ boxmot eval --yolo-model yolox_x_ablation.pt --reid-model lmbn_n_duke.pt --tracking-method boosttrack --source MOT17-ablation --verbose 
# MOT20 results
$ boxmot eval --yolo-model yolox_x_ablation.pt --reid-model lmbn_n_duke.pt --tracking-method boosttrack --source MOT20-ablation --verbose 
# Dancetrack results
$ boxmot eval --yolo-model yolox_x_ablation.pt --reid-model lmbn_n_duke.pt --tracking-method boosttrack --source dancetrack-ablation --verbose 
# metrics on custom dataset
$ boxmot eval --yolo-model yolov8n.pt --reid-model osnet_x0_25_msmt17.pt --tracking-method deepocsort  --source ./assets/MOT17-mini/train --verbose
```

add `--gsi` to your command for postprocessing the MOT results by gaussian smoothed interpolation. Detections and embeddings are stored for the selected YOLO and ReID model respectively. They can then be loaded into any tracking algorithm. Avoiding the overhead of repeatedly generating this data.
</details>


<details>
<summary>Evolution</summary>

We use a fast and elitist multiobjective genetic algorithm for tracker hyperparameter tuning. By default the objectives are: HOTA, MOTA, IDF1. Run it by

```bash
# saves dets and embs under ./runs/dets_n_embs separately for each selected yolo and reid model
$ boxmot generate --source ./assets/MOT17-mini/train --yolo-model yolov8n.pt yolov8s.pt --reid-model weights/osnet_x0_25_msmt17.pt
# evolve parameters for specified tracking method using the selected detections and embeddings generated in the previous step
$ boxmot tune --dets yolov8n --embs osnet_x0_25_msmt17 --n-trials 9 --tracking-method botsort --source ./assets/MOT17-mini/train
```

The set of hyperparameters leading to the best HOTA result are written to the tracker's config file.

</details>

<details>
<summary>Export</summary>

We support ReID model export to ONNX, OpenVINO, TorchScript and TensorRT

```bash
# export to ONNX
$ python3 boxmot/appearance/reid_export.py --include onnx --device cpu
# export to OpenVINO
$ python3 boxmot/appearance/reid_export.py --include openvino --device cpu
# export to TensorRT with dynamic input
$ python3 boxmot/appearance/reid_export.py --include engine --device 0 --dynamic
```

</details>


## Custom tracking examples

<div align="center">

| Example Description | Notebook |
|---------------------|----------|
| Torchvision bounding box tracking with BoxMOT | [![Notebook](https://img.shields.io/badge/Notebook-torchvision_det_boxmot.ipynb-blue)](examples/det/torchvision_boxmot.ipynb) |
| Torchvision pose tracking with BoxMOT | [![Notebook](https://img.shields.io/badge/Notebook-torchvision_pose_boxmot.ipynb-blue)](examples/pose/torchvision_boxmot.ipynb) |
| Torchvision segmentation tracking with BoxMOT | [![Notebook](https://img.shields.io/badge/Notebook-torchvision_seg_boxmot.ipynb-blue)](examples/seg/torchvision_boxmot.ipynb) |

</div>

## Contributors

<a href="https://github.com/mikel-brostrom/yolo_tracking/graphs/contributors ">
  <img src="https://contrib.rocks/image?repo=mikel-brostrom/yolo_tracking" />
</a>

## Contact

For BoxMOT bugs and feature requests please visit [GitHub Issues](https://github.com/mikel-brostrom/boxmot/issues).
For business inquiries or professional support requests please send an email to: box-mot@outlook.com
