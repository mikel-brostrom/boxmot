# BoxMOT: pluggable SOTA tracking modules for segmentation, object detection and pose estimation models

<div align="center">
  <p>
  <img src="https://github.com/mikel-brostrom/boxmot/releases/download/v12.0.0/output_640.gif" width="400"/>
  </p>
  <br>
  <div>
  <a href="https://github.com/mikel-brostrom/yolov8_tracking/actions/workflows/ci.yml"><img src="https://github.com/mikel-brostrom/yolov8_tracking/actions/workflows/ci.yml/badge.svg" alt="CI CPU testing"></a>
  <a href="https://pepy.tech/project/boxmot"><img src="https://static.pepy.tech/badge/boxmot"></a>
  <br>
  <a href="https://colab.research.google.com/drive/18nIqkBr68TkK8dHdarxTco6svHUJGggY?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
<a href="https://doi.org/10.5281/zenodo.8132989"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8132989.svg" alt="DOI"></a>
<a href="https://hub.docker.com/r/boxmot/boxmot"><img src="https://img.shields.io/docker/pulls/boxmot/boxmot?logo=docker" alt="Ultralytics Docker Pulls"></a>

  </div>
</div>

## Introduction

This repository contains a collection of pluggable, state-of-the-art multi-object trackers designed to seamlessly integrate with segmentation, object detection, and pose estimation models. For methods leveraging appearance-based tracking, we offer both heavyweight  ([CLIPReID](https://arxiv.org/pdf/2211.13977.pdf)) and lightweight ([LightMBN](https://arxiv.org/pdf/2101.10774.pdf), [OSNet](https://arxiv.org/pdf/1905.00953.pdf)) state-of-the-art ReID models, available via automatic download. Additionally, clear and practical examples demonstrate how to effectively integrate these trackers with various popular models, enabling versatility across diverse vision tasks.

<div align="center">

<!-- START TRACKER TABLE -->
| Tracker | Status  | HOTA↑ | MOTA↑ | IDF1↑ | FPS |
| :-----: | :-----: | :---: | :---: | :---: | :---: |
| [boosttrack](https://arxiv.org/abs/2408.13003) | ✅ | 68.649 | 76.042 | 81.923 | 25 |
| [botsort](https://arxiv.org/abs/2206.14651) | ✅ | 68.251 | 78.328 | 80.622 | 46 |
| [bytetrack](https://arxiv.org/abs/2110.06864) | ✅ | 67.619 | 78.081 | 79.188 | 1265 |
| [strongsort](https://arxiv.org/abs/2202.13514) | ✅ | 67.394 | 76.413 | 79.017 | 17 |
| [deepocsort](https://arxiv.org/abs/2302.11813) | ✅ | 67.348 | 75.832 | 79.584 | 12 |
| [ocsort](https://arxiv.org/abs/2203.14360) | ✅ | 66.441 | 74.546 | 77.892 | 1483 |
| [imprassoc](https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/papers/Stadler_An_Improved_Association_Pipeline_for_Multi-Person_Tracking_CVPRW_2023_paper.pdf) | ✅ | 63.699 | 76.407 | 70.837 | 26 |

<!-- END TRACKER TABLE -->

<sub> NOTES: Evaluation was conducted on the second half of the MOT17 training set, as the validation set is not publicly available and the ablation detector was trained on the first half. We employed [pre-generated detections and embeddings](https://github.com/mikel-brostrom/boxmot/releases/download/v11.0.9/runs2.zip). Each tracker was configured using the default parameters from their official repositories. </sub>

</div>

</details>



## Why BOXMOT?

Multi-object tracking solutions today depend heavily on the computational capabilities of the underlying hardware. BoxMOT offers a wide range of tracking methods designed to accommodate various hardware constraints—from CPU-only setups to high-end GPUs. Additionally, we provide scripts for rapid experimentation that allow you to save detections and embeddings once, and then load them into any tracking algorithm, eliminating the need to repeatedly generate this data.

## Installation

Start with a [**Python>=3.9**](https://www.python.org/) environment.

If you want to run the RFDETR, YOLOX or YOLOv12 examples:

```
git clone https://github.com/mikel-brostrom/boxmot.git
cd boxmot
pip install uv
uv sync --group yolo
activate .venv/bin/activate
```

but if you only want to import the tracking modules you can simply:

```
pip install boxmot
```

## RFDETR | YOLOX | YOLOv12 examples

<details>
<summary>Tracking</summary>


```bash
yolox_s.pt
$ python tracking/track.py --yolo-model rf-detr-base.pt  # bboxes only
  python tracking/track.py --yolo-model yolox_s.pt       # bboxes only
  python tracking/track.py --yolo-model yolov10n         # bboxes only
  python tracking/track.py --yolo-model yolov9s          # bboxes only
  python tracking/track.py --yolo-model yolov8n          # bboxes only
                                        yolov8n-seg      # bboxes + segmentation masks
                                        yolov8n-pose     # bboxes + pose estimation

```

  </details>

<details>
<summary>Tracking methods</summary>

```bash
$ python tracking/track.py --tracking-method deepocsort
                                             strongsort
                                             ocsort
                                             bytetrack
                                             botsort
                                             imprassoc
                                             boosttrack
```

</details>

<details>
<summary>Tracking sources</summary>

Tracking can be run on most video formats

```bash
$ python tracking/track.py --source 0                               # webcam
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
$ python tracking/track.py --source 0 --reid-model lmbn_n_cuhk03_d.pt               # lightweight
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
python tracking/track.py --source 0 --yolo-model yolov8s.pt --classes 16 17  # COCO yolov8 model. Track cats and dogs, only
```

[Here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) is a list of all the possible objects that a Yolov8 model trained on MS COCO can detect. Notice that the indexing for the classes in this repo starts at zero

</details>


</details>

<details>
<summary>Evaluation</summary>

Evaluate a combination of detector, tracking method and ReID model on standard MOT dataset or you custom one by

```bash
$ python3 tracking/val.py --yolo-model yolov8n.pt --reid-model osnet_x0_25_msmt17.pt --tracking-method deepocsort --verbose --source ./assets/MOT17-mini/train
$ python3 tracking/val.py --yolo-model yolov8n.pt --reid-model osnet_x0_25_msmt17.pt --tracking-method ocsort     --verbose --source ./tracking/val_utils/MOT17/train
```

add `--gsi` to your command for postprocessing the MOT results by gaussian smoothed interpolation. Detections and embeddings are stored for the selected YOLO and ReID model respectively. They can then be loaded into any tracking algorithm. Avoiding the overhead of repeatedly generating this data.
</details>


<details>
<summary>Evolution</summary>

We use a fast and elitist multiobjective genetic algorithm for tracker hyperparameter tuning. By default the objectives are: HOTA, MOTA, IDF1. Run it by

```bash
# saves dets and embs under ./runs/dets_n_embs separately for each selected yolo and reid model
$ python tracking/generate_dets_n_embs.py --source ./assets/MOT17-mini/train --yolo-model yolov8n.pt yolov8s.pt --reid-model weights/osnet_x0_25_msmt17.pt
# evolve parameters for specified tracking method using the selected detections and embeddings generated in the previous step
$ python tracking/evolve.py --dets yolov8n --embs osnet_x0_25_msmt17 --n-trials 9 --tracking-method botsort --source ./assets/MOT17-mini/train
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
