# BoxMOT: pluggable SOTA tracking modules for segmentation, object detection and pose estimation models

<div align="center">
  <p>
  <img src="assets/images/track_all_seg_1280_025conf.gif" width="400"/>
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

This repo contains a collections of pluggable state-of-the-art multi-object trackers for segmentation, object detection and pose estimation models. For the methods using appearance description, both heavy ([CLIPReID](https://arxiv.org/pdf/2211.13977.pdf)) and lightweight state-of-the-art ReID models ([LightMBN](https://arxiv.org/pdf/2101.10774.pdf), [OSNet](https://arxiv.org/pdf/1905.00953.pdf) and more) are available for automatic download. We provide examples on how to use this package together with popular object detection models such as: [YOLOv8, YOLOv9 and YOLOv10](https://github.com/ultralytics)

<div align="center">

<!-- START TRACKER TABLE -->
| Tracker | Status  | HOTA↑ | MOTA↑ | IDF1↑ | FPS |
| :-----: | :-----: | :---: | :---: | :---: | :---: |
| [botsort](https://arxiv.org/abs/2206.14651) | ✅ | 68.504 | 77.165 | 80.986 | 46 |
| [strongsort](https://arxiv.org/abs/2202.13514) | ✅ | 68.329 | 76.348 | 81.206 | 17 |
| [bytetrack](https://arxiv.org/abs/2110.06864) | ✅ | 66.536 | 76.909 | 77.855 | 1265 |
| [ocsort](https://arxiv.org/abs/2203.14360) | ✅ | 65.187 | 74.819 | 75.957 | 1483 |
| [imprassoc](https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/papers/Stadler_An_Improved_Association_Pipeline_for_Multi-Person_Tracking_CVPRW_2023_paper.pdf) | ✅ | 64.096 | 76.511 | 71.875 | 26 |
| [deepocsort](https://arxiv.org/abs/2302.11813) | ✅ | 62.913 | 74.483 | 73.459 | 12 |
| [hybridsort](https://arxiv.org/abs/2308.00783) | ❌ |  |  |  |  |

<!-- END TRACKER TABLE -->

<sub> NOTES: The evaluation was conducted on the second half of the MOT17 training set, as the validation set is not publicly accessible. The pre-generated detections and embeddings used, were sourced from [here](https://drive.google.com/drive/folders/1zzzUROXYXt8NjxO1WUcwSzqD-nn7rPNr). Each tracker was configured with the original parameters provided in their official repositories. </sub>

</div>

</details>



## Why BOXMOT?

Today's multi-object tracking options are heavily dependant on the computation capabilities of the underlaying hardware. BoxMOT provides a great variety of tracking methods that meet different hardware limitations, all the way from CPU only to larger GPUs. Morover, we provide scripts for ultra fast experimentation by saving detections and embeddings, which then be loaded into any tracking algorithm. Avoiding the overhead of repeatedly generating this data.

## Installation

Start with [**Python>=3.9**](https://www.python.org/) environment.

If you want to run the YOLOv8, YOLOv9 or YOLOv10 examples:

```
git clone https://github.com/mikel-brostrom/boxmot.git
cd boxmot
pip install poetry
poetry install --with yolo  # installed boxmot + yolo dependencies
poetry shell  # activates the newly created environment with the installed dependencies
```

but if you only want to import the tracking modules you can simply:

```
pip install boxmot
```

## YOLOv8 | YOLOv9 | YOLOv10 examples

<details>
<summary>Tracking</summary>

<details>
<summary>Yolo models</summary>



```bash
$ python tracking/track.py --yolo-model yolov10n      # bboxes only
  python tracking/track.py --yolo-model yolov9s       # bboxes only
  python tracking/track.py --yolo-model yolov8n       # bboxes only
                                        yolov8n-seg   # bboxes + segmentation masks
                                        yolov8n-pose  # bboxes + pose estimation

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

Detections and embeddings are stored for the selected YOLO and ReID model respectively, which then be loaded into any tracking algorithm. Avoiding the overhead of repeatedly generating this data.
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

The set of hyperparameters leading to the best HOTA result are written to the tracker's config file.

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

For Yolo tracking bugs and feature requests please visit [GitHub Issues](https://github.com/mikel-brostrom/yolo_tracking/issues).
For business inquiries or professional support requests please send an email to: box-mot@outlook.com
