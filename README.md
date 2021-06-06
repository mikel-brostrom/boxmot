# Yolov5 + Deep Sort with PyTorch

![](Town.gif)

## Introduction

This repository contains a moded version of PyTorch YOLOv5 (https://github.com/ultralytics/yolov5). It filters out every detection that is not a person. The detections of persons are then passed to a Deep Sort algorithm (https://github.com/ZQPei/deep_sort_pytorch) which tracks the persons. The reason behind the fact that it just tracks persons is that the deep association metric is trained on a person ONLY datatset.

## Description

The implementation is based on two papers:

- Simple Online and Realtime Tracking with a Deep Association Metric
https://arxiv.org/abs/1703.07402
- YOLOv4: Optimal Speed and Accuracy of Object Detection
https://arxiv.org/pdf/2004.10934.pdf

## Tutorials

* [Yolov5 training on Custom Data (link to external repository)](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp;
* [Deep Sort deep descriptor training (link to external repository)](https://github.com/ZQPei/deep_sort_pytorch#training-the-re-id-model)&nbsp;
* [Yolov5 deep_sort pytorch evaluation](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/wiki/Evaluation)&nbsp;



## Requirements

Python 3.8 or later with all requirements.txt dependencies installed, including torch>=1.7. To install run:

`pip install -U -r requirements.txt`

All dependencies are included in the associated docker images. Docker requirements are: 
- `nvidia-docker`
- Nvidia Driver Version >= 440.44

## Before you run the tracker

1. Clone the repository recursively:

`git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git`

If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

2. Github block pushes of files larger than 100 MB (https://help.github.com/en/github/managing-large-files/conditions-for-large-files). Hence you need to download two different weights: the ones for yolo and the ones for deep sort

- download the yolov5 weight from the latest realease https://github.com/ultralytics/yolov5/releases. Place the downlaoded `.pt` file under `yolov5/weights/`
- download the deep sort weights from https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6. Place ckpt.t7 file under`deep_sort/deep/checkpoint/`

## Tracking sources

Tracking can be run on most video formats

```bash
python3 track.py --source ...
```

- Video:  `--source file.mp4`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`


## Filter tracked classes

By default the tracker tracks class 0, this is, persons. In this case I recommend you to get [these weights](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing)

```python
parser.add_argument('--classes', nargs='+', default=[0], type=int, help='filter by class') # tracks persons only
```

If you want to track all 80 MS COCO classes just delete the default value

```python
parser.add_argument('--classes', nargs='+', type=int, help='filter by class') # tracks all MS COCO classes
```

If you want to select only a few classes to track add their associated integer to default

```python
parser.add_argument('--classes', nargs='+', default=[16, 17], type=int, help='filter by class') # tracks cats and dogs, only
```

## MOT compliant results

Can be saved to `inference/output` by 

```bash
python3 track.py --source ... --save-txt
```

## Other information

For more detailed information about the algorithms and their corresponding lisences used in this project access their official github implementations.

