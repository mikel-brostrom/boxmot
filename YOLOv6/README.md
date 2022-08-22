# YOLOv6
## Introduction

YOLOv6 is a single-stage object detection framework dedicated to industrial applications, with hardware-friendly efficient design and high performance.

<img src="assets/picture.png" width="800">

YOLOv6-nano achieves 35.0 mAP on COCO val2017 dataset with 1242 FPS on T4 using TensorRT FP16 for bs32 inference, and YOLOv6-s achieves 43.1 mAP on COCO val2017 dataset with 520 FPS on T4 using TensorRT FP16 for bs32 inference.

YOLOv6 is composed of the following methods:

- Hardware-friendly Design for Backbone and Neck
- Efficient Decoupled Head with SIoU Loss


## Coming soon

- [ ] YOLOv6 m/l/x model.
- [ ] Deployment for MNN/TNN/NCNN/CoreML...
- [ ] Quantization tools


## Quick Start

### Install

```shell
git clone https://github.com/meituan/YOLOv6
cd YOLOv6
pip install -r requirements.txt
```

### Inference

First, download a pretrained model from the YOLOv6 [release](https://github.com/meituan/YOLOv6/releases/tag/0.1.0)

Second, run inference with `tools/infer.py`

```shell
python tools/infer.py --weights yolov6s.pt --source img.jpg / imgdir
                                yolov6n.pt
```

### Training

Single GPU

```shell
python tools/train.py --batch 32 --conf configs/yolov6s.py --data data/coco.yaml --device 0
                                        configs/yolov6n.py
```

Multi GPUs (DDP mode recommended)

```shell
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 256 --conf configs/yolov6s.py --data data/coco.yaml --device 0,1,2,3,4,5,6,7
                                                                                        configs/yolov6n.py
```

- conf: select config file to specify network/optimizer/hyperparameters
- data: prepare [COCO](http://cocodataset.org) dataset, [YOLO format coco labels](https://github.com/meituan/YOLOv6/releases/download/0.1.0/coco2017labels.zip) and specify dataset paths in data.yaml
- make sure your dataset structure as follows:
```
├── coco
│   ├── annotations
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── images
│   │   ├── train2017
│   │   └── val2017
│   ├── labels
│   │   ├── train2017
│   │   ├── val2017
│   ├── LICENSE
│   ├── README.txt
```


### Evaluation

Reproduce mAP on COCO val2017 dataset

```shell
python tools/eval.py --data data/coco.yaml --batch 32 --weights yolov6s.pt --task val
                                                                yolov6n.pt
```

### Resume
If your training process is corrupted, you can resume training by
```
# single GPU traning.
python tools/train.py --resume
# multi GPU training.
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --resume
```
Your can also specify a checkpoint path to `--resume` parameter by
```
# remember replace /path/to/your/checkpoint/path to the checkpoint path which you want to resume training.
--resume /path/to/your/checkpoint/path

```

### Deployment

*  [ONNX](./deploy/ONNX)
*  [OpenCV Python/C++](./deploy/ONNX/OpenCV)
*  [OpenVINO](./deploy/OpenVINO)
*  [Partial Quantization](./tools/partial_quantization)
*  [TensorRT](./deploy/TensorRT)

### Tutorials

*  [Train custom data](./docs/Train_custom_data.md)
*  [Test speed](./docs/Test_speed.md)
*  [Tutorial of RepOpt for YOLOv6](./docs/tutorial_repopt.md)

## Benchmark


| Model           | Size        | mAP<sup>val<br/>0.5:0.95 | Speed<sup>V100<br/>fp16 b32 <br/>(ms) | Speed<sup>V100<br/>fp32 b32 <br/>(ms) | Speed<sup>T4<br/>trt fp16 b1 <br/>(fps) | Speed<sup>T4<br/>trt fp16 b32 <br/>(fps) | Params<br/><sup> (M) | FLOPs<br/><sup> (G) |
| :-------------- | ----------- | :----------------------- | :------------------------------------ | :------------------------------------ | ---------------------------------------- | ----------------------------------------- | --------------- | -------------- |
| [**YOLOv6-n**](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6n.pt)    | 416<br/>640 | 30.8<br/>35.0            | 0.3<br/>0.5                           | 0.4<br/>0.7                           | 1100<br/>788                             | 2716<br/>1242                             | 4.3<br/>4.3     | 4.7<br/>11.1   |
| [**YOLOv6-tiny**](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6t.pt) | 640         | 41.3                     | 0.9                                   | 1.5                                   | 425                                      | 602                                       | 15.0            | 36.7           |
| [**YOLOv6-s**](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.pt)    | 640         | 43.1                     | 1.0                                   | 1.7                                   | 373                                      | 520                                       | 17.2            | 44.2           |


- Comparisons of the mAP and speed of different object detectors are tested on [COCO val2017](https://cocodataset.org/#download) dataset.
- Refer to [Test speed](./docs/Test_speed.md) tutorial to reproduce the speed results of YOLOv6.
- Params and FLOPs of YOLOv6 are estimated on deployed model.
- Speed results of other methods are tested in our environment using official codebase and model if not found from the corresponding official release.

 ## Third-party resources
 * YOLOv6 NCNN Android app demo: [ncnn-android-yolov6](https://github.com/FeiGeChuanShu/ncnn-android-yolov6) from [FeiGeChuanShu](https://github.com/FeiGeChuanShu)
 * YOLOv6 ONNXRuntime/MNN/TNN C++: [YOLOv6-ORT](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/ort/cv/yolov6.cpp), [YOLOv6-MNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/mnn/cv/mnn_yolov6.cpp) and [YOLOv6-TNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/tnn/cv/tnn_yolov6.cpp) from [DefTruth](https://github.com/DefTruth)
 * YOLOv6 TensorRT Python: [yolov6-tensorrt-python](https://github.com/Linaom1214/tensorrt-python/blob/main/yolov6/trt.py) from [Linaom1214](https://github.com/Linaom1214)
 * YOLOv6 TensorRT Windows C++: [yolort](https://github.com/zhiqwang/yolov5-rt-stack/tree/main/deployment/tensorrt-yolov6) from [Wei Zeng](https://github.com/Wulingtian)
 * YOLOv6 Quantization and Auto Compression Example [YOLOv6-ACT](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/pytorch_yolov6) from [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)
 * [YOLOv6 web demo](https://huggingface.co/spaces/nateraw/yolov6) on [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/nateraw/yolov6)
 * Tutorial: [How to train YOLOv6 on a custom dataset](https://blog.roboflow.com/how-to-train-yolov6-on-a-custom-dataset/) <a href="https://colab.research.google.com/drive/1YnbqOinBZV-c9I7fk_UL6acgnnmkXDMM"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
 * Demo of YOLOv6 inference on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahdilamb/YOLOv6/blob/main/inference.ipynb)
