# Real-time multi-object, segmentation and pose tracking using Yolov8 | Yolo-NAS | YOLOX with DeepOCSORT and LightMBN


<div align="center">
  <p>
  <img src="boxmot/strongsort/results/track_all_seg_1280_025conf.gif" width="400"/>
  </p>
  <br>
  <div>
  <a href="https://github.com/mikel-brostrom/yolov8_tracking/actions/workflows/ci.yml"><img src="https://github.com/mikel-brostrom/yolov8_tracking/actions/workflows/ci.yml/badge.svg" alt="CI CPU testing"></a>
  <a href="https://pepy.tech/project/boxmot"><img src="https://static.pepy.tech/personalized-badge/boxmot?period=month&units=international_system&left_color=grey&right_color=orange&left_text=Downloads"></a>
  <br>  
  <a href="https://colab.research.google.com/drive/18nIqkBr68TkK8dHdarxTco6svHUJGggY?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
<a href="https://doi.org/10.5281/zenodo.7629840"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7629840.svg" alt="DOI"></a>
  </div>
</div>


## Introduction

This repo contains a collections of state-of-the-art multi-object trackers. Some of them are based on motion only, others on motion + appearance description. For the latter, state-of-the-art ReID model are downloaded automatically as well. Supported ones at the moment are: [DeepOCSORT](https://arxiv.org/abs/2302.11813) [LightMBN](https://arxiv.org/pdf/2101.10774.pdf), [BoTSORT](https://arxiv.org/abs/2206.14651) [LightMBN](https://github.com/jixunbo/LightMBN)[](https://arxiv.org/pdf/2101.10774.pdf), [StrongSORT](https://github.com/dyhBUPT/StrongSORT)[](https://arxiv.org/abs/2202.13514) [LightMBN](https://github.com/jixunbo/LightMBN)[](https://arxiv.org/pdf/2101.10774.pdf), [OCSORT](https://github.com/noahcao/OC_SORT)[](https://arxiv.org/abs/2203.14360) and [ByteTrack](https://github.com/ifzhang/ByteTrack)[](https://arxiv.org/abs/2110.06864).

We provide examples on how to use this package together with popular object detection models. Right now [Yolov8](https://github.com/ultralytics), [Yolo-NAS](https://github.com/Deci-AI/super-gradients) and YOLOX are available.

<details>
<summary>Tutorials</summary>

* [Yolov8 training (link to external repository)](https://docs.ultralytics.com/modes/train/)&nbsp;
* [Deep appearance descriptor training (link to external repository)](https://kaiyangzhou.github.io/deep-person-reid/user_guide.html)&nbsp;
* [ReID model export to ONNX, OpenVINO, TensorRT and TorchScript](https://github.com/mikel-brostrom/yolov8_tracking/wiki/ReID-multi-framework-model-export)&nbsp;
* [Evaluation on custom tracking dataset](https://github.com/mikel-brostrom/yolov8_tracking/wiki/How-to-evaluate-on-custom-tracking-dataset)&nbsp;
* [ReID inference acceleration with Nebullvm](https://colab.research.google.com/drive/1APUZ1ijCiQFBR9xD0gUvFUOC8yOJIvHm?usp=sharing)&nbsp;
  
  </details>
  
<details>
<summary>Experiments</summary>

In inverse chronological order:

* [Evaluation of the params evolved for first half of MOT17 on the complete MOT17](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/wiki/Evaluation-of-the-params-evolved-for-first-half-of-MOT17-on-the-complete-MOT17)

* [Segmentation model vs object detetion model on MOT metrics](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/wiki/Segmentation-model-vs-object-detetion-model-on-MOT-metrics)
  
* [Effect of masking objects before feature extraction](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/wiki/Masked-detection-crops-vs-regular-detection-crops-for-ReID-feature-extraction)
  
* [conf-thres vs HOTA, MOTA and IDF1](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/wiki/conf-thres-vs-MOT-metrics)

* [Effect of KF updates ahead for tracks with no associations on MOT17](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/wiki/Effect-of-KF-updates-ahead-for-tracks-with-no-associations,-on-MOT17)

* [Effect of full images vs 1280 input to StrongSORT on MOT17](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/wiki/Effect-of-passing-full-image-input-vs-1280-re-scaled-to-StrongSORT-on-MOT17)

* [Effect of different OSNet architectures on MOT16](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/wiki/OSNet-architecture-performances-on-MOT16)

* [Yolov5 StrongSORT vs BoTSORT vs OCSORT](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/wiki/StrongSORT-vs-BoTSORT-vs-OCSORT)
    * Yolov5 [BoTSORT](https://arxiv.org/abs/2206.14651) branch: https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/tree/botsort

* [Yolov5 StrongSORT OSNet vs other trackers MOT17](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/wiki/MOT-17-evaluation-(private-detector))&nbsp;

* [StrongSORT MOT16 ablation study](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/wiki/Yolov5DeepSORTwithOSNet-vs-Yolov5StrongSORTwithOSNet-ablation-study-on-MOT16)&nbsp;

* [Yolov5 StrongSORT OSNet vs other trackers MOT16 (deprecated)](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/wiki/MOT-16-evaluation)&nbsp;

  </details>

## Why using this tracking toolbox?

Everything is designed with simplicity and flexibility in mind. We don't hyperfocus on results on a single dataset, we prioritize real-world results. If you don't get good tracking results on your custom dataset with the out-of-the-box tracker configurations, use the `examples/evolve.py` script for tracker hyperparameter tuning.

## Installation

Start with [**Python>=3.8**](https://www.python.org/) environment.

If you want to run the YOLOv8, YOLO-NAS or YOLOX examples:

```
git clone https://github.com/mikel-brostrom/yolo_tracking.git
pip install -v -e .
```

but if you only want to import the tracking modules you can simply:

```
pip install boxmot
```


## YOLOv8 | YOLO-NAS | YOLOX examples

<details>
<summary>Tracking</summary>

<details>
<summary>Yolo models</summary>
  


```bash
$ python examples/track.py --yolo-model yolov8n       # bboxes only
  python examples/track.py --yolo-model yolo_nas_s    # bboxes only
  python examples/track.py --yolo-model yolox_n       # bboxes only
                                        yolov8n-seg   # bboxes + segmentation masks
                                        yolov8n-pose  # bboxes + pose estimation

```
  
  </details>

<details>
<summary>Tracking methods</summary>

```bash
$ python examples/track.py --tracking-method deepocsort
                                             strongsort
                                             ocsort
                                             bytetrack
                                             botsort
```
  
</details>

<details>
<summary>Tracking sources</summary>

Tracking can be run on most video formats

```bash
$ python examples/track.py --source 0                               # webcam
                                    img.jpg                         # image
                                    vid.mp4                         # video
                                    path/                           # directory
                                    path/*.jpg                      # glob
                                    'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                    'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Select Yolov8 model</summary>

There is a clear trade-off between model inference speed and overall performance. In order to make it possible to fulfill your inference speed/accuracy needs you can select a Yolov5 family model for automatic download. These model can be further optimized for you needs by the [export.py](https://github.com/ultralytics/yolov5/blob/master/export.py) script

```bash
$ python examples/track.py --source 0 --yolo-model yolov8n.pt --img 640
                                          yolov8s.tflite
                                          yolov8m.pt
                                          yolov8l.onnx 
                                          yolov8x.pt --img 1280
                                          ...
```
  
</details>

<details>
<summary>Select ReID model</summary>

Some tracking methods combine appearance description and motion in the process of tracking. For those which use appearance, you can choose a ReID model based on your needs from this [ReID model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO). These model can be further optimized for you needs by the [reid_export.py](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/blob/master/reid_export.py) script

```bash
$ python examples/track.py --source 0 --reid-model lmbn_n_cuhk03_d.pt
                                                   osnet_x0_25_market1501.pt
                                                   mobilenetv2_x1_4_msmt17.engine
                                                   resnet50_msmt17.onnx
                                                   osnet_x1_0_msmt17.pt
                                                   ...
```

</details>
  
<details>
<summary>Filter tracked classes</summary>

By default the tracker tracks all MS COCO classes.

If you want to track a subset of the classes that you model predicts, add their corresponding index after the classes flag,

```bash
python examples/track.py --source 0 --yolo-model yolov8s.pt --classes 16 17  # COCO yolov8 model. Track cats and dogs, only
```

[Here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) is a list of all the possible objects that a Yolov8 model trained on MS COCO can detect. Notice that the indexing for the classes in this repo starts at zero

</details>

<details>
<summary>MOT compliant results</summary>
  
Can be saved to your experiment folder `runs/track/<yolo_model>_<deep_sort_model>/` by 

```bash
python examples/track.py --source ... --save-txt
```

</details>

</details>
  
<details>
<summary>Evaluation</summary>

Evaluate a combination of detector, tracking method and ReID model on standard MOT dataset or you custom one by

```bash
$ python3 examples/val.py --yolo-model yolo_nas_s.pt --reid-model osnetx1_0_dukemtcereid.pt --tracking-method deepocsort --benchmark MOT16
                          --yolo-model yolox_n.pt    --reid-model osnet_ain_x1_0_msmt17.pt  --tracking-method ocsort     --benchmark MOT17 
                          --yolo-model yolov8s.pt    --reid-model lmbn_n_market.pt          --tracking-method strongsort --benchmark <your-custom-dataset>
```
  
</details>
  
<details>
<summary>Evolution</summary>

We use a fast and elitist multiobjective genetic algorithm for tracker hyperparameter tuning. By default the objectives are: HOTA, MOTA, IDF1. Run it by

```bash
$ python examples/evolve.py --tracking-method strongsort --benchmark MOT17 --n-trials 100  # tune strongsort for MOT17
                            --tracking-method ocsort     --benchmark <your-custom-dataset> --objective HOTA # tune ocsort for maximizing HOTA on your custom tracking dataset
```

The set of hyperparameters leading to the best HOTA result are written to the tracker's config file.
  
</details>


## Custom object detection model example
  
<details>
<summary>Click to exapand!</summary>

```python
from boxmot import DeepOCSORT
from pathlib import Path


tracker = DeepOCSORT(
  model_weights=Path('osnet_x0_25_msmt17.pt'),  # which ReID model to use
  device='cuda:0',  # 'cpu', 'cuda:0', 'cuda:1', ... 'cuda:N'
  fp16=True,  # wether to run the ReID model with half precision or not
)
  
cap = cv.VideoCapture(0)
while True:
    ret, im = cap.read()
    ...
    # dets (numpy.ndarray): 
    #  - your model's nms:ed outputs of shape Nx6 (x, y, x, y, conf, cls)
    # im   (numpy.ndarray):
    #  - the original hxwx3 image (for better ReID results)
    #  - the downscaled hxwx3 image fed to you model (faster)
    tracker_outputs = tracker.update(dets, im)  # --> (x, y, x, y, id, conf, cls)
    ...
```
  
</details>
  

## Contact 

For Yolov8 tracking bugs and feature requests please visit [GitHub Issues](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/issues). 
For business inquiries or professional support requests please send an email to: yolov5.deepsort.pytorch@gmail.com
