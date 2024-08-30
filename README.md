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

|  Tracker | HOTA↑ | MOTA↑ | IDF1↑ |
| -------- | ----- | ----- | ----- |
| [BoTSORT](https://arxiv.org/pdf/2206.14651.pdf)    | 77.8 | 78.9 | 88.9 |
| [DeepOCSORT](https://arxiv.org/pdf/2302.11813.pdf) | 77.4 | 78.4 | 89.0 |
| [OCSORT](https://arxiv.org/pdf/2203.14360.pdf)     | 77.4 | 78.4 | 89.0 |
| [HybridSORT](https://arxiv.org/pdf/2308.00783.pdf) | 77.3 | 77.9 | 88.8 |
| [ByteTrack](https://arxiv.org/pdf/2110.06864.pdf)  | 75.6 | 74.6 | 86.0 |
| [StrongSORT](https://arxiv.org/pdf/2202.13514.pdf) |      | | |
| [ImprAssoc](https://ieeexplore.ieee.org/document/10223159)| | |
| <img width=200/>                                   | <img width=100/> | <img width=100/> | <img width=100/> |

<sub> NOTES: performed on the 10 first frames of each MOT17 sequence. The detector used is ByteTrack's YoloXm, trained on: CrowdHuman, MOT17, Cityperson and ETHZ. Each tracker is configured with its original parameters found in their respective official repository.</sub>

</div>

</details>

<details>
<summary>3rd party resources</summary>

* [ROS wrapper for boxmot (link to external repository)](https://github.com/KalanaRatnayake/boxmot_ros)&nbsp;
* [Yolov10 Integration with BoxMOT (link to external Notebook)](https://colab.research.google.com/drive/1-QV2TNfqaMsh14w5VxieEyanugVBG14V?usp=drive_link)&nbsp;
* [Yolov8 training (link to external repository)](https://docs.ultralytics.com/modes/train/)&nbsp;
* [ReID model training (link to external repository)](https://kaiyangzhou.github.io/deep-person-reid/user_guide.html)&nbsp;
* [ReID model inference acceleration with Nebullvm (link to external Notebook)](https://colab.research.google.com/drive/1APUZ1ijCiQFBR9xD0gUvFUOC8yOJIvHm?usp=sharing)&nbsp;

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

#### News

* Added YOLOv9 and YOLOv10 support
* Hyperparameter search for all trackers using RayTune (July 2024)
* Added Improved Association Pipeline Tracker by [rolson24](https://github.com/rolson24) in [#1527](https://github.com/mikel-brostrom/boxmot/pull/1527) (July 2024)


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
$ python3 tracking/val.py --benchmark MOT17-mini --yolo-model yolov8n.pt --reid-model osnet_x0_25_msmt17.pt --tracking-method deepocsort --verbose --source ./assert/MOT17-mini/train
$ python3 tracking/val.py --benchmark MOT17      --yolo-model yolov8n.pt --reid-model osnet_x0_25_msmt17.pt --tracking-method ocsort     --verbose --source ./tracking/val_utils/MOT17/train
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
$ python tracking/evolve.py --benchmark MOT17-mini --dets yolov8n --embs osnet_x0_25_msmt17 --n-trials 9 --tracking-method botsort
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

<details>
<summary>Detection</summary>

```python
import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT


tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    device='cuda:0',
    fp16=False,
)

vid = cv2.VideoCapture(0)

while True:
    ret, im = vid.read()

    # substitute by your object detector, output has to be N X (x, y, x, y, conf, cls)
    dets = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])

    # Check if there are any detections
    if dets.size > 0:
        tracker.update(dets, im) # --> M X (x, y, x, y, id, conf, cls, ind)
    # If no detections, make prediction ahead
    else:   
        dets = np.empty((0, 6))  # empty N X (x, y, x, y, conf, cls)
        tracker.update(dets, im) # --> M X (x, y, x, y, id, conf, cls, ind)
    tracker.plot_results(im, show_trajectories=True)

    # break on pressing q or space
    cv2.imshow('BoxMOT detection', im)     
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
```

</details>


<details>
<summary>Pose & segmentation</summary>

```python
import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT


tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    device='cuda:0',
    fp16=True,
)

vid = cv2.VideoCapture(0)

while True:
    ret, im = vid.read()

    keypoints = np.random.rand(2, 17, 3)
    mask = np.random.rand(2, 480, 640)
    # substitute by your object detector, input to tracker has to be N X (x, y, x, y, conf, cls)
    dets = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])

    tracks = tracker.update(dets, im) # --> M x (x, y, x, y, id, conf, cls, ind)

    # xyxys = tracks[:, 0:4].astype('int') # float64 to int
    # ids = tracks[:, 4].astype('int') # float64 to int
    # confs = tracks[:, 5]
    # clss = tracks[:, 6].astype('int') # float64 to int
    inds = tracks[:, 7].astype('int') # float64 to int

    # in case you have segmentations or poses alongside with your detections you can use
    # the ind variable in order to identify which track is associated to each seg or pose by:
    # masks = masks[inds]
    # keypoints = keypoints[inds]
    # such that you then can: zip(tracks, masks) or zip(tracks, keypoints)

    # break on pressing q or space
    cv2.imshow('BoxMOT segmentation | pose', im)     
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
```

</details>

<details>
<summary>Tiled inference</summary>
  
```py
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import numpy as np
from pathlib import Path
from boxmot import DeepOCSORT


tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    device='cpu',
    fp16=False,
)

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='yolov8n.pt',
    confidence_threshold=0.5,
    device="cpu",  # or 'cuda:0'
)

vid = cv2.VideoCapture(0)
color = (0, 0, 255)  # BGR
thickness = 2
fontscale = 0.5

while True:
    ret, im = vid.read()

    # get sliced predictions
    result = get_sliced_prediction(
        im,
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    num_predictions = len(result.object_prediction_list)
    dets = np.zeros([num_predictions, 6], dtype=np.float32)
    for ind, object_prediction in enumerate(result.object_prediction_list):
        dets[ind, :4] = np.array(object_prediction.bbox.to_xyxy(), dtype=np.float32)
        dets[ind, 4] = object_prediction.score.value
        dets[ind, 5] = object_prediction.category.id

    tracks = tracker.update(dets, im) # --> (x, y, x, y, id, conf, cls, ind)

    tracker.plot_results(im, show_trajectories=True)

    # break on pressing q or space
    cv2.imshow('BoxMOT tiled inference', im)     
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
```

</details>

## Contributors

<a href="https://github.com/mikel-brostrom/yolo_tracking/graphs/contributors ">
  <img src="https://contrib.rocks/image?repo=mikel-brostrom/yolo_tracking" />
</a>

## Contact

For Yolo tracking bugs and feature requests please visit [GitHub Issues](https://github.com/mikel-brostrom/yolo_tracking/issues).
For business inquiries or professional support requests please send an email to: box-mot@outlook.com
