# Detectors

Detector configs live under `boxmot/configs/detectors` and define
reusable models independently of datasets and experiments.

```yaml
id: yolox-x-mot17

box_type: aabb

classes:
  0: person

inference:
  image_size: [800, 1440]
  confidence_threshold: 0.01

checkpoints:
  ablation:
    path: models/yolox_x_MOT17_ablation.pt
    uri: https://...
  test:
    path: models/yolox_x_MOT17_test.pt
    uri: https://...
```

Experiments select checkpoints explicitly. A dataset split never implicitly
chooses a detector checkpoint.

The resolver requires exactly two positive `image_size` values in height-width
order, a confidence threshold in `[0, 1]`, and matching detector/dataset box
types.

Detector profiles belong to experiment resolution. In direct-source and
model-free `--dataset` workflows, the CLI's `--detector` option is a detector
weight path or model identifier such as `yolov8n`; it is not a profile selector.
