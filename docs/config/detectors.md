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

Experiments resolve their detector profile before computing a build ID.
Materialization accepts only an experiment and has no detector or geometry
override; create another experiment to select a different detector checkpoint.
Python callers construct a `DetectorSpec` with the resolved artifact path,
SHA-256, preprocessing, precision, geometry, and normalized backend options.
