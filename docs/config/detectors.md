# Detectors

Detector configs live under `boxmot/configs/detectors` and define
reusable models independently of datasets and experiments.

## Select a model

Live tracking accepts a built-in profile, an official Ultralytics model name,
or a local checkpoint. A profile with several checkpoints requires an explicit
`profile/checkpoint` selector:

```bash
boxmot track --source video.mp4 --detector yolo26n --tracker bytetrack
boxmot track --source video.mp4 --detector yolox-x-mot17/ablation --tracker bytetrack
boxmot track --source video.mp4 --detector ./models/custom.pt --tracker bytetrack
```

The registered backends are `ultralytics`, `yolox`, and `rtdetr`. They cover
Ultralytics box/OBB/segmentation checkpoints, YOLOX, and Hugging Face RT-DETR
snapshots respectively. Ultralytics `rtdetr-l.pt` and `rtdetr-x.pt` use the
Ultralytics backend; bare `rtdetr_v2_*` selectors resolve a `PekingU` Hub
snapshot. Install the matching [optional runtime](../getting-started/installation.md)
before constructing a model.

## Author a profile

This example is the bundled MOT17 profile; the class IDs describe the model's
output labels, before the experiment's class mapping.

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
    uri: https://huggingface.co/Lekim89/yolox/resolve/main/yolox_x_MOT17_ablation.pt
  test:
    path: models/yolox_x_MOT17_test.pt
    uri: https://huggingface.co/Lekim89/yolox/resolve/main/yolox_x_MOT17_test.pt
```

Experiments select checkpoints explicitly. A dataset split never implicitly
chooses a detector checkpoint.

`path` is required for each checkpoint. `uri` is optional for an existing local
file; use it to acquire missing weights. An optional lowercase SHA-256 digest
in `sha256` verifies checkpoint content. A local profile resolves relative
artifact paths beside its YAML; built-in profile paths are relative to the
working directory (for example `./models/yolo26n.pt`).

The resolver requires exactly two positive `image_size` values in height-width
order, a confidence threshold in `[0, 1]`, and matching detector/dataset box
types.

Experiments resolve their detector profile before computing a build ID.
Materialization accepts only an experiment and has no detector or geometry
override; create another experiment to select a different detector checkpoint.
Python callers construct a `DetectorSpec` with the resolved artifact path,
SHA-256, preprocessing, precision, geometry, and normalized backend options.

For ordinary Python use, let the factory resolve the profile and its artifact:

```python
from boxmot.detectors import create_detector

detector = create_detector("yolo26n", device="cpu", precision="fp32")
```

Set `allow_download=False` to require local artifacts. Factory construction
resolves and hashes the weights; it is not a metadata-only operation.

## Detection masks and segmentors

A segmentation-capable detector can return masks alongside its boxes. A
separate segmentor refines an existing detection batch and preserves its row
order; it does not choose new tracker identities. The registered segmentors
are `sam`, `maskrcnn`, and `edgetam`, selected through an explicit checkpoint
specification. Add one to an experiment when its detector does not supply the
masks you need, then materialize with `--publish-masks`.

See [instance masks](../tasks/masks.md) for complete segmentor configurations
and [the detector API](../python/low-level.md#detection) for canonical `Frame` and
`Detections` usage.
