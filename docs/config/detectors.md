# Detectors

Detector configs live under `boxmot/configs/detectors`.

## Organization

Detector profiles are grouped by family, for example:

```text
detectors/
  ultralytics/
    default.yaml
    yolo11s_obb.yaml
  yolox/
    yolox_x_mot17_ablation.yaml
```

## Resolution order

1. exact match by model filename against a YAML `model` field
2. family default such as `ultralytics/default.yaml`

## Typical fields

```yaml
id: yolox_x_mot17_ablation
model: models/yolox_x_MOT17_ablation.pt
url: https://...
imgsz: [800, 1440]
conf: 0.01
box_type: aabb
classes:
  0: person
```

Use detector configs directly through `--detector` or indirectly through `--benchmark`.
