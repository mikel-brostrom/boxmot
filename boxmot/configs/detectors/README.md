# Detector Configs

Detector configs are organized by family:

```
detectors/
  ultralytics/
    default.yaml          # COCO-80, shared by all standard Ultralytics models
    yolo11s_obb.yaml      # custom OBB model
    yolo11l_3ch.yaml
  yolox/
    yolox_x_visdrone.yaml
    yolox_x_mot17_ablation.yaml
    ...
```

## Resolution order

1. **Exact match** — recursive search across all subfolders for a YAML whose
   `model` (or `default_model`) field matches the model filename. If two files
   match the same model, an error is raised.
2. **Family default** — if no exact match, the model family is inferred from
   the name (e.g. `yolov8m` → `ultralytics`) and the corresponding
   `<family>/default.yaml` is used.

## Config schema

```yaml
id: yolox_x_mot17_ablation
model: models/yolox_x_MOT17_ablation.pt   # used for exact matching
url: https://...                           # optional download URL
imgsz: [800, 1440]
conf: 0.01
box_type: aabb                             # aabb (default) or obb
classes:
  0: person
```

The `model` field is optional in family defaults (e.g. `ultralytics/default.yaml`).
