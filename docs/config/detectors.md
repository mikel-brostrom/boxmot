# Detectors

Detector settings are defined inline in benchmark YAMLs under `boxmot/configs/benchmarks`.

## Organization

Each benchmark includes a detector block, for example:

```text
detector:
  id: yolox_x_mot17_ablation
  model: models/yolox_x_MOT17_ablation.pt
  url: https://...
  imgsz: [800, 1440]
  conf: 0.01
  classes:
    0: person
```

## Resolution order

1. benchmark-selected detector block for the active benchmark
2. model filename lookup across benchmark detector blocks

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

Use detector defaults through `--benchmark`, or override with `--detector`.
