# Detector Configs

This directory contains one YAML per detector profile.

Each detector config should define:

- `id`: detector config identifier
- `model`: detector weights path
- `url`: optional detector download URL
- `imgsz`: default image size
- `conf`: default detection confidence
- `classes`: detector class taxonomy

Example:

```yaml
id: yolox_x_mot17_ablation
model: models/yolox_x_MOT17_ablation.pt
url: https://drive.google.com/uc?id=1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob
imgsz: [800, 1440]
conf: 0.01
classes:
  0: person
```
