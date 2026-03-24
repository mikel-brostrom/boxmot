# ReID Configs

This directory contains one YAML per ReID profile.

Each ReID config should define:

- `id`: ReID config identifier
- `model`: ReID weights path
- `url`: optional ReID download URL
- `device`: optional ReID runtime device override
- `half`: optional ReID half-precision override

Example:

```yaml
id: lmbn_n_duke
model: models/lmbn_n_duke.pt
url: https://github.com/mikel-brostrom/yolov8_tracking/releases/download/v9.0/lmbn_n_duke.pth
device: ""
half: false
```
