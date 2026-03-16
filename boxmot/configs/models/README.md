# Model Configs

This directory contains detector and ReID defaults for dataset-driven BoxMOT
runs.

A model config should only describe runtime model settings:

- `id`: model config identifier
- `detector.model`: detector weights path
- `detector.url`: optional detector download URL
- `detector.imgsz`: default image size
- `detector.conf`: default detector confidence threshold
- `detector.classes`: detector class taxonomy
- `reid.model`: ReID weights path
- `reid.url`: optional ReID download URL

Example:

```yaml
id: mot17-ablation-models

detector:
  model: models/yolox_x_MOT17_ablation.pt
  url: https://drive.google.com/uc?id=1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob
  imgsz: [800, 1440]
  conf: 0.01
  classes:
    0: person

reid:
  model: models/lmbn_n_duke.pt
```

Use a model config explicitly:

```bash
boxmot eval --data mot17-ablation --models mot17-ablation-models --tracker boosttrack
```

A dataset config can also point to its default model config through the
`models:` field.

Tracker hyperparameters live separately under `boxmot/configs/trackers/`.
