# Configs

This directory contains the YAML configs used by BoxMOT's config-driven
`generate`, `eval`, and `tune` flows.

The primary config split is:

- `modes.yaml`: shared defaults consumed by both the CLI and the high-level Python API
- `datasets/`: dataset definitions including default detector and ReID profiles
- `detectors/`: reusable detector profiles (model weights, imgsz, conf)
- `reid/`: reusable ReID profiles (model weights, device, preprocessing)
- `trackers/`: tracker hyperparameters and tuning ranges

Example:

```bash
boxmot eval --benchmark mot17-ablation --tracker boosttrack
```

In this layout:

- `boxmot/configs/modes.yaml` provides the shared defaults for `track`, `generate`, `eval`, `tune`, and `export`
- `--benchmark mot17-ablation` resolves `boxmot/configs/datasets/mot17-ablation.yaml`
- the dataset config includes its default detector and ReID profiles
- `--tracker boosttrack` loads `boxmot/configs/trackers/boosttrack.yaml`
- `--split test` overrides the default split defined in the dataset config
