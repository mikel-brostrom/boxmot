# Configs

This directory contains the YAML configs used by BoxMOT's config-driven
`track`, `generate`, `eval`, and `tune` flows.

The config split is:

- `datasets/`: dataset and evaluation metadata
- `models/`: detector and ReID defaults
- `trackers/`: tracker hyperparameters and tuning ranges

Use them together like this:

```bash
boxmot eval --data mot17-ablation --models mot17-ablation-models --tracker boosttrack
```

You can also rely on the dataset's default model config:

```bash
boxmot eval --data mot17-ablation --tracker boosttrack
```

Resolution order is:

- explicit CLI arguments
- selected model config
- selected dataset config
- built-in runtime defaults

Tracker configs are selected by tracker name. For example, `--tracker boosttrack`
loads `boxmot/configs/trackers/boosttrack.yaml`.
