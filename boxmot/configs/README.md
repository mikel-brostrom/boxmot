# Configs

This directory contains the YAML configs used by BoxMOT's config-driven
`generate`, `eval`, and `tune` flows.

The primary config split is:

- `modes.yaml`: shared defaults consumed by both the CLI and the high-level Python API
- `datasets/`: dataset and evaluation metadata
- `benchmarks/`: thin benchmark bundles that select a dataset plus the
  detector and ReID profiles associated with that benchmark
- `detectors/`: reusable detector profiles
- `reid/`: reusable ReID profiles
- `trackers/`: tracker hyperparameters and tuning ranges

Example:

```bash
boxmot eval --benchmark mot17-ablation --tracker boosttrack
```

In this layout:

- `boxmot/configs/modes.yaml` provides the shared defaults for `track`, `generate`, `eval`, `tune`, and `export`
- `--benchmark mot17-ablation` resolves `boxmot/configs/benchmarks/mot17-ablation.yaml`
- the benchmark config selects `boxmot/configs/datasets/mot17-ablation.yaml`
- the benchmark config selects its associated detector and ReID profiles
- `--tracker boosttrack` loads `boxmot/configs/trackers/boosttrack.yaml`
