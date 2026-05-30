# Configs

This directory contains the YAML configs used by BoxMOT's config-driven
`generate`, `eval`, and `tune` flows.

The primary config split is:

- `modes.yaml`: shared defaults consumed by both the CLI and the high-level Python API
- `benchmarks/`: self-contained benchmark bundles (dataset + detector + ReID + download config)
- `trackers/`: tracker hyperparameters and tuning ranges

Example:

```bash
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack
```

In this layout:

- `boxmot/configs/modes.yaml` provides the shared defaults for `track`, `generate`, `eval`, `tune`, and `export`
- `--benchmark mot17 --split ablation` resolves `boxmot/configs/benchmarks/mot17.yaml`
- the benchmark config contains dataset, detector, and ReID settings in one file
- `--tracker boosttrack` loads `boxmot/configs/trackers/boosttrack.yaml`
- `--split test` overrides the default split defined in the dataset config
