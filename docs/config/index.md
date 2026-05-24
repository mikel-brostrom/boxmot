# Config System Overview

BoxMOT uses YAML configuration files to keep benchmark workflows repeatable across the CLI and Python API.

## Config families

- `modes.yaml` for shared defaults across `track`, `generate`, `eval`, `tune`, `research`, and `export`
- `benchmarks/` for thin benchmark bundles
- `datasets/` for dataset metadata and evaluation layout
- `detectors/` for detector profiles and family defaults
- `reid/` for ReID profiles
- `trackers/` for tracker runtime defaults and tuning ranges

## Why it matters

This layout lets commands such as:

```bash
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack
```

resolve detector, ReID, and dataset defaults without forcing you to repeat the same paths for every run.

## Related pages

- [Mode Defaults](modes.md)
- [Benchmarks](benchmarks.md)
- [Datasets](datasets.md)
- [Detectors](detectors.md)
- [ReID Profiles](reid.md)
- [Tracker YAMLs](trackers.md)
