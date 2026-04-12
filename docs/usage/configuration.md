# Configuration

BoxMOT uses YAML configs to make benchmark-driven workflows repeatable across both the CLI and Python API.

## Config families

| Config family | Purpose |
| --- | --- |
| `modes.yaml` | Shared defaults for `track`, `generate`, `eval`, `tune`, `research`, and `export` |
| `benchmarks/` | Thin workflow bundles that choose dataset, detector, and ReID defaults |
| `datasets/` | Dataset metadata and evaluation layout |
| `detectors/` | Detector profiles and family defaults |
| `reid/` | ReID model profiles |
| `trackers/` | Tracker runtime defaults and tuning spaces |

## Resolution flow

When you run:

```bash
boxmot eval --benchmark mot17-ablation --tracker boosttrack
```

BoxMOT resolves the benchmark config first, then loads the associated dataset, detector, ReID, and tracker configs automatically.

## Why it matters

This setup lets you:

- avoid repeating long dataset and model paths
- reuse caches across `generate`, `eval`, `tune`, and `research`
- keep experiment defaults in version-controlled YAML instead of shell history

## Detailed config pages

- [Config System Overview](../config/index.md)
- [Mode Defaults](../config/modes.md)
- [Benchmarks](../config/benchmarks.md)
- [Datasets](../config/datasets.md)
- [Detectors](../config/detectors.md)
- [ReID Profiles](../config/reid.md)
- [Tracker YAMLs](../config/trackers.md)
