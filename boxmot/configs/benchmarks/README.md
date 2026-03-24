# Benchmark Configs

This directory contains benchmark-specific YAMLs for BoxMOT's config-driven
`generate`, `eval`, and `tune` workflows.

Each benchmark config should define:

- the dataset config ID associated with the benchmark
- the detector config ID associated with the benchmark
- the ReID config ID associated with the benchmark
- optional cached-runs download URLs

Example:

```yaml
id: mot17-ablation
dataset: mot17-ablation
detector: yolox_x_mot17_ablation
reid: lmbn_n_duke
```
