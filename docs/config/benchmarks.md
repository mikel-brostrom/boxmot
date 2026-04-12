# Benchmarks

Benchmark configs live under `boxmot/configs/benchmarks`.

## Role

A benchmark YAML ties together:

- one dataset config
- one detector profile
- one ReID profile
- optional cache download URLs

## Example

```yaml
id: mot17-ablation
dataset: mot17-ablation
detector: yolox_x_mot17_ablation
reid: lmbn_n_duke
```

## Use from the CLI

```bash
boxmot eval --benchmark mot17-ablation --tracker boosttrack
```

That benchmark name selects the corresponding YAML and all linked profiles.
