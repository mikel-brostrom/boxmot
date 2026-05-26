# Benchmarks

Benchmark configs live under `boxmot/configs/datasets`.

## Role

Each dataset YAML is a self-contained benchmark definition that includes:

- dataset path, split, and class definitions
- default detector profile
- default ReID profile
- optional cache download URLs

## Example

```yaml
id: mot17

path: "boxmot/engine/eval/trackeval/data/MOT17-mini"
split: "train"
train: "train"

layout: mot
box_type: aabb

detector: yolox_x_mot17_ablation
reid: lmbn_n_duke

names:
  1: pedestrian
```

## Use from the CLI

```bash
# Use default split from config
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack

# Override split
boxmot eval --benchmark sportsmot --split test --tracker boosttrack
```

That benchmark name selects the corresponding YAML and all linked profiles.
The `--split` flag overrides the default split defined in the config.
