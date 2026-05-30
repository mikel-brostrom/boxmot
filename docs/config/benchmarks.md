# Benchmarks

Benchmark configs live under `boxmot/configs/benchmarks`.

## Role

Each benchmark YAML is a self-contained benchmark definition that includes:

- dataset path, split, and class definitions
- default detector profile
- default ReID profile
- optional cache download URLs

Built-in examples include `mot17`, `sportsmot`, and `mmot-obb`.

## Example

```yaml
id: mot17

dataset:
  id: mot17
  root: "boxmot/engine/eval/trackeval/data/MOT17"
  splits:
    train: "train"
    val: "val"
    test: "test"
    ablation: "ablation"
  default_split: ablation

detector:
  id: yolox_x_mot17_ablation
  model: "models/yolox_x_MOT17_ablation.pt"

reid:
  id: lmbn_n_duke
  model: "models/lmbn_n_duke.pt"

download:
  dataset: "hf://Lekim89/MOT17"
  runs:
    ablation: "https://github.com/mikel-brostrom/boxmot/releases/download/v18.0.0/runs.zip"
```

## Use from the CLI

```bash
# Use default split from config
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack

# Override split
boxmot eval --benchmark sportsmot --split test --tracker boosttrack

# MMOT benchmark config (OBB-backed)
boxmot eval --benchmark mmot-obb --split test --tracker botsort
```

That benchmark name selects the corresponding YAML and all linked profiles.
The `--split` flag overrides the default split defined in the config.
