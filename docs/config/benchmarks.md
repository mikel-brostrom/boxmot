# Benchmarks

Benchmark configs live under `boxmot/configs/benchmarks`.

## Role

Each benchmark YAML is a self-contained benchmark definition that includes:

- dataset path, split, and class definitions
- default detector profile
- default ReID profile
- optional cache download URLs

Built-in examples include `mot17`, `sportsmot`, `mmot`, and `mmot-mini`.

Downloaded MOT-style benchmark data is stored under `boxmot/datasets/mot/`.
Downloaded ReID datasets should use `boxmot/datasets/reid/`.

## Example

```yaml
id: mot17

dataset:
  id: mot17
  root: "boxmot/datasets/mot/MOT17"
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

# Public detections for --detection-source
public_detectors:
  frcnn:
    id: mot17_public_frcnn
    parquet: "data/detections/frcnn"
    label: "FRCNN (Faster R-CNN public detections)"
  sdp:
    id: mot17_public_sdp
    parquet: "data/detections/sdp"
    label: "SDP (Deformable Parts Model v5)"
  dpm:
    id: mot17_public_dpm
    parquet: "data/detections/dpm"
    label: "DPM (Deformable Parts Model)"

download:
  source: parquet
  parquet_repo: "Lekim89/mot17-parquet"
  public_detector: FRCNN
  runs:
    ablation:
      url: "hf://Lekim89/runs/runs/dets_n_embs/mot17/ablation"
      detector: yolox_x_mot17_ablation
      reid: lmbn_n_duke
```

## Use from the CLI

```bash
# Use default split from config
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack

# Override split
boxmot eval --benchmark sportsmot --split test --tracker boosttrack

# Use public detections instead of the configured detector
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --detection-source frcnn

# MMOT benchmark config (OBB-backed)
boxmot eval --benchmark mmot --split test --tracker botsort

# MMOT mini benchmark config rooted at assets/mmot-mini
boxmot eval --benchmark mmot-mini --split train --tracker botsort
```

That benchmark name selects the corresponding YAML and all linked profiles.
The `--split` flag overrides the default split defined in the config.

## Public detectors block

The optional `public_detectors` section maps `--detection-source` values to parquet paths in the benchmark's HuggingFace repository. Each entry has:

- `id` – cache key for the public detection bucket
- `parquet` – relative path within the parquet repo to the detection files
- `label` – human-readable description

When `--detection-source public` is used, the default is resolved from `download.public_detector`.
