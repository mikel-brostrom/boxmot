# Evaluation and Postprocessing

Use this guide when you need to interpret benchmark outputs from `boxmot eval`, `BoxMOT.val(...)`, `tune`, or `research`.

For cache reuse, benchmark ids, and replay image-loading behavior, see [Benchmark Workflows](benchmarks.md).

## Core metrics

- `HOTA` for overall tracking quality
- `MOTA` for CLEAR-style summary quality
- `IDF1` for identity consistency
- `AssA` and `AssRe` for association quality
- `IDSW` and `IDs` for identity-switch context

## Where metrics appear

- `eval` reports benchmark results directly
- `tune` uses validation results to score parameter trials
- `research` optimizes code changes against combined benchmark summaries

For raw runtime summaries from the Python API, `evaluate(...)` aggregates counts and timings but does not replace ground-truth MOT metric evaluation.

## Detection sources

By default, benchmark modes run the detector configured in the benchmark YAML (`--detection-source private`, the implicit default). Use `--detection-source` to switch to public MOTChallenge detections:

| Value | Behavior |
| --- | --- |
| *(omitted)* or `private` | Run the configured detector model |
| `public` | Use the default public detector from the benchmark YAML |
| `frcnn` | Use Faster R-CNN public detections |
| `sdp` | Use SDP public detections |
| `dpm` | Use DPM public detections |

Public detections are downloaded from the benchmark's `public_detectors` config and cached alongside the standard detection cache. ReID embeddings are generated for the public detections automatically.

```bash
# Generate and evaluate with public FRCNN detections
boxmot generate --benchmark mot17 --split ablation --detection-source frcnn
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --detection-source frcnn
```

## Kalman filter noise tuning

Use `--tune-kf` to estimate per-sequence Kalman filter process and measurement noise (Q/R matrices) from cached detections and ground truth before tracking:

```bash
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --tune-kf
```

This fits noise parameters to the specific dataset and is most useful for KF-based trackers. It requires ground truth to be available for the selected split.

For `tune`, `--tune-kf` estimates noise once before the search loop and reuses it for all trials:

```bash
boxmot tune --benchmark mot17 --split ablation --tracker botsort --tune-kf --n-trials 20
```

For runtime adaptation without ground truth (e.g., deployment to new domains), use `--adaptive-kf` instead, which estimates noise online via the Mehra (1970) method.

## Postprocessing modes

`eval` supports three postprocessing modes through `--postprocessing`.
Multiple steps can be chained in order using comma separation:

- `none` – no postprocessing (default)
- `gsi` – Gaussian-smoothed interpolation: fills gaps via linear interpolation, then smooths trajectories with a Gaussian process
- `gbrc` – gradient-boosting reconnection: uses a `GradientBoostingRegressor` to interpolate and smooth trajectories
- `gta` – global tracklet association: offline pipeline that splits and reconnects tracklets across the full sequence using cached ReID embeddings

```bash
# Single step
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --postprocessing gsi

# Multiple steps applied in order
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --postprocessing gbrc,gta
```

!!! warning "Chained steps overwrite in place"
    When chaining multiple steps (e.g., `gsi,gta`), each step reads the MOT result files from the experiment directory, transforms them, and writes the results back. The second step operates on the first step's output, not the original tracker output.

## Native C++ tracker backends

`eval`, `tune`, and `research` can swap the cached tracking replay stage to a native C++ tracker runner via `--tracker-backend cpp`. See [Native C++ Integration](../native/index.md) for supported trackers, build requirements, and ReID notes.

## Common commands

```bash
# Standard evaluation
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack

# With postprocessing
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --postprocessing gsi,gta

# With KF noise tuning
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --tune-kf

# With public detections
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --detection-source frcnn

# Native C++ replay
boxmot eval --benchmark mot17 --split ablation --tracker bytetrack --tracker-backend cpp
boxmot eval --benchmark mot17 --split ablation --tracker botsort --tracker-backend cpp
```

## Main outputs

- combined benchmark metrics such as `HOTA`, `MOTA`, and `IDF1`
- per-sequence summaries
- MOT-style tracker outputs
- reused cache paths and evaluation artifacts in the run directory
