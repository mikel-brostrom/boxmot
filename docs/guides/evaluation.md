# Evaluation and Postprocessing

Use this guide when you need to interpret benchmark outputs from `boxmot eval`, `BoxMOT.val(...)`, `tune`, or `research`.

For cache reuse, experiment IDs, and replay image-loading behavior, see [Experiment Workflows](experiments.md).

## Core metrics

- `HOTA` for overall tracking quality
- `MOTA` for CLEAR-style summary quality
- `IDF1` for identity consistency
- `AssA` and `AssRe` for association quality
- `IDSW` for ground-truth identities switching tracker IDs
- `IDt` for tracker IDs transferring to another ground-truth identity
- `IDa` for switches to a previously unmatched tracker ID
- `IDm` for transfers to a previously unmatched ground-truth identity
- `IDs` and `GT_IDs` for the number of tracker and ground-truth identities

The default console summary remains compact. The `IDt`, `IDa`, and `IDm`
diagnostics are available in returned metrics dictionaries and CI JSON output.

## Where metrics appear

- `eval` reports benchmark results directly
- `tune` uses validation results to score parameter trials
- `research` optimizes code changes against combined benchmark summaries

For raw runtime summaries from the Python API, `evaluate(...)` aggregates counts and timings but does not replace ground-truth MOT metric evaluation.

Metric evaluation runs independent sequences in separate worker processes. Its
worker count is the smaller of the sequence count and the computer's logical CPU
count minus two, with at least one worker. This is independent of `--n-threads`.
A single sequence is evaluated in the calling process, and results retain
deterministic sequence order.

## Detection sources

The experiment selected by `--experiment` declares whether a run uses a model,
named public detections, or a precomputed artifact:

- `mot17-ablation-yolox-lmbn` runs the model checkpoint selected by the experiment.
- `mot17-ablation-frcnn-lmbn`, `mot17-ablation-sdp-lmbn`, and
  `mot17-ablation-dpm-lmbn` select the corresponding named public artifact.
- `mot17-ablation-precomputed` selects the declared detection and embedding artifact.

Public detections are resolved from `boxmot/configs/artifacts` and cached
alongside normal detection caches. ReID embeddings are generated for public
detections automatically. The compatibility option `--detection-source`
accepts only `public` or `private`; prefer a source-specific experiment ID when
the exact public producer matters.

```bash
# Generate and evaluate with public FRCNN detections
boxmot generate --experiment mot17-ablation-frcnn-lmbn
boxmot eval --experiment mot17-ablation-frcnn-lmbn --tracker boosttrack
```

## Kalman filter noise tuning

Use `--tune-kf` to estimate per-sequence Kalman filter process and measurement noise (Q/R matrices) from cached detections and ground truth before tracking:

```bash
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack --tune-kf
```

This fits noise parameters to the specific dataset and is most useful for KF-based trackers. It requires ground truth to be available for the selected split.

For `tune`, `--tune-kf` estimates noise once before the search loop and reuses it for all trials:

```bash
boxmot tune --experiment mot17-ablation-yolox-lmbn --tracker botsort --tune-kf --n-trials 20
```

For runtime adaptation without ground truth, `boosttrack` and `occluboost`
expose an `adaptive_kf` tracker setting that estimates noise online via the
Mehra (1970) method. This is a tracker configuration value, not a CLI flag.
For example:

```python
from boxmot import BoxMOT

boxmot = BoxMOT(
    detector="yolov8n",
    reid="lmbn_n_duke",
    tracker="boosttrack",
    tracker_kwargs={"adaptive_kf": True},
)
metrics = boxmot.val(experiment="mot17-ablation-yolox-lmbn")
```

## Postprocessing modes

`eval` supports three postprocessing modes through `--postprocessing`.
Multiple steps can be chained in order using comma separation:

- `none` – no postprocessing (default)
- `gsi` – Gaussian-smoothed interpolation: fills gaps via linear interpolation, then smooths trajectories with a Gaussian process
- `gbrc` – gradient-boosting reconnection: uses a `GradientBoostingRegressor` to interpolate and smooth trajectories
- `gta` – global tracklet association: offline pipeline that splits and reconnects tracklets across the full sequence using cached ReID embeddings

```bash
# Single step
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack --postprocessing gsi

# Multiple steps applied in order
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack --postprocessing gbrc,gta
```

!!! warning "Chained steps overwrite in place"
    When chaining multiple steps (e.g., `gsi,gta`), each step reads the MOT result files from the experiment directory, transforms them, and writes the results back. The second step operates on the first step's output, not the original tracker output.

## Native C++ tracker backends

`eval` and `tune` can swap the cached tracking replay stage to a native C++
tracker runner via `--tracker-backend cpp`. Research currently edits and
scores Python tracker source. See [Native C++ Integration](../native/index.md)
for supported trackers, build requirements, and ReID notes.

## Common commands

```bash
# Standard evaluation
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack

# With postprocessing
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack --postprocessing gsi,gta

# With KF noise tuning
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack --tune-kf

# With public detections
boxmot eval --experiment mot17-ablation-frcnn-lmbn --tracker boosttrack

# Native C++ replay
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker bytetrack --tracker-backend cpp
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker botsort --tracker-backend cpp
```

## Main outputs

- combined benchmark metrics such as `HOTA`, `MOTA`, and `IDF1`
- per-sequence summaries
- MOT-style tracker outputs
- reused cache paths and evaluation artifacts in the run directory
