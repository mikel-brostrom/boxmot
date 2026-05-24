# Evaluation and Postprocessing

Use this guide when you need to interpret benchmark outputs from `boxmot eval`, `Boxmot.val(...)`, `tune`, or `research`.

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

For raw runtime summaries from the Python API, `evaluate(...)` aggregates counts and timings but does not replace TrackEval ground-truth evaluation.

## Postprocessing modes

`eval` supports three postprocessing modes through `--postprocessing`.
Multiple steps can be chained in order using comma separation:

- `none` – no postprocessing (default)
- `gsi` – Gaussian-smoothed interpolation
- `gbrc` – gradient-boosting-based reconnection and interpolation
- `gta` – global tracklet association

```bash
# Single step
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --postprocessing gsi

# Multiple steps applied in order
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --postprocessing gbrc,gta
```

## Native C++ tracker backends

`eval`, `tune`, and `research` can swap the cached tracking replay stage to a native C++ tracker runner via `--tracker-backend cpp`. See [Native C++ Integration](../native/index.md) for supported trackers, build requirements, and ReID notes.

## Common commands

```bash
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --postprocessing gbrc,gta
boxmot eval --benchmark mot17 --split ablation --tracker bytetrack --tracker-backend cpp
boxmot eval --benchmark mot17 --split ablation --tracker botsort:cpp
```

## Main outputs

- combined benchmark metrics such as `HOTA`, `MOTA`, and `IDF1`
- per-sequence summaries
- MOT-style tracker outputs
- reused cache paths and evaluation artifacts in the run directory
