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

`eval` supports three postprocessing modes through `--postprocessing`:

- `none`
- `gsi` for Gaussian-smoothed interpolation
- `gbrc` for gradient-boosting-based reconnection and interpolation

## Common commands

```bash
boxmot eval --benchmark mot17-ablation --tracker boosttrack
boxmot eval --benchmark mot17-ablation --tracker boosttrack --postprocessing gsi
boxmot eval --benchmark mot17-ablation --tracker boosttrack --postprocessing gbrc
```

## Main outputs

- combined benchmark metrics such as `HOTA`, `MOTA`, and `IDF1`
- per-sequence summaries
- MOT-style tracker outputs
- reused cache paths and evaluation artifacts in the run directory

See [Results and Artifacts](results.md).
