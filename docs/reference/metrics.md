# Metrics and Evaluation

BoxMOT's benchmark workflows use TrackEval-style metrics for MOT evaluation.

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
