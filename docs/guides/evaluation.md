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

## Native C++ tracker backends

`eval`, `tune`, and `research` can swap the cached tracking replay stage to a native C++ tracker runner:

```bash
boxmot eval --benchmark mot17-ablation --tracker bytetrack --tracker-backend cpp
```

This keeps the existing TrackEval scoring flow, but runs the tracker replay step through the C++ executable under `boxmot/native/trackers/<tracker>`.

Native replay and live tracking are currently registered for:

| Tracker | `track` live backend | cached replay backend | Notes |
| --- | --- | --- | --- |
| `botsort` | Yes | Yes | Supports AABB/OBB and native ONNX ReID fallback. |
| `bytetrack` | Yes | Yes | Supports AABB/OBB and does not require ReID. |
| `ocsort` | Yes | Yes | Supports AABB/OBB; native backend currently uses `asso_func=iou`. |
| `sfsort` | Yes | Yes | Supports AABB/OBB and does not require ReID. |

`--tracking-backend cpp` still works as a compatibility alias, but `--tracker-backend cpp` is the preferred selector because it distinguishes tracker implementation from the process/thread executor used for cached replay.

Build requirements for native backends:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

Native builds are lazy. The first `--tracker-backend cpp` run configures and builds the matching executable or shared library under `build/native/<tracker>/`.

If you want to embed a native tracker in a standalone C++ application, see [Native C++ Integration](native-cpp.md).

For BoTSORT, native ONNX ReID inference is used as a fallback when an embedding cache is missing. If the selected ReID model is a `.pt` file, BoxMOT exports it to a native OpenCV-compatible `*_opencv.onnx` cache file and reuses that export for later native runs.

The native replay path accepts both AABB benchmark caches and OBB caches. OBB replay outputs are written in the MMOT corner format expected by the OBB evaluation flow.

## Common commands

```bash
boxmot eval --benchmark mot17-ablation --tracker boosttrack
boxmot eval --benchmark mot17-ablation --tracker boosttrack --postprocessing gsi
boxmot eval --benchmark mot17-ablation --tracker boosttrack --postprocessing gbrc
boxmot eval --benchmark mot17-ablation --tracker bytetrack --tracker-backend cpp
boxmot eval --benchmark mot17-ablation --tracker botsort:cpp
```

## Main outputs

- combined benchmark metrics such as `HOTA`, `MOTA`, and `IDF1`
- per-sequence summaries
- MOT-style tracker outputs
- reused cache paths and evaluation artifacts in the run directory

See [Results and Artifacts](results.md).
