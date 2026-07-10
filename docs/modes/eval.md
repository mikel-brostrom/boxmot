# Evaluate

Use `eval` to score tracking runs on MOT-style datasets with BoxMOT's in-repo MOT metrics.

## Examples

!!! example

    === "CLI"

        ```bash
        boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --verbose
        ```

    === "Python"

        ```python
        from boxmot import BoxMOT

        boxmot = BoxMOT(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
        metrics = boxmot.val(benchmark="mot17", split="ablation")
        print(metrics)
        ```

## Typical workflow

!!! example

    === "CLI"

        For repeated experiments:

        ```bash
        boxmot generate --benchmark mot17 --split ablation
        boxmot eval --benchmark mot17 --split ablation --tracker boosttrack
        ```

        This lets `eval` reuse precomputed detections and embeddings.

    === "Python"

        ```python
        from boxmot import BoxMOT

        boxmot = BoxMOT(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
        metrics = boxmot.val(benchmark="mot17", split="ablation")
        print(metrics)
        ```

## Public detections

Use `--detection-source` to run with public MOTChallenge detections instead of the benchmark's configured detector:

```bash
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --detection-source frcnn
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --detection-source sdp
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --detection-source dpm
```

`--detection-source public` uses the default public detector defined in the benchmark YAML.
When omitted (or `--detection-source private`), `eval` runs the configured detector model.

See [Benchmark Workflows](../guides/benchmarks.md#public-detections) for details on how public detections are resolved.

## Kalman filter noise tuning

Use `--tune-kf` to estimate per-sequence Kalman filter process and measurement noise (Q/R matrices) from the cached detections and ground truth before tracking:

```bash
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --tune-kf
```

This is most useful for trackers with Kalman-filter-based motion models. It requires cached detections and ground truth to be available.

For runtime adaptation without ground truth, use `--adaptive-kf` instead, which estimates noise online via the Mehra (1970) method.

## Postprocessing

!!! example

    === "CLI"

        `eval` can apply optional postprocessing before scoring.
        Multiple steps can be chained with commas and are applied sequentially to the same result files:

        ```bash
        # Single step
        boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --postprocessing gsi

        # Chained: GSI runs first, then GTA reads GSI's output
        boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --postprocessing gsi,gta
        ```

        Available steps:

        | Step | Description |
        | --- | --- |
        | `gsi` | Gaussian-smoothed interpolation — fills gaps and smooths trajectories |
        | `gbrc` | Gradient-boosting reconnection — ML-based interpolation and smoothing |
        | `gta` | Global tracklet association — offline split-and-connect across the full sequence |

    === "Python"

        `BoxMOT.val(...)` is the Python-facing validation entry point. Postprocessing details and metric interpretation are the same as in the CLI evaluation pipeline.

!!! warning "Chained steps overwrite in place"
    When chaining multiple postprocessing steps, each step reads the MOT result files, transforms them, and writes back to the same directory. The second step operates on the output of the first.

See [Evaluation and Postprocessing](../guides/evaluation.md).

See [Benchmark Workflows](../guides/benchmarks.md) for cache reuse, MMOT benchmark ids, and replay image-loading behavior.

## Native C++ replay

Use `--tracker-backend cpp` to run the cached replay stage through a native tracker implementation:

```bash
boxmot eval --benchmark mot17 --split ablation --tracker bytetrack --tracker-backend cpp
boxmot eval --benchmark mot17 --split ablation --tracker ocsort:cpp
```

Native replay is currently available for `botsort`, `bytetrack`, `ocsort`, `occluboost`, and `sfsort`. `--tracking-backend cpp` is still accepted as a compatibility alias, but `--tracker-backend cpp` is the canonical selector.

## Main outputs

- combined benchmark metrics such as `HOTA`, `MOTA`, and `IDF1`
- per-sequence summaries
- optional runtime timing summary with `--show-timing`
- MOT-style tracker outputs
- reused cache paths and evaluation artifacts in the run directory

See [Evaluation and Postprocessing](../guides/evaluation.md).

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: eval
    :style: table
    :prog_name: boxmot eval
