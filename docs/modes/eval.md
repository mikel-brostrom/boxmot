# Evaluate

Use `eval` to score tracking runs on MOT-style datasets with TrackEval-backed metrics.

## Examples

!!! example

    === "CLI"

        ```bash
        boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --verbose
        ```

    === "Python"

        ```python
        from boxmot import Boxmot

        boxmot = Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
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
        from boxmot import Boxmot

        boxmot = Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
        metrics = boxmot.val(benchmark="mot17", split="ablation")
        print(metrics)
        ```

## Postprocessing

!!! example

    === "CLI"

        `eval` can apply optional postprocessing before scoring.
        Multiple steps can be chained with commas (applied in order):

        ```bash
        boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --postprocessing gsi
        boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --postprocessing gbrc,gta
        ```

    === "Python"

        `Boxmot.val(...)` is the Python-facing validation entry point. Postprocessing details and metric interpretation are the same as in the CLI evaluation pipeline.

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
