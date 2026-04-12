# Evaluate

Use `eval` to score tracking runs on MOT-style datasets with TrackEval-backed metrics.

## Examples

!!! example

    === "CLI"

        ```bash
        boxmot eval --benchmark mot17-ablation --tracker boosttrack --verbose
        ```

    === "Python"

        ```python
        from boxmot.api import Boxmot

        boxmot = Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
        metrics = boxmot.val(benchmark="mot17-ablation")
        print(metrics.summary)
        ```

## Typical workflow

!!! example

    === "CLI"

        For repeated experiments:

        ```bash
        boxmot generate --benchmark mot17-ablation
        boxmot eval --benchmark mot17-ablation --tracker boosttrack
        ```

        This lets `eval` reuse precomputed detections and embeddings.

    === "Python"

        ```python
        from boxmot.api import Boxmot

        boxmot = Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
        metrics = boxmot.val(benchmark="mot17-ablation")
        print(metrics.summary)
        ```

## Postprocessing

!!! example

    === "CLI"

        `eval` can apply optional postprocessing before scoring:

        ```bash
        boxmot eval --benchmark mot17-ablation --tracker boosttrack --postprocessing gsi
        boxmot eval --benchmark mot17-ablation --tracker boosttrack --postprocessing gbrc
        ```

    === "Python"

        `Boxmot.val(...)` is the Python-facing validation entry point. Postprocessing details and metric interpretation are the same as in the CLI evaluation pipeline.

See [Evaluation and Postprocessing](../guides/evaluation.md).

## Main outputs

- combined benchmark metrics such as `HOTA`, `MOTA`, and `IDF1`
- per-sequence summaries
- MOT-style tracker outputs
- reused cache paths and evaluation artifacts in the run directory

See [Evaluation and Postprocessing](../guides/evaluation.md) and [Results and Artifacts](../guides/results.md).

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: eval
    :style: table
    :prog_name: boxmot eval
