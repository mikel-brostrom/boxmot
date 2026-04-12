# Tune

Use `tune` to search tracker hyperparameters against one or more objective metrics.

## Examples

!!! example

    === "CLI"

        ```bash
        boxmot tune --benchmark mot17-ablation --tracker ocsort --n-trials 10
        ```

    === "Python"

        ```python
        from boxmot import Boxmot

        boxmot = Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="ocsort")
        tuned = boxmot.tune(benchmark="mot17-ablation", n_trials=10)
        print(tuned.best_yaml)
        ```

## How it works

Tracker search spaces come from the selected tracker YAML in `boxmot/configs/trackers`. Runtime defaults use each parameter's `default` value, while tuning uses its `type`, `range`, and `options`.

## Objective configuration

!!! example

    === "CLI"

        Single-objective tuning:

        ```bash
        boxmot tune --benchmark mot17-ablation --tracker bytetrack --objectives HOTA
        ```

        Multi-objective tuning:

        ```bash
        boxmot tune --benchmark mot17-ablation --tracker bytetrack \
          --objectives HOTA IDF1_rate \
          --maximize HOTA \
          --minimize IDSW_rate
        ```

    === "Python"

        ```python
        from boxmot import Boxmot

        boxmot = Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="bytetrack")
        tuned = boxmot.tune(
            benchmark="mot17-ablation",
            n_trials=10,
            maximize=("HOTA",),
            minimize=("IDSW_rate",),
        )
        print(tuned.best_config)
        ```

## Outputs

Tuning writes trial artifacts and a `best.yaml` tracker config that can be reused in later runs.

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: tune
    :style: table
    :prog_name: boxmot tune
