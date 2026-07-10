# Tune

Use `tune` to search tracker hyperparameters against one or more objective metrics.

## Examples

!!! example

    === "CLI"

        ```bash
        boxmot tune --benchmark mot17 --split ablation --tracker ocsort --n-trials 10
        ```

    === "Python"

        ```python
        from boxmot import BoxMOT

        boxmot = BoxMOT(detector="yolov8n", reid="lmbn_n_duke", tracker="ocsort")
        tuned = boxmot.tune(benchmark="mot17", split="ablation", n_trials=10)
        print(tuned)
        print(tuned.best_yaml)
        ```

## How it works

Tracker search spaces come from the selected tracker YAML in `boxmot/configs/trackers`. Runtime defaults use each parameter's `default` value, while tuning uses its `type`, `range`, and `options`.

## Public detections

Use `--detection-source` to tune against public MOTChallenge detections instead of the configured detector:

```bash
boxmot tune --benchmark mot17 --split ablation --tracker ocsort --detection-source frcnn --n-trials 10
```

See [Evaluate — Public detections](eval.md#public-detections) for the full list of sources.

## Kalman filter noise tuning

Use `--tune-kf` to estimate Kalman filter noise matrices (Q/R) once before the tuning loop. The estimated noise is then reused for all trials:

```bash
boxmot tune --benchmark mot17 --split ablation --tracker botsort --tune-kf --n-trials 20
```

This is especially useful for KF-based trackers where the default noise parameters may not suit the dataset.

## Postprocessing

Use `--postprocessing` to apply postprocessing after each trial's tracking run before scoring:

```bash
boxmot tune --benchmark mot17 --split ablation --tracker ocsort --postprocessing gsi --n-trials 10
```

See [Evaluate — Postprocessing](eval.md#postprocessing) for available steps and chaining behavior.

## Native C++ trials

Use `--tracker-backend cpp` when you want each trial to score the native C++ tracker backend instead of the Python backend:

```bash
boxmot tune --benchmark mot17 --split ablation --tracker sfsort --tracker-backend cpp --n-trials 10
```

Native tuning uses the same search space YAML as the Python tracker and swaps only the tracker implementation used during cached replay. Native replay is currently available for `botsort`, `bytetrack`, `ocsort`, `occluboost`, and `sfsort`.

## Objective configuration

!!! example

    === "CLI"

        Single-objective tuning:

        ```bash
        boxmot tune --benchmark mot17 --split ablation --tracker bytetrack --objectives HOTA
        ```

        Multi-objective tuning:

        ```bash
        boxmot tune --benchmark mot17 --split ablation --tracker bytetrack \
          --objectives HOTA IDF1_rate \
          --maximize HOTA \
          --minimize IDSW_rate
        ```

    === "Python"

        ```python
        from boxmot import BoxMOT

        boxmot = BoxMOT(detector="yolov8n", reid="lmbn_n_duke", tracker="bytetrack")
        tuned = boxmot.tune(
            benchmark="mot17",
            split="ablation",
            n_trials=10,
            maximize=("HOTA",),
            minimize=("IDSW_rate",),
        )
        print(tuned)
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
