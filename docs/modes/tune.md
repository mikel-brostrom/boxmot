# Tune

Use `tune` to search tracker hyperparameters against one or more objective metrics.

## Examples

!!! example

    === "CLI"

        ```bash
        boxmot tune --experiment mot17-ablation-yolox-lmbn --tracker ocsort --n-trials 10
        ```

    === "Python"

        ```python
        from boxmot import BoxMOT

        boxmot = BoxMOT(detector="yolov8n", reid="lmbn_n_duke", tracker="ocsort")
        tuned = boxmot.tune(experiment="mot17-ablation-yolox-lmbn", n_trials=10)
        print(tuned)
        print(tuned.best_yaml)
        ```

## How it works

Runtime defaults and search metadata come from the same
`boxmot/configs/trackers/<tracker>.yaml`. Runtime construction extracts each
parameter's `default`, while the tuner reads its search policy and combines it
with any selected runtime overrides to form the baseline.

## Public detections

Select a public-detection experiment to tune against public MOTChallenge detections:

```bash
boxmot tune --experiment mot17-ablation-frcnn-lmbn --tracker ocsort --n-trials 10
```

See [Evaluate — Public detections](eval.md#public-detections) for the full list of sources.

## Kalman filter noise tuning

Use `--tune-kf` to estimate Kalman filter noise matrices (Q/R) once before the tuning loop. The estimated noise is then reused for all trials:

```bash
boxmot tune --experiment mot17-ablation-yolox-lmbn --tracker botsort --tune-kf --n-trials 20
```

This is especially useful for KF-based trackers where the default noise parameters may not suit the dataset.

## Postprocessing

Use `--postprocessing` to apply postprocessing after each trial's tracking run before scoring:

```bash
boxmot tune --experiment mot17-ablation-yolox-lmbn --tracker ocsort --postprocessing gsi --n-trials 10
```

See [Evaluate — Postprocessing](eval.md#postprocessing) for available steps and chaining behavior.

## Native C++ trials

Use `--tracker-backend cpp` when you want each trial to score the native C++ tracker backend instead of the Python backend:

```bash
boxmot tune --experiment mot17-ablation-yolox-lmbn --tracker sfsort --tracker-backend cpp --n-trials 10
```

Native tuning uses the same tracker YAML search space as the Python tracker and swaps only the tracker implementation used during cached replay. Native replay is currently available for `botsort`, `bytetrack`, `ocsort`, `occluboost`, and `sfsort`.

## Objective configuration

!!! example

    === "CLI"

        Single-objective tuning:

        ```bash
        boxmot tune --experiment mot17-ablation-yolox-lmbn --tracker bytetrack --objectives HOTA
        ```

        Multi-objective tuning:

        ```bash
        boxmot tune --experiment mot17-ablation-yolox-lmbn --tracker bytetrack \
          --objectives HOTA IDF1 IDSW_rate \
          --maximize HOTA IDF1 \
          --minimize IDSW_rate
        ```

    === "Python"

        ```python
        from boxmot import BoxMOT

        boxmot = BoxMOT(detector="yolov8n", reid="lmbn_n_duke", tracker="bytetrack")
        tuned = boxmot.tune(
            experiment="mot17-ablation-yolox-lmbn",
            split="ablation",
            n_trials=10,
            maximize=("HOTA",),
            minimize=("IDSW_rate",),
        )
        print(tuned)
        print(tuned.best_config)
        ```

## Outputs

Tuning writes trial artifacts and a fully resolved scalar `best.yaml` tracker
config that can be reused by `create_tracker`.

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: tune
    :style: table
    :prog_name: boxmot tune
