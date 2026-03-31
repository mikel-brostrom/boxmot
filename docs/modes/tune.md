---
description: Tune BoxMOT tracker hyperparameters against one or more benchmark metrics.
---

# Tune Mode

`tune` searches the selected tracker YAML parameter space with Ray Tune and returns the best configuration together with the corresponding benchmark metrics.

!!! example "Tune from CLI or Python"

    === "CLI"

        ```bash
        boxmot tune \
          --benchmark mot17-ablation \
          --tracker deepocsort \
          --n-trials 10 \
          --objectives HOTA \
          --objectives IDF1
        ```

    === "Python"

        ```python
        from boxmot import boxmot

        model = boxmot(
            tracker="deepocsort",
            detector="yolo11s",
            reid="osnet_x0_25_msmt17",
        )

        results = model.tune(
            benchmark="mot17-ablation",
            n_trials=10,
            objectives=["HOTA", "IDF1"],
        )

        print(results.best_config)
        print(results.best_yaml)
        print(results.HOTA)
        print(results.IDF1)
        ```

## Notes

- `tune` requires a benchmark bundle and uses the selected tracker YAML as its search space source.
- Python uses `model.tune(...)` and returns a `TuneResults` wrapper with `best`, `trials`, and direct metric accessors.
- `tune` also reuses or generates detection and embedding caches automatically.

## CLI Reference

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: tune
    :style: table
    :prog_name: boxmot tune
