---
description: Evaluate BoxMOT trackers on benchmark bundles and inspect TrackEval metrics.
---

# Evaluate Mode

`eval` prepares benchmark inputs, runs the selected tracker, and returns TrackEval metrics such as `HOTA`, `MOTA`, and `IDF1`.

!!! example "Evaluate from CLI or Python"

    === "CLI"

        ```bash
        boxmot eval \
          --benchmark mot17-ablation \
          --tracker boosttrack \
          --postprocessing gsi \
          --verbose
        ```

    === "Python"

        ```python
        from boxmot import boxmot

        model = boxmot(
            tracker="boosttrack",
            detector="yolo11s",
            reid="osnet_x0_25_msmt17",
        )

        metrics = model.val(
            benchmark="mot17-ablation",
            postprocessing="gsi",
            device="0",
        )

        print(metrics.HOTA)
        print(metrics.MOTA)
        print(metrics.IDF1)
        print(metrics.summary)
        ```

## Notes

- `eval` requires a benchmark bundle. There is no direct-source evaluation mode.
- Python uses `model.val(...)` rather than `model.eval(...)`.
- If caches do not exist yet, `eval` generates them before running the tracker.

## CLI Reference

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: eval
    :style: table
    :prog_name: boxmot eval
