# Modes Overview

The engine exposes one command group. Domain components remain usable on their
own and pipelines contain no source, output, persistence, retry, or display
logic.

Each mode guide includes **CLI** and **Python** example tabs. Selecting a tab
switches other examples with the same label and keeps that preference as you
browse. Python examples use the existing engine workflows and return their
results to your program. Pass options as keyword arguments; the workflow
supplies defaults and normalizes them internally.

| Mode | Purpose | Required input |
| --- | --- | --- |
| [`ui`](ui.md) | Configure and launch common commands interactively | interactive terminal |
| [`install`](install.md) | Install optional dependencies or the KITTI object evaluator | `--extra`, `--requirement`, or `--kitti-devkit` |
| [`track`](track.md) | Track media or replay saved TrackR-CNN predictions | `--source` plus components, or `--detections`, `--images`, and `--instances` |
| [`materialize`](materialize.md) | Publish perception or derive a timestamped frame-loss dataset | image experiment, or `--time-variant` with dataset, sequence, and `--build` |
| [`eval`](eval.md) | Evaluate a perception build, saved 2D boxes, or sensor inputs | image experiment/build, saved-box dataset, or sensor dataset with EagerMOT |
| [`tune`](tune.md) | Optimize tracker parameters from a build or saved sensor inputs | image experiment/build, or sensor dataset with EagerMOT |
| [`research`](research.md) | Score proposed tracker changes against a build | experiment, `--build`, and proposal-model credentials |
| [`train-reid`](train.md) | Train an appearance backbone | ReID dataset/config |
| [`eval-reid`](eval-reid.md) | Evaluate query/gallery retrieval | checkpoint and ReID dataset |
| [`compare-reid`](compare-reid.md) | Compare checkpoints across ReID datasets | checkpoints and targets |
| [`export`](export.md) | Export a ReID backbone | checkpoint and formats |
| [`build`](../native/index.md#building) | Compile native tracker and ReID libraries | build toolchain; optional tracker selector |

The CLI also lists `export-edgetam`, an unfinished adapter whose conversion
backend is absent. It is not a working export workflow; see the
[mask support notes](../tasks/masks.md).

## Live tracking

=== "CLI"

    ```bash
    boxmot track \
      --source video.mp4 \
      --detector yolov8n \
      --reid osnet_x0_25_msmt17 \
      --tracker botsort \
      --save
    ```

=== "Python"

    ```python
    from boxmot.engine.tracking.workflow import run_track

    result = run_track(
        source="video.mp4",
        detector="yolov8n",
        reid="osnet_x0_25_msmt17",
        tracker="botsort",
        save=True,
    )
    print(result.video_path)
    ```

The engine creates canonical RGB `Frame` values, runs a `TrackingPipeline`, and
sends each `(frame, result)` pair to configured sinks.

## Reproducible benchmark workflow

Perception runs once during materialization. Eval and tune can perform that
deterministic step automatically; research consumes an explicit build:

=== "CLI"

    ```bash
    boxmot materialize --experiment mot17/ablation-yolox-lmbn.yaml
    boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID
    boxmot tune --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID
    boxmot research --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID
    ```

=== "Python"

    ```python
    from boxmot.engine.eval.evaluator import run_eval
    from boxmot.engine.materialization.workflow import materialize
    from boxmot.engine.research.runner import run_research
    from boxmot.engine.tuning.tuner import run_tune

    if __name__ == "__main__":
        build = materialize(experiment="mot17/ablation-yolox-lmbn.yaml")
        evaluation = run_eval(experiment="mot17/ablation-yolox-lmbn.yaml", build=build)
        tuning = run_tune(experiment="mot17/ablation-yolox-lmbn.yaml", build=build)
        research = run_research(experiment="mot17/ablation-yolox-lmbn.yaml", build=build)
    ```

Use the [source setup](../getting-started/pytorch.md#source-checkout) for YOLOX,
add `evolve` for tuning and `research` for code search, and replace `BUILD_ID`
with materialization's printed ID. Research also needs credentials for its
proposal model; see [Research](research.md) before running it.

The Python replay functions require an explicit build. Here `materialize`
returns the build path consumed by each subsequent call. Keep the
`if __name__ == "__main__":` guard in scripts that start worker processes.

These workflows never select a latest build. They verify source, split,
taxonomy, geometry, and component fingerprints before replay.

See [Materialize](materialize.md), [Evaluate](eval.md), [Tune](tune.md), and
[Research](research.md).

Use [`materialize --time-variant`](time-variant.md) to compare timing modes under
reproducible frame loss without rerunning perception.

## ReID workflows

Each guide pairs command-line options with Python configuration and execution:

- [Train ReID](train.md) on identity-labeled crops.
- [Evaluate ReID](eval-reid.md) on a query/gallery split.
- [Compare ReID](compare-reid.md) across checkpoints and datasets.
- [Export](export.md) a trained appearance encoder for deployment.

## Python composition

Python callers use factories and canonical structures directly. See the
[Python API](../python/index.md) for detector, segmentor, appearance encoder,
tracker, and pipeline examples.
