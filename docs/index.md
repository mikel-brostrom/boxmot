# Quickstart

!!! example "Quickstart"

    === "CLI"

        Install BoxMOT and inspect the CLI:

        ```bash
        pip install boxmot
        boxmot --help
        ```

        Track a video:

        ```bash
        boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --source video.mp4 --save
        ```

        Benchmark a tracker on a built-in config:

        ```bash
        boxmot eval --benchmark mot17-ablation --tracker boosttrack --verbose
        ```

        Research tracker code changes on a built-in config:

        ```bash
        boxmot research --benchmark mot17-ablation --tracker bytetrack --proposal-model openai/gpt-5.4 --max-metric-calls 24
        ```

    === "Python"

        Use the high-level Python API:

        ```python
        from boxmot import Boxmot

        model = Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
        run = model.track(source="video.mp4", save=True)
        print(run.summary)

        metrics = model.val(benchmark="mot17-mini")
        print(metrics.summary)
        ```

The canonical public API lives in `boxmot/api.py` and is re-exported from `boxmot`. Shared CLI and Python defaults come from `boxmot/configs/modes.yaml`, so detector, ReID, tracker, and runtime defaults stay aligned across both entry points.

Next steps:

- [Choose a Mode](getting-started/workflows.md)
- [Modes Overview](modes/index.md)
- [CLI Usage](usage/cli.md)
- [Python Usage](usage/python.md)
- [Configuration](usage/configuration.md)
- [Python API](python/index.md)
- [Trackers](trackers/index.md)
