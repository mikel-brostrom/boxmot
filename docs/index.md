# Quickstart

!!! example "Quickstart"

    === "CLI"

        Install BoxMOT and inspect the CLI:

        ```bash
        pip install "boxmot[yolo]"
        boxmot --help
        ```

        Track a video:

        ```bash
        boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --source video.mp4 --save
        ```

        Run a tracker experiment from a built-in config:

        ```bash
        boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack --verbose
        ```

        Research tracker code changes on a built-in config:

        ```bash
        pip install "boxmot[yolo,research]"
        boxmot research --experiment mot17-ablation-yolox-lmbn --tracker bytetrack --proposal-model openai/gpt-5.4 --max-metric-calls 24
        ```

    === "Python"

        Use the high-level Python API:

        ```python
        from boxmot import BoxMOT

        boxmot = BoxMOT(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
        run = boxmot.track(source="video.mp4", save=True)
        print(run)

        metrics = boxmot.val(experiment="mot17-mini-train-yolox-lmbn")
        print(metrics)
        ```

The high-level Python API is available directly from `boxmot`. Shared tracking
defaults come from `boxmot/configs/runtime.yaml`, so the CLI and Python entry
points remain aligned.

Next steps:

- [Modes Overview](modes/index.md)
- [CLI Usage](usage/index.md)
- [Python API](python/index.md)
- [Configuration](config/index.md)
- [API Reference](python/high-level.md)
- [Trackers](trackers/index.md)
- [Native C++ Integration](native/index.md)
