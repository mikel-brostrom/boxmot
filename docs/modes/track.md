---
description: Run BoxMOT tracking on webcams, videos, image folders, and streams.
---

# Track Mode

`track` runs the integrated detector + ReID + tracker pipeline on a concrete source such as a webcam, video file, image directory, glob, or stream URL.

!!! example "Track from CLI or Python"

    === "CLI"

        ```bash
        boxmot track \
          --detector yolo11s \
          --reid osnet_x0_25_msmt17 \
          --tracker botsort \
          --source video.mp4 \
          --show-trajectories \
          --save \
          --save-txt
        ```

    === "Python"

        ```python
        from boxmot import boxmot

        model = boxmot(
            detector="yolo11s",
            reid="osnet_x0_25_msmt17",
            tracker="botsort",
        )

        results = model.track(
            source="video.mp4",
            show_trajectories=True,
            save=True,
            save_txt=True,
        )

        print(results.video_path)
        print(results.text_path)
        print(results.timings)
        ```

## Notes

- `track` requires `--source` in CLI form or `source=` in Python form.
- Use motion-only trackers such as `bytetrack`, `ocsort`, or `sfsort` when you do not want ReID embeddings.
- For frame-by-frame custom loops, use the lower-level streaming helper shown in [Python API](../usage/python.md). It yields structured `Tracks` objects with `.xyxy`, `.xywha`, `.conf`, `.cls`, `.id`, and `.det_ind`.

## CLI Reference

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: track
    :style: table
    :prog_name: boxmot track
