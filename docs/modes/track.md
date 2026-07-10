# Track

Use `track` when you want end-to-end detector + tracker execution on a real source such as a webcam, video file, image directory, or stream.

## Examples

!!! example

    === "CLI"

        ```bash
        boxmot track \
          --detector yolov8n \
          --reid osnet_x0_25_msmt17 \
          --tracker botsort \
          --source video.mp4 \
          --save
        ```

    === "Python"

        ```python
        from boxmot import BoxMOT

        boxmot = BoxMOT(detector="yolov8n", reid="osnet_x0_25_msmt17", tracker="botsort")
        run = boxmot.track(source="video.mp4", save=True)
        print(run)
        ```

## Common source values

- `0` for a webcam
- `video.mp4` for a local video
- `path/to/images` for an image directory
- `path/*.jpg` for a glob
- `rtsp://...` or `http://...` for a network stream
- YouTube URLs for supported detector backends

## Typical patterns

!!! example

    === "CLI"

        Track with trajectories:

        ```bash
        boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort \
          --source video.mp4 --show-trajectories --save
        ```

        Track selected classes only:

        ```bash
        boxmot track --detector yolov8s --tracker bytetrack --source 0 --classes 16,17
        ```

        Track each class independently:

        ```bash
        boxmot track --detector yolov8n --tracker bytetrack --source video.mp4 --per-class --save
        ```

    === "Python"

        ```python
        from boxmot import BoxMOT

        boxmot = BoxMOT(detector="yolov8n", reid="osnet_x0_25_msmt17", tracker="botsort")
        saved = boxmot.track(source="video.mp4", save=True, save_txt=True)
        print(saved.video_path)
        print(saved.text_path)

        filtered = BoxMOT(detector="yolov8s", tracker="bytetrack", classes=[16, 17])
        webcam_run = filtered.track(source=0, verbose=False)
        print(webcam_run.summary)
        ```

        Class filtering in Python is configured on `BoxMOT(...)` via `classes=[...]`, not passed to `track(...)` directly.

## Outputs

Depending on flags, `track` can produce:

- annotated videos or rendered frames
- MOT-style text outputs via `--save-txt`
- cropped detections via `--save-crop`
- a structured `TrackRunResult` from the Python API (see [High-level API](../python/high-level.md))

## Native C++ tracking

Use `--tracker-backend cpp` when you want the in-process native C++ tracker implementation instead of the Python implementation:

```bash
boxmot track --detector yolov8n --tracker bytetrack --tracker-backend cpp --source video.mp4
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --tracker-backend cpp --source 0
```

Native live tracking is currently registered for `botsort`, `bytetrack`, `ocsort`, `occluboost`, and `sfsort`. See [Native C++ Integration](../native/index.md) for build requirements and embedding details.

## Detection geometry

`track` accepts either AABB or OBB detections, and BoxMOT switches automatically based on tensor shape. See [Concepts](../concepts/index.md).

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: track
    :style: table
    :prog_name: boxmot track
