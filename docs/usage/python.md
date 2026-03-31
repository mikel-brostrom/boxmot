---
description: Use BoxMOT from Python with tracker classes, the tracker factory, and custom detector loops.
---

# Python API

BoxMOT exposes two Python layers:

- a stateful workflow wrapper through lowercase `boxmot(...)` and the `BoxMOT` class
- lower-level tracking building blocks such as tracker classes, `create_tracker(...)`, and the generator-style `track(...)` helper

Use the workflow wrapper when you want the Python API to mirror the CLI. Use the lower-level building blocks when you already have detections, embeddings, or a custom frame loop.

Common Python patterns are:

- you already have detections from your own detector
- you want to combine a detector callable, an optional ReID model, and a BoxMOT tracker in your own loop
- you want benchmark metrics such as `HOTA`, `MOTA`, and `IDF1` from `model.val()`
- you want to launch tracker hyperparameter tuning from `model.tune()` and inspect the best trial from Python
- you want to run the integrated tracking pipeline from `model.track()`
- you want to export ReID weights from `model.export()`

## Stateful Workflow Wrapper

Lowercase `boxmot(...)` returns a `BoxMOT` instance:

```python
from boxmot import BoxMOT, boxmot

model = boxmot(tracker="boosttrack")
assert isinstance(model, BoxMOT)
```

The wrapper remembers the most recent overrides you pass in. That makes repeated `val()`, `tune()`, `track()`, and `export()` calls feel like the CLI with persistent defaults.

Normalization rules worth knowing:

- `benchmark=` and `data=` map to the same benchmark setting
- CLI-style dashed flags become Python keyword arguments such as `batch_size`, `save_txt`, and `show_trajectories`
- `batch=` is accepted as a convenience alias for `batch_size`
- bare model names such as `"yolo11s"` or `"osnet_x0_25_msmt17"` resolve to `.pt` files under the BoxMOT weights directory
- `detector` and `reid` can be strings, `Path` objects, or lists for benchmark-oriented workflows; the current eval/tune run uses the first detector/ReID pair as the active configuration

## Benchmark Evaluation with `val()`

Use `boxmot(...)` when you want a YOLO-style evaluation object that remembers its benchmark and runtime settings across repeated validations.

```python
from boxmot import boxmot

tracker = boxmot(
    tracker="boosttrack",
)

metrics = tracker.val(benchmark="mot17-mini")

print(metrics.HOTA)
print(metrics.MOTA)
print(metrics.IDF1)
print(metrics.summary_name)
print(metrics.summary)
print(metrics.to_dict())
```

You can override settings on any call, and the latest values are remembered for the next `val()` run:

```python
metrics = tracker.val(data="mot17-mini", imgsz=640, batch=16, conf=0.25, iou=0.7, device="0")
metrics = tracker.val()
```

`TrackEvalMetrics` gives you both flattened summary access and class-aware access:

```python
print(metrics["HOTA"])
print(metrics.hota)              # case-insensitive attribute access
print(metrics.classes)           # per-class metrics when available
print(metrics.per_sequence)      # per-sequence metrics for flat benchmark outputs
print(metrics.raw)               # deep-copied raw result dict
```

## Hyperparameter Search with `tune()`

Use the same factory when you want a Python entry point for the existing `boxmot tune` workflow.

```python
from boxmot import boxmot

tracker = boxmot(
    tracker="boosttrack",
    reid="osnet_x0_25_msmt17",
)

results = tracker.tune(data="mot17-mini", n_trials=8, device="0", batch_size=16)

print(results.HOTA)          # best-trial HOTA
print(results.best_config)   # tracker params for the best trial
print(results.best_yaml)     # saved YAML path under runs/ray/
print(results.summary_path)  # generated Markdown summary
print(results.best.trial_id)
print(results.best.metrics.IDF1)
```

As with `val()`, overrides are remembered for later calls:

```python
results = tracker.tune(data="mot17-mini", n_trials=12, objectives=["HOTA", "IDF1"])
results = tracker.tune()
```

`TuneResults` exposes both the best trial and the full trial list:

```python
print(results.objectives)
print(results.maximize)
print(results.minimize)
print(results.trials[0].config)
print(results.trials[0].metrics.to_dict())
```

## Integrated Tracking with `track()`

Use the same factory when you want the Python API to mirror `boxmot track` and remember its runtime overrides.

```python
from boxmot import boxmot

tracker = boxmot(
    tracker="boosttrack",
    detector="yolo11s",
    reid="osnet_x0_25_msmt17",
)

results = tracker.track(
    source="video.mp4",
    imgsz=640,
    conf=0.25,
    iou=0.7,
    device="0",
    save=True,
    save_txt=True,
)

print(results.video_path)
print(results.text_path)
print(results.timings)
print(results.to_dict())
```

`track()` requires a `source=` on the first call, then remembers it for later runs.

## ReID Export with `export()`

Use `export()` when you want a Python entry point for the existing `boxmot export` workflow.

```python
from boxmot import boxmot

tracker = boxmot(reid="osnet_x0_25_msmt17")

results = tracker.export(include=("onnx", "engine"), device="0", half=True)

print(results.weights)
print(results.onnx)
print(results.engine)
print(results.files)
print(results.output_dir)
```

`export()` uses `weights=` explicitly when provided, otherwise it falls back to the configured `reid=` model.

## Result Objects

| Return type | Produced by | Useful attributes |
| --- | --- | --- |
| `TrackEvalMetrics` | `val()` | metric accessors like `HOTA`, `MOTA`, `IDF1`, plus `summary`, `classes`, `per_sequence`, `raw` |
| `TuneResults` | `tune()` | `best`, `best_config`, `best_yaml`, `summary_path`, `trials`, plus direct metric access |
| `TrackResults` | `track()` | `save_dir`, `video_path`, `text_path`, `frames`, `timings`, `raw` |
| `ExportResults` | `export()` | `weights`, `files`, format attributes such as `.onnx`, `output_dir`, `input_shape`, `output_shape` |

The lower-level streaming helper `track(...)` yields one `Tracks` object per frame rather than a raw string or a bare array.

## Detection Shapes

BoxMOT switches between AABB and OBB tracking based on the shape of the detection array you pass in.

| Geometry | Input detections | Output tracks |
| --- | --- | --- |
| AABB | `(N, 6)` = `(x1, y1, x2, y2, conf, cls)` | `(N, 8)` = `(x1, y1, x2, y2, id, conf, cls, det_ind)` |
| OBB | `(N, 7)` = `(cx, cy, w, h, angle, conf, cls)` | `(N, 9)` = `(cx, cy, w, h, angle, id, conf, cls, det_ind)` |

## Use a Tracker Class Directly

If you already have per-frame detections, instantiate a tracker and call `update(...)` once per frame.

```python
from pathlib import Path

import cv2
import numpy as np

from boxmot import BotSort

tracker = BotSort(
    reid_weights=Path("osnet_x0_25_msmt17.pt"),
    device="cpu",
    half=False,
)

cap = cv2.VideoCapture("video.mp4")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Replace this with your detector output for the current frame.
    detections = np.empty((0, 6), dtype=np.float32)
    # detections = your_detector(frame)

    tracks = tracker.update(detections, frame)

    print(tracks)
    tracker.plot_results(frame, show_trajectories=True)

cap.release()
cv2.destroyAllWindows()
```

## Use the Tracker Factory

Use `create_tracker()` when you want BoxMOT to load the default tracker parameters from the matching YAML file.

```python
from pathlib import Path

from boxmot import create_tracker, get_tracker_config

tracker = create_tracker(
    tracker_type="deepocsort",
    tracker_config=get_tracker_config("deepocsort"),
    reid_weights=Path("osnet_x0_25_msmt17.pt"),
    device="cpu",
    half=False,
    per_class=False,
)
```

This is the same configuration path used by the CLI when you select `--tracker deepocsort`.

## End-to-End Loop with `track()`

If you prefer a generator-style loop, BoxMOT also exposes a `track()` helper that combines:

- a source
- a detector callable
- an optional ReID callable
- a tracker

Your detector callable must return a detection array for one frame at a time.

```python
from pathlib import Path

import numpy as np

from boxmot import ReID, create_tracker, get_tracker_config, track


def detector(frame: np.ndarray) -> np.ndarray:
    # Replace this with your detector implementation.
    # Return (N, 6) AABB detections or (N, 7) OBB detections.
    return np.empty((0, 6), dtype=np.float32)


reid = ReID(Path("osnet_x0_25_msmt17.pt"), device="cpu", half=False)
tracker = create_tracker(
    tracker_type="botsort",
    tracker_config=get_tracker_config("botsort"),
    reid_weights=Path("osnet_x0_25_msmt17.pt"),
    device="cpu",
    half=False,
    per_class=False,
)

for result in track("video.mp4", detector, reid, tracker, verbose=False):
    print(result.xyxy)
    print(result.id)
    if not result.show():
        break
```

For motion-only trackers such as `bytetrack`, `ocsort`, or `sfsort`, pass `reid=None`.

Each yielded item is a structured per-frame result:

```python
results = track("video.mp4", detector, reid, tracker, verbose=False)

for frame_tracks in results:
    print(frame_tracks.xyxy)    # AABB tracks
    print(frame_tracks.xywha)   # OBB tracks, otherwise None
    print(frame_tracks.conf)
    print(frame_tracks.cls)
    print(frame_tracks.id)
    print(frame_tracks.det_ind)
    print(frame_tracks.tracks)  # raw tracker output for the frame

# let BoxMOT drain the stream and write MOT text when needed
results = track("video.mp4", detector, reid, tracker, verbose=False)
results.save("tracks.txt")
```

`Tracks.xyxy` is always available. For OBB trackers, `Tracks.xywha` contains the oriented boxes and `Tracks.xyxy` returns the enclosing axis-aligned boxes for convenience.

## Working with Track IDs

The final column of every BoxMOT output row is `det_ind`, which points back to the detector row used to update the track. This is useful when you need to join track results to detector metadata or other per-detection features.

## Choosing the Right Tracker

- Use `bytetrack` or `sfsort` when you want fast motion-only baselines.
- Use `botsort`, `strongsort`, `deepocsort`, `hybridsort`, or `boosttrack` when appearance embeddings matter.
- Use `botsort`, `bytetrack`, `ocsort`, or `sfsort` when you need OBB support.

## Next Steps

- Open [Trackers Overview](../trackers/index.md) to compare tracker backends.
- Open [Track Mode](../modes/track.md) to mirror the same setup from the CLI.
- Open [CLI Usage](cli.md) if you want the command-line equivalent of the same workflows.
