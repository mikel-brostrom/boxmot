# Python API

Use `boxmot` for the high-level workflow facade, and explicit modules such as `boxmot.detectors`, `boxmot.reid`, and `boxmot.trackers.tracker_zoo` when you want lower-level control.

## High-level facade

Use `Boxmot` when you want the Python equivalent of the CLI with minimal boilerplate:

```python
from boxmot import Boxmot

boxmot = Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
run = boxmot.track(source="video.mp4", save=True)
print(run)

cache = Boxmot().generate(benchmark="mot17-mini")
print(cache.cache_dir)

metrics = boxmot.val(benchmark="mot17-mini")
print(metrics)

tuned = boxmot.tune(benchmark="mot17-mini", n_trials=2)
print(tuned)
```

ReID lifecycle workflows are available from the same facade:

```python
from boxmot import Boxmot

api = Boxmot()
train_result = api.train(
    model="mobilenetv2_x1_0",
    dataset="market1501",
    data_dir="assets/reid-mini",
    device="cpu",
    epochs=5,
    batch_size=16,
)

metrics = api.eval_reid(
    weights=train_result.weights_path,
    model="mobilenetv2_x1_0",
    dataset="market1501",
    data_dir="assets/reid-mini",
    device="cpu",
)
print(metrics)
```

The same facade also exposes `research(...)` for GEPA-backed benchmark optimization, `train(...)` and `eval_reid(...)` for ReID model lifecycle workflows, and `export(...)` for ReID conversion workflows.

Use `.summary`, `.timings`, `.delta_summary`, or `.to_dict()` on returned results when you need structured data instead of the human-readable report.

## Native C++ backends

Use `tracker_backend="cpp"` when the selected tracker has a native backend:

```python
from boxmot import Boxmot

native_track = Boxmot(detector="yolov8n", tracker="bytetrack")
run = native_track.track(source="video.mp4", tracker_backend="cpp")

native_eval = Boxmot(tracker="ocsort")
metrics = native_eval.val(benchmark="mot17", split="ablation", tracker_backend="cpp")
```

Native C++ backends are currently registered for `botsort`, `bytetrack`, `ocsort`, `occluboost`, and `sfsort`.

## Streaming frame results

When you want per-frame access to tracks, detections, and embeddings, iterate the results yourself instead of passing `show=True` or `save=True`:

```python
from boxmot import Boxmot

model = Boxmot(detector="yolov8l.pt", reid="lmbn_n_duke.pt", tracker="occluboost")
results = model.track(source=0)

for frame_result in results:
    tracks = frame_result.tracks          # (M, 8) TrackResults array
    ids    = frame_result.tracks.id       # (M,) track IDs
    confs  = frame_result.tracks.conf     # (M,) confidences
    boxes  = frame_result.tracks.xyxy     # (M, 4) bounding boxes
    dets   = frame_result.detections      # (M, 6) matched detections, aligned to tracks
    embs   = frame_result.embeddings      # (M, D) matched embeddings, aligned to tracks

    print(f"Frame {frame_result.frame_idx}: {len(ids)} tracks")

    frame_result.save_csv("tracks.csv")   # append tracks to CSV
    frame_result.save_vid("output.mp4")   # append frame to video (auto-detects FPS)

    if not frame_result.show():           # display frame, quit on 'q'
        break

frame_result.close_vid()                  # finalize the video file
```

!!! note "Detections and embeddings are track-aligned"
    `frame_result.detections[i]` and `frame_result.embeddings[i]` correspond to `frame_result.tracks[i]`.
    Coasting tracks (no matched detection) have zero-filled rows.
    Use `frame_result.tracks.det_ind` to check which tracks are coasting (`-1`).

!!! warning "Avoid `show=True` / `save=True` when iterating"
    Passing `show=True` or `save=True` to `model.track(...)` consumes the stream
    internally. The returned object will be exhausted, so your `for` loop gets nothing.
    Handle display and saving yourself inside the loop as shown above.

## Composable runtime

If you need more control, compose the detector, ReID runtime, and tracker explicitly:

```python
from boxmot import track
from boxmot.reid import ReID
from boxmot.trackers import StrongSort
from boxmot.detectors import Detector

detector = Detector("yolov8n.pt", device="cpu")
reid = ReID("osnet_x0_25_msmt17.pt", device="cpu")
tracker = StrongSort(reid_weights="osnet_x0_25_msmt17.pt", device="cpu", half=False)

results = track("video.mp4", detector, reid, tracker, verbose=False)
print(results.summary())
```

## Importing trackers directly

Every tracker class is exported from `boxmot.trackers`, so you can import any of them into your own project:

```python
from boxmot.trackers import (
    BoostTrack,
    BotSort,
    ByteTrack,
    DeepOcSort,
    HybridSort,
    OccluBoost,
    OcSort,
    SFSORT,
    StrongSort,
)
```

### Using the tracker factory

The `create_tracker` factory builds a tracker from its string name and loads its default YAML config automatically:

```python
from boxmot.trackers.tracker_zoo import create_tracker

# Motion-only tracker (no ReID model needed)
tracker = create_tracker("bytetrack")

# ReID-aware tracker — pass weights so the factory builds the ReID backend
tracker = create_tracker(
    "botsort",
    reid_weights="osnet_x0_25_msmt17.pt",
    device="cpu",
    half=False,
)
```

### Instantiating a tracker class directly

Import the class and pass parameters yourself for full control:

```python
import numpy as np
from boxmot.trackers import ByteTrack

tracker = ByteTrack(
    track_high_thresh=0.6,
    track_low_thresh=0.1,
    track_buffer=30,
)

# Feed detections frame-by-frame
# dets: (N, 6) array with columns [x1, y1, x2, y2, conf, cls]
# img:  the current frame as a numpy array (H, W, 3)
tracks = tracker.update(dets, img)
```

For ReID-aware trackers, supply a ReID model:

```python
from boxmot.trackers import OccluBoost
from boxmot.reid.core import ReID

reid = ReID(weights="osnet_x0_25_msmt17.pt", device="cpu", half=False)

tracker = OccluBoost(reid_model=reid.model)

tracks = tracker.update(dets, img)

# tracks is a TrackResults array (M, 8) with columns:
# [x1, y1, x2, y2, id, conf, cls, det_ind]
print(tracks.id)    # track IDs
print(tracks.xyxy)  # bounding boxes
print(tracks.conf)  # confidences
```

### Available trackers

| Import name | String key | Uses ReID |
| --- | --- | --- |
| `ByteTrack` | `bytetrack` | No |
| `BotSort` | `botsort` | Yes |
| `StrongSort` | `strongsort` | Yes |
| `OcSort` | `ocsort` | No |
| `DeepOcSort` | `deepocsort` | Yes |
| `HybridSort` | `hybridsort` | Yes |
| `BoostTrack` | `boosttrack` | Yes |
| `OccluBoost` | `occluboost` | Yes |
| `SFSORT` | `sfsort` | No |

!!! tip "Custom config overrides"
    Pass `tracker_config` to `create_tracker` to load a non-default YAML, or
    pass `evolve_param_dict` with a plain dict of parameters to skip YAML
    entirely:

    ```python
    from boxmot.trackers.tracker_zoo import create_tracker

    tracker = create_tracker(
        "ocsort",
        evolve_param_dict={"det_thresh": 0.3, "iou_thresh": 0.2, "max_age": 50},
    )
    ```

## Reference pages

- [High-level API](high-level.md) — `Boxmot` facade, `track(...)`, `evaluate(...)`, and result objects
- [Low-level API](low-level.md) — `Detector`, `ReID`, and the tracker factory
