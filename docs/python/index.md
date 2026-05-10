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

The same facade also exposes `research(...)` for GEPA-backed benchmark optimization and `export(...)` for ReID conversion workflows.

Use `.summary`, `.timings`, `.delta_summary`, or `.to_dict()` on returned results when you need structured data instead of the human-readable report.

## Native C++ backends

Use `tracker_backend="cpp"` or an inline tracker spec such as `"bytetrack:cpp"` when the selected tracker has a native backend:

```python
from boxmot import Boxmot

native_track = Boxmot(detector="yolov8n", tracker="bytetrack:cpp")
run = native_track.track(source="video.mp4")

native_eval = Boxmot(tracker="ocsort")
metrics = native_eval.val(benchmark="mot17-ablation", tracker_backend="cpp")
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

## Reference pages

- [High-level API](high-level.md) — `Boxmot` facade, `track(...)`, `evaluate(...)`, and result objects
- [Low-level API](low-level.md) — `Detector`, `ReID`, and the tracker factory
