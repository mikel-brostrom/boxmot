# Python API

Use `boxmot` for the high-level workflow facade and runtime wrappers, and explicit modules such as `boxmot.trackers.registry` or `boxmot.trackers.bbox` when you want lower-level control.

## High-level facade

Use `BoxMOT` when you want the Python equivalent of the CLI with minimal boilerplate:

```python
from boxmot import BoxMOT

boxmot = BoxMOT(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
run = boxmot.track(source="video.mp4", save=True)
print(run)

cache = BoxMOT().generate(benchmark="mot17-mini")
print(cache.cache_dir)

metrics = boxmot.val(benchmark="mot17-mini")
print(metrics)

tuned = boxmot.tune(benchmark="mot17-mini", n_trials=2)
print(tuned)
```

Component strings have component-specific meanings: detector strings resolve model names or artifacts, ReID strings resolve model names or paths, and tracker strings resolve registered tracker algorithms. Keep component-specific settings grouped so options such as `half` and `max_age` do not become ambiguous:

```python
from boxmot import BoxMOT

model = BoxMOT(
    detector="yolox_x_MOT17_ablation",
    reid="models/lmbn_n_duke.onnx",
    tracker="occluboost",
    detector_kwargs={
        "confidence": 0.25,
        "image_size": 640,
        "half": True,
    },
    reid_kwargs={
        "half": True,
    },
    tracker_kwargs={
        "with_reid": True,
    },
)
```

Tracker selection supports the three public forms:

```python
from boxmot import BoxMOT
from boxmot.trackers import OccluBoost

simple = BoxMOT(tracker="occluboost")

configured = BoxMOT(
    tracker="occluboost",
    tracker_kwargs={"with_reid": True},
)

by_class = BoxMOT(
    tracker=OccluBoost,
    tracker_kwargs={"with_reid": True},
)

tracker = OccluBoost(reid_model=my_reid, with_reid=True)
injected = BoxMOT(tracker=tracker)
```

ReID lifecycle workflows are available from the same facade:

```python
from boxmot import BoxMOT

api = BoxMOT()
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

You can also bind the facade to a ReID weight file and use the same object for
training, export, and direct embedding extraction:

```python
from boxmot import BoxMOT

reid = BoxMOT(reid="models/lmbn_n_duke.pt")
reid.train(cfg="custom_config.yaml")
reid = reid.export(format="onnx", half=True)
embeddings = reid.embed(source="path/to/image.jpg")
```

The same facade also exposes `research(...)` for GEPA-backed benchmark optimization, `train(...)` and `eval_reid(...)` for ReID model lifecycle workflows, `export(...)` for ReID conversion workflows, and `embed(...)` for direct ReID inference.

Use `.summary`, `.timings`, `.delta_summary`, or `.to_dict()` on returned results when you need structured data instead of the human-readable report.

## Native C++ backends

Use `tracker_backend="cpp"` when the selected tracker has a native backend:

```python
from boxmot import BoxMOT

native_track = BoxMOT(detector="yolov8n", tracker="bytetrack")
run = native_track.track(source="video.mp4", tracker_backend="cpp")

native_eval = BoxMOT(tracker="ocsort")
metrics = native_eval.val(benchmark="mot17", split="ablation", tracker_backend="cpp")
```

Native C++ backends are currently registered for `botsort`, `bytetrack`, `ocsort`, `occluboost`, and `sfsort`.

## Streaming frame results

When you want per-frame access to tracks, detections, and embeddings, iterate the results yourself instead of passing `show=True` or `save=True`:

```python
from boxmot import BoxMOT

model = BoxMOT(detector="yolov8l.pt", reid="lmbn_n_duke.pt", tracker="occluboost")
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
import cv2

from boxmot import Detector, ReIDModel
from boxmot.trackers import OccluBoost

image = cv2.imread("image.jpg")
detector = Detector("yolov8n.pt", device="cpu")
reid = ReIDModel("osnet_x0_25_msmt17.pt", device="cpu")
tracker = OccluBoost(reid_model=reid, with_reid=True)

detections = detector.predict(image)
embeddings = reid.embed(image, boxes=detections.xyxy)
tracks = tracker.update(detections, image=image, embeddings=embeddings)
```

## Importing trackers directly

`OccluBoost` is the package-level tracker export:

```python
from boxmot.trackers import OccluBoost
```

Use the registry for string-based construction, or import other concrete tracker classes from `boxmot.trackers.bbox.<name>`.

### Using the tracker factory

The `create_tracker` factory builds a tracker from its string name and loads its default YAML config automatically:

```python
from boxmot.trackers.registry import create_tracker

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
from boxmot.trackers.bbox.bytetrack import ByteTrack

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
from boxmot import ReIDModel

reid = ReIDModel("osnet_x0_25_msmt17.pt", device="cpu", half=False)

tracker = OccluBoost(reid_model=reid, with_reid=True)

embeddings = reid.embed(img, boxes=dets[:, :4])
tracks = tracker.update(dets, image=img, embeddings=embeddings)

# tracks is a TrackResults array (M, 8) with columns:
# [x1, y1, x2, y2, id, conf, cls, det_ind]
print(tracks.id)    # track IDs
print(tracks.xyxy)  # bounding boxes
print(tracks.conf)  # confidences
```

### Available trackers

| Import name | String key | Uses ReID |
| --- | --- | --- |
| `boxmot.trackers.bbox.bytetrack.ByteTrack` | `bytetrack` | No |
| `boxmot.trackers.bbox.botsort.BotSort` | `botsort` | Yes |
| `boxmot.trackers.bbox.strongsort.StrongSort` | `strongsort` | Yes |
| `boxmot.trackers.bbox.ocsort.OcSort` | `ocsort` | No |
| `boxmot.trackers.bbox.deepocsort.DeepOcSort` | `deepocsort` | Yes |
| `boxmot.trackers.bbox.hybridsort.HybridSort` | `hybridsort` | Yes |
| `boxmot.trackers.bbox.boosttrack.BoostTrack` | `boosttrack` | Yes |
| `OccluBoost` | `occluboost` | Yes |
| `boxmot.trackers.bbox.sfsort.SFSORT` | `sfsort` | No |

!!! tip "Custom config overrides"
    Pass `tracker_config` to `create_tracker` to load a non-default YAML, or
    pass `evolve_param_dict` with a plain dict of parameters to skip YAML
    entirely:

    ```python
    from boxmot.trackers.registry import create_tracker

    tracker = create_tracker(
        "ocsort",
        evolve_param_dict={"det_thresh": 0.3, "iou_thresh": 0.2, "max_age": 50},
    )
    ```

## Reference pages

- [High-level API](high-level.md) — `BoxMOT`, `Detector`, `ReIDModel`, explicit workflow helpers, and result objects
- [Low-level API](low-level.md) — `Detector`, `ReID`, and the tracker factory
