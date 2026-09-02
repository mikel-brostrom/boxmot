# Python API

Use `boxmot` for the high-level workflow facade and runtime wrappers, and explicit modules such as `boxmot.trackers.registry` or `boxmot.trackers.bbox` when you want lower-level control.

## High-level facade

Use `BoxMOT` when you want the Python equivalent of the CLI with minimal boilerplate:

```python
from boxmot import BoxMOT

boxmot = BoxMOT(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
run = boxmot.track(source="video.mp4", save=True, fps=30)
print(run)
print(run.setup_timings)

cache = BoxMOT().generate(experiment="mot17-mini-train-yolox-lmbn")
print(cache.cache_dir)

metrics = boxmot.val(experiment="mot17-mini-train-yolox-lmbn")
print(metrics)

tuned = boxmot.tune(experiment="mot17-mini-train-yolox-lmbn", n_trials=2)
print(tuned)
```

`fps` is an optional positive-integer override for saved video. Omit it to use
the source video's frame rate; live sources use the 30 FPS fallback.

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

Tracker selection accepts a string key, a registered tracker class, or an
initialized tracker instance. A string can use built-in defaults or
`tracker_kwargs` overrides:

```python
from boxmot import BoxMOT, ReIDModel
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

my_reid = ReIDModel("osnet_x0_25_msmt17.pt", device="cpu")
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
train_result = reid.train(cfg="custom_config.yaml")
export_result = reid.export(format="onnx", half=True)
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
metrics = native_eval.val(experiment="mot17-ablation-yolox-lmbn", tracker_backend="cpp")
```

Native C++ backends are currently registered for `botsort`, `bytetrack`, `ocsort`, `occluboost`, and `sfsort`.

Native replay is supported by `val(...)` and `tune(...)`. The current
`research(...)` workflow evaluates Python tracker code and does not forward
native backend selectors. Native live trackers also do not support
`per_class=True`.

## Streaming frame results

When you want per-frame access to tracks, detections, and embeddings, iterate the results yourself instead of passing `show=True` or `save=True`:

```python
from boxmot import BoxMOT

model = BoxMOT(detector="yolov8l.pt", reid="lmbn_n_duke.pt", tracker="occluboost")
results = model.track(source=0)

for frame_result in results:
    tracks = frame_result.tracks          # (M, 8) AABB or (M, 9) OBB TrackResults
    ids    = frame_result.tracks.id       # (M,) track IDs
    confs  = frame_result.tracks.conf     # (M,) confidences
    boxes  = frame_result.tracks.xyxy     # (M, 4) AABBs (enclosing AABBs in OBB mode)
    obbs   = frame_result.tracks.xywha    # (M, 5) in OBB mode
    dets   = frame_result.detections      # (M, 6/7) matched detections, aligned to tracks
    embs   = frame_result.embeddings      # (M, D) matched embeddings, aligned to tracks
    masks  = frame_result.masks           # (M, H, W) aligned/refined masks, or None

    print(f"Frame {frame_result.frame_idx}: {len(ids)} tracks")

    frame_result.save_csv("tracks.csv")   # append tracks to CSV
    frame_result.save_vid("output.mp4")   # append frame to video (auto-detects FPS)

    if not frame_result.show():           # display frame, quit on 'q'
        break

frame_result.close_vid()                  # finalize the video file
```

!!! note "Detections, embeddings, and masks are track-aligned"
    `frame_result.detections[i]` and `frame_result.embeddings[i]` correspond to `frame_result.tracks[i]`.
    Coasting tracks (no matched detection) have zero-filled rows.
    Use `frame_result.tracks.det_ind` to check which tracks are coasting (`-1`).
    When masks are available, `frame_result.masks[i]` is the detector-aligned or
    tracker-refined mask for the same output row.

!!! warning "Know when the facade consumes the stream"
    A live source such as a camera index or URL remains lazy when `show` and
    `save` are false. Finite file and directory sources are consumed before the
    facade returns so their summary is immediately complete. Passing
    `show=True`, `save=True`, or `save_txt=True` also consumes results internally.
    For lazy iteration over a finite source, compose explicit components and use
    `boxmot.api.functional.track(...)`.

## Composable runtime

If you need more control, compose the detector, ReID runtime, and tracker explicitly:

```python
import cv2

from boxmot import Detector, ReIDModel
from boxmot.trackers import OccluBoost

img = cv2.imread("image.jpg")
detector = Detector("yolov8n.pt", device="cpu")
reid = ReIDModel("osnet_x0_25_msmt17.pt", device="cpu")
tracker = OccluBoost(reid_model=reid, with_reid=True)

detections = detector.predict(img)
embs = reid.embed(img, boxes=detections.boxes)  # xyxy for AABB, xywha for OBB
tracks = tracker.update(detections, img=img, embs=embs)
```

`detections.xyxy` always returns axis-aligned geometry; in OBB mode it is the
enclosing AABB. Use `detections.boxes` or `detections.xywha` when extracting
orientation-aware ReID crops.

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
    track_thresh=0.6,
    min_conf=0.1,
    track_buffer=30,
)

# Feed detections frame-by-frame. ByteTrack is motion-only, so no image is needed.
# dets: (N, 6) array with columns [x1, y1, x2, y2, conf, cls]
tracks = tracker.update(dets)
```

Every tracker exposes the same `update(dets, img=None, embs=None, masks=None)`
interface, but the engine only supplies inputs the selected tracker consumes:

| Tracker | Image | Embeddings | Masks |
| --- | --- | --- | --- |
| ByteTrack | Never used | Not used | Not used |
| OCSort | Required initially when centroid association must infer frame dimensions; otherwise not used | Not used | Not used |
| SFSORT | Only needed to infer frame margins when unequal central/marginal timeouts are configured without frame dimensions | Not used | Not used |
| BotSort, BoostTrack, OccluBoost | Needed when CMC is enabled, or when live ReID must compute missing embeddings | Used when ReID is enabled | Not used |
| DeepOCSort, HybridSort | Needed for CMC or live ReID; also required initially when centroid association must infer frame dimensions | Used when ReID is enabled | Not used |
| StrongSort | Required by CMC | Used; precomputed values avoid live ReID extraction | Not used |
| Sam2Mot | Required for image-to-mask coordinate scaling | Not used | Optional; bbox matching is the fallback |

Precomputed embeddings let a ReID-aware tracker run without image pixels when
its CMC path is disabled. If live ReID or CMC is active, omitting `img` raises a
focused input error.

For ReID-aware trackers, supply a ReID model:

```python
from boxmot.trackers import OccluBoost
from boxmot import ReIDModel

reid = ReIDModel("osnet_x0_25_msmt17.pt", device="cpu", half=False)

tracker = OccluBoost(reid_model=reid, with_reid=True)

embs = reid.embed(img, boxes=dets[:, :4])
tracks = tracker.update(dets, img=img, embs=embs)

# tracks is a TrackResults array (M, 8) with columns:
# [x1, y1, x2, y2, id, conf, cls, det_ind]
print(tracks.id)    # track IDs
print(tracks.xyxy)  # bounding boxes
print(tracks.conf)  # confidences
```

### Available trackers

| Import name | String key | Uses ReID | Uses masks |
| --- | --- | --- | --- |
| `boxmot.trackers.bbox.bytetrack.ByteTrack` | `bytetrack` | No | No |
| `boxmot.trackers.bbox.botsort.BotSort` | `botsort` | Yes | No |
| `boxmot.trackers.bbox.strongsort.StrongSort` | `strongsort` | Yes | No |
| `boxmot.trackers.bbox.ocsort.OcSort` | `ocsort` | No | No |
| `boxmot.trackers.bbox.deepocsort.DeepOcSort` | `deepocsort` | Yes | No |
| `boxmot.trackers.bbox.hybridsort.HybridSort` | `hybridsort` | Yes | No |
| `boxmot.trackers.bbox.boosttrack.BoostTrack` | `boosttrack` | Yes | No |
| `boxmot.trackers.bbox.occluboost.OccluBoost` | `occluboost` | Yes | No |
| `boxmot.trackers.bbox.sfsort.SFSORT` | `sfsort` | No | No |
| `boxmot.trackers.hybrid.sam2mot.Sam2Mot` | `sam2mot` | No | Yes |

!!! tip "Custom config overrides"
    Pass `tracker_config` to `create_tracker` to load a non-default YAML, or
    pass `evolve_param_dict` with a plain dict of parameters to skip YAML
    entirely:

    ```python
    from boxmot.trackers.registry import create_tracker

    tracker = create_tracker(
        "ocsort",
        evolve_param_dict={"det_thresh": 0.3, "iou_threshold": 0.2, "max_age": 50},
    )
    ```

## Reference pages

- [High-level API](high-level.md) — `BoxMOT`, `Detector`, `ReIDModel`, explicit workflow helpers, and result objects
- [Low-level API](low-level.md) — `Detector`, `Detections`, `ReID`, the tracker factory, and `TrackResults`
