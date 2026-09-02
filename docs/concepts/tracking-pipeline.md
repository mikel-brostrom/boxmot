# Tracking Pipeline

This page shows the main data flow from input frames to output track rows for the Python and native C++ tracker backends.

## Python live tracking

Use this path when the tracker backend is `python`, which is the default for `boxmot track` and `BoxMOT.track(...)`.

```text
User / CLI / Python API
        |
        v
boxmot track / BoxMOT.track(...)
        |
        v
run_track(...)
        |
        +--> build_detector_from_spec(...)       -> Detector
        +--> build_tracker_from_spec(...)        -> Python tracker
        +--> build_tracker_with_reid_spec(...)   -> ReID, tracker adapter, or None
        |
        v
Results(source, detector, reid, tracker)
        |
        v
for each frame from iter_source(source):
        |
        +--> Detector
        |       |
        |       +--> preprocess
        |       +--> process
        |       +--> postprocess
        |       |
        |       v
        |   detections
        |   AABB: (N, 6) = x1, y1, x2, y2, conf, cls
        |   OBB:  (N, 7) = cx, cy, w, h, angle, conf, cls
        |
        +--> sanitize_detections(detections, masks, image_shape)
        |       |
        |       +--> drop non-finite or invalid geometry rows
        |       +--> keep detection masks aligned with retained rows
        |       |
        |       v
        |   sanitized detections and masks
        |
        +--> optional ReID
        |       |
        |       +--> preprocess crops / boxes
        |       +--> process embeddings
        |       +--> postprocess features
        |       |
        |       v
        |   embeddings or None
        |
        +--> tracker.update(dets, img[, embs])
        |       |
        |       +--> select AABB or OBB layout from detection shape
        |       +--> predict existing tracks
        |       +--> associate detections to tracks
        |       +--> update matched tracks
        |       +--> create, keep, mark lost, or remove tracks
        |       |
        |       v
        |   tracks
        |   AABB: (N, 8) = x1, y1, x2, y2, id, conf, cls, det_ind
        |   OBB:  (N, 9) = cx, cy, w, h, angle, id, conf, cls, det_ind
        |
        v
FrameResult(frame_idx, frame, tracks, detections, embeddings, masks)
        |
        +--> render / show
        +--> save video
        +--> save txt as MOT / MMOT rows
        +--> summary and timing stats
```

## Native C++ live tracking from BoxMOT

Use this path when BoxMOT still owns the source, detector, output handling, and Python API, but the tracker implementation is native C++ through `--tracker-backend cpp`.

```text
User selects native tracker backend
        |
        +--> boxmot track --tracker-backend cpp
        +--> BoxMOT(..., tracker="bytetrack").track(..., tracker_backend="cpp")
        |
        v
build_tracker_from_spec(...)
        |
        +--> parse tracker name and backend
        +--> get_native_live_backend(tracker)
        +--> ensure_<tracker>_cpp_library()
        +--> load <tracker>_capi shared library with ctypes
        +--> create Native<Tracker>Tracker wrapper
        |
        v
Results loop stays in Python
        |
        +--> iter_source(source)
        +--> Python detector -> detections
        +--> optional ReID
        |       |
        |       +--> motion-only trackers: skipped
        |       +--> native ReID trackers: handled inside C++ when configured
        |       +--> fallback: external Python ReID features when needed
        |
        v
Native<Tracker>Tracker.update(dets, img[, embs])
        |
        +--> normalize numpy detections and uint8 image
        +--> validate 6-column AABB or 7-column OBB detections
        +--> call C ABI update function
        |
        v
<tracker>/src/c_api.cpp
        |
        +--> ConvertLiveDetections(...)
        +--> WrapLiveImage(...)
        +--> <tracker>::Tracker.Update(detections, image)
        +--> WriteLiveOutputs(...)
        |
        v
numpy tracks returned to Python
        |
        +--> AABB: (N, 8)
        +--> OBB:  (N, 9)
        |
        v
same FrameResult rendering, saving, and summary path as Python
```

## Standalone C++ embedding

Use this path when your own C++ program links directly against a native tracker target such as `bytetrack_core`.

```text
Your C++ application
        |
        +--> read frame / camera input
        +--> run your detector
        +--> optionally run your ReID model
        +--> create <tracker>::Config
        +--> instantiate <tracker>::Tracker
        |
        v
for each frame:
        |
        +--> fill vector of <tracker>::Detection
        |       |
        |       +--> AABB: xyxy, conf, cls, det_ind
        |       +--> OBB:  is_obb=true, xywha, conf, cls, det_ind
        |       +--> optional embedding for ReID-aware trackers
        |
        +--> tracker.Update(detections, frame)
        |       |
        |       +--> predict
        |       +--> associate
        |       +--> update track state
        |       +--> manage track lifecycle
        |
        v
vector of <tracker>::TrackOutput
        |
        +--> render, write, stream, or use tracks in your application
```

## Cached benchmark tracking

`eval`, `tune`, and `research` run tracking from cached detections and
embeddings. The detector and ReID stages can be generated once. `eval` and
`tune` can replay them with Python or native C++ trackers; `research` evaluates
editable Python tracker code.

```text
eval / tune / research
        |
        v
generate cache if needed
        |
        +--> DetectorReIDPipeline
        +--> detector outputs
        +--> ReID embeddings
        |       |
        |       +--> effective producer: python or cpp
        |       +--> model format + runtime + optional artifact hash
        |       +--> preprocessing + crop schema version
        +--> runs/dets_n_embs/<dataset>/<split>/<detector>/
                |
                +--> dets/<sequence>.npy
                +--> embs/<python|cpp>/
                      <model>-<format>-<runtime>[-wHASH]/
                      <preprocess>-cropvN/<sequence>.npy
        |
        v
run_generate_mot_results(...)
        |
        +--> tracker_backend == "python"
        |       |
        |       +--> process/thread replay workers
        |       +--> load cached detections and embeddings
        |       +--> Python tracker.update(...)
        |       +--> write MOT / MMOT result txt
        |
        +--> tracker_backend == "cpp" (eval / tune)
                |
                +--> get_native_replay_backend(tracker)
                +--> ensure_<tracker>_cpp_executable()
                +--> launch <tracker>_replay
                +--> C++ LoadSequence(...)
                +--> slice cached detections per frame
                +--> <tracker>::Tracker.Update(...)
                +--> write MOT / MMOT result txt
        |
        v
optional postprocessing
        |
        v
MOT metrics and workflow summary
```

The embedding producer is the implementation that computed the descriptor, not
the tracker algorithm that later consumes it. Native tracker selection normally
requests the C++ producer for ReID-aware trackers. If the native adapter cannot
be imported, generation selects the Python producer and stores the output in the
Python bucket; errors after the C++ producer is selected are reported instead of
being silently reclassified. Trackers may share cached embeddings when the
producer, model artifact, runtime, preprocessing, and crop version all match.

Compatible legacy caches with flat `embs/<model>/<preprocess>/` paths may be
reused only when trusted and when their embedding rows align exactly with the
cached detection rows. All newly generated embeddings use the producer-first
layout.

## Related pages

- [Detection Layouts](index.md)
- [Python API](../python/index.md)
- [Native C++ Integration](../native/index.md)
