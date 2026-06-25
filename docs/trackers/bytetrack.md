<!-- docs/api/trackers/bytetrack.md -->
# ByteTrack

[Paper: ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)

ByteTrack's main idea is simple: do not throw away low-confidence detections too early. The paper shows that a second association pass over lower-score boxes recovers occluded or partially visible objects and reduces fragmented tracks without adding much complexity. In practice, it is one of the strongest motion-only baselines because it stays fast while improving ID continuity.

## What BoxMOT Needs For ByteTrack

- Detector only. ReID features are not required.
- Supports both AABB and OBB detections in BoxMOT.
- Good default when you want a fast, strong baseline and already trust the detector.

## Native C++ Backend

BoxMOT also ships a native C++17 ByteTrack implementation under `boxmot/native/cpp/trackers/bytetrack/`. It supports:

- cached replay for `eval`, `tune`, and `research`
- live `track` through `--tracker-backend cpp`
- both AABB and OBB detection layouts in the native tracker path

Requirements:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

Example:

```bash
boxmot eval --benchmark mot17 --split ablation --tracker bytetrack --tracker-backend cpp
boxmot track --tracker bytetrack --tracker-backend cpp --source 0
```

`--tracking-backend cpp` remains available as a compatibility alias for existing benchmark scripts.

::: boxmot.trackers.bbox.bytetrack.ByteTrack
