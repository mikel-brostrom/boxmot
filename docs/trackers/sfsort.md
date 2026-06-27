# SFSORT

[Paper: SFSORT: Scene Features-based Simple Online Real-Time Tracker](https://arxiv.org/abs/2404.07553)

SFSORT is designed around speed. The paper removes the Kalman filter entirely, introduces a bounding-box similarity cost, and uses scene-derived cues to keep association strong while minimizing compute. The goal is not to be the most elaborate tracker, but to keep the tracker extremely lightweight and real-time while still remaining competitive on standard MOT benchmarks.

## What BoxMOT Needs For SFSORT

- Detector only. ReID is not required.
- Supports both AABB and OBB detections in BoxMOT.
- Best when throughput matters most and you want a very lightweight online tracker.

## Native C++ Backend

BoxMOT also ships a native C++17 SFSORT implementation under `boxmot/native/cpp/trackers/sfsort/`. It supports:

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
boxmot eval --benchmark mot17 --split ablation --tracker sfsort --tracker-backend cpp
boxmot track --tracker sfsort --tracker-backend cpp --source 0
```

`--tracking-backend cpp` remains available as a compatibility alias for existing benchmark scripts.

::: boxmot.trackers.bbox.sfsort.SFSORT
