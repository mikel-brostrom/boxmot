# OcSort

[Paper: Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking](https://arxiv.org/abs/2203.14360)

OC-SORT focuses on a specific failure mode in Kalman-filter trackers: error accumulation during occlusion and non-linear motion. The paper replaces a purely prediction-centric view with an observation-centric one, using detector observations to reconstruct a more reliable virtual trajectory across missed frames. That makes the tracker much more robust than vanilla SORT in crowded scenes while keeping the same simple online structure.

## What BoxMOT Needs For OcSort

- Detector only. ReID is not required.
- Supports both AABB and OBB detections in BoxMOT.
- A strong choice when you want a fast motion-only tracker but expect more non-linear motion or occlusion than ByteTrack handles comfortably.

## Native C++ Backend

BoxMOT ships a native C++17 OCSORT implementation under `boxmot/native/cpp/trackers/ocsort/`. It supports:

- cached replay for `eval` and `tune`
- live `track` through `--tracker-backend cpp`
- both AABB and OBB detection layouts in the native tracker path

Requirements:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

Example:

```bash
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker ocsort --tracker-backend cpp
boxmot track --tracker ocsort --tracker-backend cpp --source 0
```

The native backend honors `asso_func` from
`boxmot/configs/trackers/ocsort.yaml`. AABB and OBB tracking support `iou`,
`giou`, `diou`, `ciou`, `hmiou`, and `centroid`. Centroid association uses the
first live image to initialize and cache the frame dimensions. OBB `ciou` is a
custom experimental aspect-ratio adaptation, while OBB `hmiou` is an
experimental global-y height cue intended only for scenes where image vertical
is meaningful. See the [association function guide](../config/trackers.md#association-function)
for the exact OBB semantics.

::: boxmot.trackers.bbox.ocsort.OcSort
