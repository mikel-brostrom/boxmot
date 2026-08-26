# BotSort

[Paper: BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/abs/2206.14651)

BoT-SORT extends the ByteTrack family by combining motion, appearance, and camera-motion compensation more explicitly. The paper improves the Kalman state, uses global motion compensation, and fuses ReID cues with IoU-based association to make identity assignment more stable in crowded scenes and moving-camera footage. The result is a tracker that is still online and practical, but more robust than motion-only alternatives when identities are ambiguous.

## What BoxMOT Needs For BotSort

- A detector and, for the full method, a ReID model.
- Supports both AABB and OBB detections in BoxMOT.
- Best when you need stronger identity preservation than ByteTrack, especially with camera motion or repeated occlusions.

## Native C++ Backend

BoxMOT also ships a native C++17 BoTSORT implementation under `boxmot/native/cpp/trackers/botsort/`. It supports:

- cached replay for `eval` and `tune`
- live `track` through `--tracker-backend cpp`
- both AABB and OBB detections for live tracking and cached replay
- ReID inference through the shared native `OnnxReIdModel` for both live tracking and cache generation (no Python ONNXRuntime in the loop)
- automatic `.pt -> .onnx` export for native cpp inference when you pass PyTorch ReID weights

Requirements:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

Example:

```bash
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker botsort --tracker-backend cpp
boxmot track --tracker botsort --tracker-backend cpp --reid models/lmbn_n_duke.pt --source 0
```

For cached `eval` and `tune` workflows, `--tracking-backend cpp`
remains available as a compatibility alias. Live `track` uses
`--tracker-backend cpp`.

When `--tracker-backend cpp` is set, cached replay requests the native C++ ReID
producer. Cache paths record the effective embedding producer, so native and
Python results live under distinct `embs/cpp/` and `embs/python/` buckets.
Native initialization or model-loading failures are surfaced instead of silently
changing producer. If the native ReID module cannot be imported, backend
resolution selects the Python producer first and therefore uses `embs/python/`.
The producer is independent of the BoT-SORT algorithm, so
trackers can share a bucket when their model, runtime, preprocessing, and crop
semantics match. See [Native C++ Integration](../native/index.md#embedding-cache-layout)
for the full layout and the runtime knobs (`BOXMOT_REID_BACKEND`,
`BOXMOT_REID_DEVICE`).

For OBB replay, the native runner consumes 8-column OBB caches and writes MMOT-style corner outputs so the native replay stage matches the existing OBB evaluation pipeline.

::: boxmot.trackers.bbox.botsort.BotSort
