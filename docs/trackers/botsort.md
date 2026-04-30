# BotSort

[Paper: BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/abs/2206.14651)

BoT-SORT extends the ByteTrack family by combining motion, appearance, and camera-motion compensation more explicitly. The paper improves the Kalman state, uses global motion compensation, and fuses ReID cues with IoU-based association to make identity assignment more stable in crowded scenes and moving-camera footage. The result is a tracker that is still online and practical, but more robust than motion-only alternatives when identities are ambiguous.

## What BoxMOT Needs For BotSort

- A detector and, for the full method, a ReID model.
- Supports both AABB and OBB detections in BoxMOT.
- Best when you need stronger identity preservation than ByteTrack, especially with camera motion or repeated occlusions.

## Native C++ Backend

BoxMOT also ships a native C++17 BoTSORT implementation under `boxmot/native/trackers/botsort/`. It supports:

- cached replay for `eval`, `tune`, and `research`
- live `track` through `--tracker-backend cpp`
- both AABB and OBB detections for live tracking and cached replay
- internal ONNX ReID inference for native live tracking
- automatic `.pt -> .onnx` export for native cpp inference when you pass PyTorch ReID weights

Requirements:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

Example:

```bash
boxmot eval --benchmark mot17-ablation --tracker botsort --tracker-backend cpp
boxmot track --tracker botsort --tracker-backend cpp --reid models/lmbn_n_duke.pt --source 0
```

`--tracking-backend cpp` remains available as a compatibility alias for existing benchmark scripts.

For cached replay, native ONNX ReID inference is used as a fallback when an embedding cache is missing. If the selected ReID model is a `.pt` file, BoxMOT exports it to a native OpenCV-compatible `*_opencv.onnx` cache file and reuses that export for the native run.

For OBB replay, the native runner consumes 8-column OBB caches and writes MMOT-style corner outputs so the native replay stage matches the existing OBB evaluation pipeline.

::: boxmot.trackers.botsort.botsort.BotSort
