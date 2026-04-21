# SFSORT

[Paper: SFSORT: Scene Features-based Simple Online Real-Time Tracker](https://arxiv.org/abs/2404.07553)

SFSORT is designed around speed. The paper removes the Kalman filter entirely, introduces a bounding-box similarity cost, and uses scene-derived cues to keep association strong while minimizing compute. The goal is not to be the most elaborate tracker, but to keep the tracker extremely lightweight and real-time while still remaining competitive on standard MOT benchmarks.

## What BoxMOT Needs For SFSORT

- Detector only. ReID is not required.
- Supports both AABB and OBB detections in BoxMOT.
- Best when throughput matters most and you want a very lightweight online tracker.

::: boxmot.trackers.sfsort.sfsort.SFSORT
