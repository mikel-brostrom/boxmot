# OcSort

[Paper: Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking](https://arxiv.org/abs/2203.14360)

OC-SORT focuses on a specific failure mode in Kalman-filter trackers: error accumulation during occlusion and non-linear motion. The paper replaces a purely prediction-centric view with an observation-centric one, using detector observations to reconstruct a more reliable virtual trajectory across missed frames. That makes the tracker much more robust than vanilla SORT in crowded scenes while keeping the same simple online structure.

## What BoxMOT Needs For OcSort

- Detector only. ReID is not required.
- Supports both AABB and OBB detections in BoxMOT.
- A strong choice when you want a fast motion-only tracker but expect more non-linear motion or occlusion than ByteTrack handles comfortably.

::: boxmot.trackers.ocsort.ocsort.OcSort
