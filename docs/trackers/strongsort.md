# StrongSort

[Paper: StrongSORT: Make DeepSORT Great Again](https://arxiv.org/abs/2202.13514)

StrongSORT revisits DeepSORT and shows that a stronger baseline matters. The paper improves the detector and appearance encoder, adds better motion handling and camera compensation, and then layers on lightweight postprocessing ideas to recover missed links and detections. The core message is that a carefully engineered DeepSORT-style tracker can remain competitive without changing the online MOT formulation.

## What BoxMOT Needs For StrongSort

- A detector plus a ReID model. Appearance cues are central to this tracker.
- AABB detections only in BoxMOT.
- Good when appearance matching matters more than raw speed, especially for pedestrian-style MOT benchmarks.

::: boxmot.trackers.bbox.strongsort.StrongSort
