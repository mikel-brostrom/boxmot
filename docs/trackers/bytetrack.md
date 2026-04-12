<!-- docs/api/trackers/bytetrack.md -->
# ByteTrack

[Paper: ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)

ByteTrack's main idea is simple: do not throw away low-confidence detections too early. The paper shows that a second association pass over lower-score boxes recovers occluded or partially visible objects and reduces fragmented tracks without adding much complexity. In practice, it is one of the strongest motion-only baselines because it stays fast while improving ID continuity.

## What BoxMOT Needs For ByteTrack

- Detector only. ReID features are not required.
- Supports both AABB and OBB detections in BoxMOT.
- Good default when you want a fast, strong baseline and already trust the detector.

::: boxmot.trackers.bytetrack.bytetrack.ByteTrack
