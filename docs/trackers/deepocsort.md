# DeepOcSort

[Paper: Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification](https://arxiv.org/abs/2302.11813)

Deep OC-SORT starts from OC-SORT's motion-centric association and adds appearance in a more adaptive way than earlier ReID heuristics. The paper argues that appearance should not dominate all the time, but should be integrated when it is actually helpful, especially under long occlusions and dense interactions. This makes it a stronger tracker than pure OC-SORT when motion cues alone are not enough to keep identities stable.

## What BoxMOT Needs For DeepOcSort

- A detector plus a ReID model.
- AABB detections only in BoxMOT.
- Useful when OC-SORT is close but still loses IDs in crowded scenes where appearance recovery matters.

::: boxmot.trackers.bbox.deepocsort.DeepOcSort
