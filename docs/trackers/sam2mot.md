# Sam2Mot

[Paper: SAM2MOT: A Novel Paradigm of Multi-Object Tracking by Segmentation](https://arxiv.org/abs/2504.04519)

SAM2MOT puts segmentation at the center of multi-object tracking. The paper
combines mask-driven tracking with a trajectory manager for adding and removing
objects and a cross-object interaction module for handling occlusion. This makes
mask continuity a primary association signal instead of treating segmentation
as a visual add-on to bounding-box tracks.

Its implementation lives under `boxmot/trackers/multimodal/sam2mot` because
box geometry and full-frame masks are both fundamental tracking
representations. The `multimodal` package is an internal organization boundary;
construct the tracker through `create_tracker` or import `boxmot.Sam2Mot`.

## What BoxMOT Needs For Sam2Mot

- Detections carrying one row-aligned full-frame mask per detection. Masks may
  be detector-native or attached by a standalone segmentor upstream.
- No ReID model. The BoxMOT implementation consumes masks passed to `update()`
  and does not instantiate SAM 2 itself, so masks can come from any compatible
  segmentation model.
- Supports both AABB and OBB detections. Mask operations use enclosing AABBs,
  while oriented geometry is retained for association and 9-column OBB output.
- Best when reliable instance masks are available and overlap or occlusion makes
  box-only association ambiguous.

BoxMOT combines geometry and mask IoU in two-stage association, applies
cross-object interaction handling, and performs a third recovery stage for
objects classified as having left the frame. Returned masks stay row-aligned
with the emitted tracks.

Generic SAM and SAM 2 model loading, prompting, and inference remain in
`boxmot.segmentors`. Only tracking lifecycle, association, and tracker-owned
state belong to Sam2Mot.

::: boxmot.Sam2Mot
