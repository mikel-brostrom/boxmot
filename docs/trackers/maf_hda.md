# MafHda

[Paper: Online Multi-Object Tracking and Segmentation with GMPHD Filter and Mask-based Affinity Fusion](https://arxiv.org/abs/2009.00100)

MAF_HDA combines Gaussian-mixture motion estimates with masked correlation-filter
appearance scores. Hierarchical association first matches segments to tracks,
then reconnects new tracklets to lost tracks. Mask merging reduces duplicate
instances. The BoxMOT implementation is a Python port of the
[authors' GMPHD_MAF source](https://github.com/SonginCV/GMPHD_MAF).

## What BoxMOT needs

- Axis-aligned boxes with a nonempty, full-frame instance mask for each detection.
- The current image on every update, including frames without detections. The
  tracker extracts masked HOG and Lab features for its correlation filters.
- Canonical `Detections` input, since packed NumPy box rows cannot carry masks.
- No ReID weights or additional model downloads. Generate masks upstream with
  an instance segmentation detector or a standalone segmentor.

The tracker supports class-separated tracking through `per_class=True` and
returns canonical `Tracks` with row-aligned masks. OBB geometry and a native C++
backend are not supported. Its implementation lives under
`boxmot/trackers/multimodal/maf_hda` because both geometry and masks are
fundamental to its state and appearance model.

## Python example

This complete example uses one synthetic instance. Replace the image, boxes,
scores, classes, and masks with each frame's perception results in a real loop.

```python
import torch

from boxmot import create_tracker
from boxmot.structures import Boxes, Detections, Frame, MaskBatch
from boxmot.trackers import TrackerSpec

tracker = create_tracker(TrackerSpec(name="maf_hda", per_class=True))

image = torch.zeros((3, 96, 128), dtype=torch.uint8)  # RGB CHW
image[:, 24:72, 32:64] = 180
masks = torch.zeros((1, 96, 128), dtype=torch.bool)
masks[0, 24:72, 32:64] = True

frame = Frame(
    image=image,
    sample_id="example:0",
    sequence_id="example",
    frame_index=0,
)
detections = Detections(
    geometry=Boxes(torch.tensor([[32, 24, 64, 72]], dtype=torch.float32)),
    scores=torch.tensor([0.95], dtype=torch.float32),
    class_ids=torch.tensor([0], dtype=torch.int64),
    masks=MaskBatch(masks),
    sample_id=frame.sample_id,
)

tracks = tracker.update(detections, frame)
print(tracks.track_ids, tracks.masks.values.shape)
tracker.reset()
```

For an empty frame, pass an empty `Detections` with a `MaskBatch` of shape
`(0, height, width)` and the current `Frame`. Import `MafHda` directly from
`boxmot` when constructing the class without the factory.

## Configuration and source

Defaults and tuning ranges are in `boxmot/configs/trackers/maf_hda.yaml`.
The defaults follow the upstream MOTS20 training preset. `s2ta_mode` and
`t2ta_mode` choose motion, appearance, or fused (`maf`) affinity independently
for segment-to-track and track-to-track association. `asso_func` selects the
overlap affinity used in association and gating; its default is `iou`.

`max_age` bounds how long lost tracks can be recovered. Outputs contain only
currently observed tracks with at least `min_hits` detections. When
`min_hits > 1`, confirmation begins on the current frame; the online API does
not delay output or backfill earlier frames as the original implementation did.
Candidate appearance scoring does not modify track memory. Filters are refreshed
from accepted observations, without the source's additional training on an
unmatched prediction. Numerical and benchmark parity with the C++ executable has
not been established.

The implementation package retains the upstream BSD 2-Clause license and the
appearance code's third-party notices. It runs through BoxMOT's online update
interface and does not require the original Windows project or an installed
`MAF_HDA` package.

::: boxmot.MafHda
