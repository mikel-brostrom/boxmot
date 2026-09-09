# Mask Tracking

Masks are canonical full-frame, detection-aligned `MaskBatch` values with
`bool[N,H,W]` CPU-contiguous storage.

```bash
boxmot track \
  --detector yolo11n-seg.pt \
  --geometry aabb \
  --tracker maf_hda \
  --per-class \
  --source video.mp4 \
  --save
```

[MafHda](../trackers/maf_hda.md) uses AABB detections, their masks, and the current
image to combine motion with masked correlation-filter appearance. Each detection
mask must contain foreground. Supply the image on every update, including frames
with no detections; OBB geometry is not supported by this tracker.

Masks can come from the detector or an upstream segmentor. Generic SAM and SAM 2
inference belongs to `boxmot.segmentors`.

```python
from boxmot.structures import MaskBatch

enriched = detections.with_masks(
    MaskBatch(full_frame_masks_bool_cpu.contiguous())
)
tracks = tracker.update(enriched, frame=frame)
```

MafHda emits currently observed, confirmed tracks with aligned full-frame masks.
Lost tracks retain state for later recovery and are not emitted on missing
observations. Renderers prefer `Tracks.masks`; if absent, they map detection
masks through each nonnegative `detection_indices` value. A coasting row from a
box tracker with index `-1` has no current detection mask.

Standalone segmentors receive `(frames, detections)` and preserve detection
order. Requested empty outputs are present `bool[0,H,W]` tensors rather than
`None`, and empty inputs avoid model computation.
