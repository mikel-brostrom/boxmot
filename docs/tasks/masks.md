# Mask Tracking

Masks are canonical full-frame, detection-aligned `MaskBatch` values with
`bool[N,H,W]` CPU-contiguous storage.

```bash
boxmot track \
  --detector yolo11n-seg.pt \
  --geometry aabb \
  --tracker sam2mot \
  --source video.mp4 \
  --save
```

Sam2Mot consumes masks supplied upstream; it never instantiates SAM 2 inside
the tracker. Generic SAM and SAM 2 inference belongs to `boxmot.segmentors`;
the multimodal tracker owns only tracking lifecycle, association, and
tracker-specific state. Every non-empty input mask must contain foreground.

```python
from boxmot.structures import MaskBatch

enriched = detections.with_masks(
    MaskBatch(full_frame_masks_bool_cpu.contiguous())
)
tracks = tracker.update(enriched, frame=frame)
```

Sam2Mot emits full-frame masks aligned to track rows, including propagated
tracks. Other renderers prefer `Tracks.masks`; if absent, they map detection
masks through each nonnegative `detection_indices` value. A coasting row with
index `-1` has no current detection mask.

Standalone segmentors receive `(frames, detections)` and preserve detection
order. Requested empty outputs are present `bool[0,H,W]` tensors rather than
`None`, and empty inputs avoid model computation.
