# Tracking Tasks

BoxMOT tracks detector outputs through time. The task determines the geometry
and optional per-object data entering the tracker; the selected tracker decides
how those observations are associated.

## Supported tasks

| Task | Detection input | Track output | Start here |
| --- | --- | --- | --- |
| Axis-aligned boxes | `Boxes` with `float32[N,4]` `xyxy` | `Tracks` with AABB geometry | [Bounding Boxes](aabb.md) |
| Oriented boxes | `OrientedBoxes` with `float32[N,5]` `cxcywha` | `Tracks` with OBB geometry | [Oriented Boxes](obb.md) |
| Instance masks | Full-frame `MaskBatch` aligned to detections | Optional `MaskBatch` aligned to tracks | [Masks](masks.md) |

Angles are expressed in radians. `N` is the number of detections and `M` is
the number of emitted tracks; the values can differ when a tracker is coasting
through a missed detection.

## Shared workflow

Every task follows the same pipeline:

```text
Frame -> detector -> optional segmentor/ReID -> tracker -> PipelineResult
```

The resolved tracker has a fixed AABB or OBB mode. Segmentation masks stay
aligned through immutable selection and permutation. Use
`sam2mot` when masks should influence association rather than only accompany
and visualize box tracks.

## Pose detectors

An Ultralytics pose model can provide its detected boxes to the tracking
pipeline. BoxMOT currently tracks those boxes; keypoints are not part of the
canonical `Tracks` contract.

## Related pages

- [Track mode](../modes/track.md)
- [Choose a tracker](../compare/index.md)
- [Detection layouts](../concepts/index.md)
- [Tracking pipeline](../concepts/tracking-pipeline.md)
