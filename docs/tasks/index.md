# Tracking Tasks

Choose a main tracking task, then a variant to match the observations you have.
The task defines the tracked geometry and outputs; the variant determines which
cues the tracker uses to associate observations over time.

## Main tasks and variants

| Main task | Common variants | Track output |
| --- | --- | --- |
| [2D boxes (AABB)](aabb.md) | [Motion-only](aabb.md#motion-only); [AABB + ReID](aabb.md#appearance-assisted-reid) | Axis-aligned boxes and track IDs |
| [Oriented boxes (OBB)](obb.md) | [Motion-only](obb.md#motion-only); [OBB + ReID](obb.md#appearance-assisted-reid) | Oriented boxes and track IDs |
| [Masks](masks.md) | AABB + masks with [masked appearance](masks.md#motion-and-masked-appearance) or [motion-only association](masks.md#motion-only-association) | Axis-aligned boxes, instance masks, and track IDs |
| [2D/3D fusion](fusion.md) | Calibrated 2D + 3D detections, with or without ego poses | Separate image and spatial tracks with shared IDs |

ReID supplies appearance embeddings for association. Both variants of a box
task retain the same geometry and output format. Choose an appearance-assisted
tracker to use ReID; adding an encoder does not make a motion-only tracker
consume its embeddings. See [every tracker's tasks and variants](../trackers/index.md#tasks-and-variants)
for the available algorithms and their defaults.

MafHda uses masks during tracking; its default appearance cue is a masked
correlation filter. Other box pipelines can retain masks for display or export,
and supported Python AABB trackers can use [temporal mask guidance](masks.md#use-temporal-masks-in-association).
These are additional options on the box tasks.

## Data contracts

AABB detections use `Boxes` with `float32[N,4]` `xyxy`; OBB detections use
`OrientedBoxes` with `float32[N,5]` `cxcywha`, with angles in radians.
Instance masks use full-frame `bool[N,H,W]` `MaskBatch` values aligned to the
detections. Sensor fusion accepts independent `Detections` and `Detections3D`
batches plus a `CameraModel` for calibration.

Canonical image tracking returns `Tracks`; sensor fusion returns
`MultimodalTracks`. Standalone box-only NumPy AABB6 or OBB7 input returns a
C-contiguous `float64` AABB8 or OBB9 matrix. EagerMot returns separate image
and spatial arrays for NumPy input. See each task page for the full contract.

## Shared workflow

Image tracking uses this pipeline:

```text
Frame -> detector -> optional upstream segmentor/ReID
      -> tracker (optional private ReID) -> PipelineResult
```

The resolved tracker has a fixed AABB or OBB mode. Segmentation masks stay
aligned through immutable selection and permutation. Use `maf_hda` with AABB
detections, nonempty full-frame masks, and current images for association using
motion and masked correlation-filter appearance. A ReID-enabled tracker adapter
can extract missing embeddings privately from the frame; attached upstream
embeddings bypass that path. Native adapters pass generated or attached
embeddings through the typed ABI to their model-free C++ tracker libraries.

Sensor fusion uses EagerMot's extended `update()` API or the saved-sensor
`eval` and `tune` commands. The image `TrackingPipeline` does not accept 3D
detections or calibration. See [2D/3D fusion](fusion.md).

## Pose detectors

An Ultralytics pose model can provide its detected boxes to the tracking
pipeline. BoxMOT currently tracks those boxes; keypoints are not part of the
canonical `Tracks` contract.

## Related pages

- [Track mode](../modes/track.md)
- [Choose a tracker](../guides/comparison.md)
- [Detection layouts](../concepts/index.md)
- [Tracking pipeline](../concepts/tracking-pipeline.md)
