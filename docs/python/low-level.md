# Component API Reference

## Canonical structures

Import canonical values from `boxmot.structures`. Their implementation modules
group 2D and 3D values by role:

| Module | Structures |
| --- | --- |
| [Geometry](../reference/boxmot/structures/geometry.md) | `Boxes`, `OrientedBoxes`, `Boxes3D` |
| [Detections](../reference/boxmot/structures/detections.md) | `Detections`, `Detections3D` |
| [Tracks](../reference/boxmot/structures/tracks.md) | `Tracks`, `Tracks3D`, `MultimodalTracks` |
| [Camera](../reference/boxmot/structures/camera.md) | `CameraModel` |

## Detection

::: boxmot.detectors.protocols.Detector

::: boxmot.detectors.protocols.DetectorCapabilities

::: boxmot.detectors.specs.DetectorSpec

::: boxmot.detectors.factory.create_detector

## Segmentation

::: boxmot.segmentors.protocols.Segmentor

::: boxmot.segmentors.specs.SegmentorSpec

::: boxmot.segmentors.factory.create_segmentor

## Appearance encoding

::: boxmot.reid.protocols.AppearanceEncoder

::: boxmot.reid.protocols.EncoderRequirements

::: boxmot.reid.specs.ReIDEncoderSpec

::: boxmot.reid.factory.create_reid_encoder

## Tracking

::: boxmot.trackers.common.protocols.Tracker

::: boxmot.trackers.common.protocols.ReIDConfigurableTracker

::: boxmot.trackers.common.protocols.TrackerRequirements

::: boxmot.trackers.common.specs.TrackerSpec

::: boxmot.trackers.common.factory.create_tracker

Trackers accept canonical `Detections` or, for standalone box-only calls, exact
NumPy AABB `N x 6` / OBB `N x 7` rows. An optional canonical `Frame` may be
passed separately. NumPy input returns packed `float64` AABB `M x 8` / OBB
`M x 9` rows; canonical input returns `Tracks`. Appearance and mask inference
normally belong upstream in a `PerceptionPipeline`. Every high-level
ReID-enabled tracker adapter can also lazily invoke ReID when its input has no
embeddings and a `Frame` is supplied. Attached embeddings bypass that internal
backend. Native adapters then pass the resolved features to their model-free
C++ tracker libraries.
