# Component API Reference

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

::: boxmot.trackers.protocols.Tracker

::: boxmot.trackers.protocols.ReIDConfigurableTracker

::: boxmot.trackers.protocols.TrackerRequirements

::: boxmot.trackers.specs.TrackerSpec

::: boxmot.trackers.factory.create_tracker

Trackers accept canonical `Detections` or, for standalone box-only calls, exact
NumPy AABB `N x 6` / OBB `N x 7` rows. An optional canonical `Frame` may be
passed separately. NumPy input returns packed `float64` AABB `M x 8` / OBB
`M x 9` rows; canonical input returns `Tracks`. Appearance and mask inference
normally belong upstream in a `PerceptionPipeline`. Every high-level
ReID-enabled tracker adapter can also lazily invoke ReID when its input has no
embeddings and a `Frame` is supplied. Attached embeddings bypass that internal
backend. Native adapters then pass the resolved features to their model-free
C++ tracker libraries.
