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

::: boxmot.trackers.protocols.TrackerRequirements

::: boxmot.trackers.specs.TrackerSpec

::: boxmot.trackers.factory.create_tracker

Trackers accept canonical `Detections` and optional `Frame` values. Appearance
and mask inference belong upstream in a `PerceptionPipeline`; tracker code does
not invoke either model.
