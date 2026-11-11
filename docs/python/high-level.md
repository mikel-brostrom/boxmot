# Public API Reference

The supported Python surface is split by ownership. Heavy model and native
runtimes remain lazy until their factories—or a ReID-enabled tracker adapter's
live embedding path—are invoked.

## Structures

::: boxmot.structures.Frame

::: boxmot.structures.Boxes

::: boxmot.structures.OrientedBoxes

::: boxmot.structures.MaskBatch

::: boxmot.structures.Detections

::: boxmot.structures.Tracks

## Sensor fusion structures

Use these values with [EagerMot](../trackers/eagermot.md#sensor-fusion-example).
Image and spatial observations are independent batches; output collections
share track IDs rather than row positions.

::: boxmot.structures.Boxes3D

::: boxmot.structures.Detections3D

::: boxmot.structures.CameraModel

::: boxmot.structures.Tracks3D

::: boxmot.structures.MultimodalTracks

## Pipelines

::: boxmot.pipelines.PerceptionPipeline

::: boxmot.pipelines.TrackingPipeline

::: boxmot.pipelines.PipelineOutputs

::: boxmot.pipelines.PipelineResult

## Dataset loader

::: boxmot.datasets.DatasetSample

::: boxmot.datasets.CachedVisionDataset
