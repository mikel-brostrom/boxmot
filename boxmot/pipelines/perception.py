"""Batch composition of detection, segmentation, and appearance encoding."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from boxmot.detectors.protocols import Detector, DetectorCapabilities
from boxmot.reid.protocols import AppearanceEncoder, EncoderRequirements
from boxmot.segmentors.protocols import Segmentor
from boxmot.structures import Detections, Frame, MaskBatch
from boxmot.trackers.protocols import TrackerRequirements


@dataclass(frozen=True, slots=True)
class PipelineOutputs:
    """Optional enrichments requested from a perception pipeline."""

    masks: bool = False
    embeddings: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.masks, bool) or not isinstance(self.embeddings, bool):
            raise TypeError("PipelineOutputs values must be bools.")


def _detector_capabilities(detector: Detector | None) -> DetectorCapabilities:
    if detector is None:
        return DetectorCapabilities()
    capabilities = getattr(detector, "capabilities", None)
    if not isinstance(capabilities, DetectorCapabilities):
        raise TypeError("detector.capabilities must be a DetectorCapabilities object.")
    return capabilities


def _declares_output(detector: Detector | None, name: str) -> bool:
    """Read a canonical detector output capability."""
    return getattr(_detector_capabilities(detector), f"provides_{name}")


def _encoder_requirements(encoder: AppearanceEncoder) -> EncoderRequirements:
    requirements = getattr(encoder, "requirements", None)
    if not isinstance(requirements, EncoderRequirements):
        raise TypeError("reid.requirements must be an EncoderRequirements object.")
    return requirements


def _encoder_embedding_dim(encoder: AppearanceEncoder) -> int:
    embedding_dim = encoder.embedding_dim
    if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise TypeError("reid.embedding_dim must be a positive integer.")
    return embedding_dim


def _validate_embedding_tensor(
    values: object,
    *,
    rows: int,
    width: int | None,
    source: str,
) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{source} embeddings must be a torch.Tensor.")
    if values.dtype != torch.float32:
        raise TypeError(f"{source} embeddings must have dtype torch.float32, got {values.dtype}.")
    if values.device.type != "cpu":
        raise ValueError(f"{source} embeddings must be on CPU, got device {values.device}.")
    if values.ndim != 2:
        raise ValueError(f"{source} embeddings must have shape [N, D], got {tuple(values.shape)}.")
    if values.shape[0] != rows or (width is not None and values.shape[1] != width):
        expected = f"({rows}, {width})" if width is not None else f"({rows}, D)"
        raise ValueError(f"{source} embeddings have shape {tuple(values.shape)}; expected {expected}.")
    if values.shape[1] <= 0:
        raise ValueError(f"{source} embeddings must have a positive feature dimension.")
    if not values.is_contiguous():
        raise ValueError(f"{source} embeddings must be contiguous.")
    if values.numel() and not bool(torch.isfinite(values).all()):
        raise ValueError(f"{source} embeddings must contain only finite values.")
    return values


@dataclass(slots=True)
class PerceptionPipeline:
    """Progressively enrich batches of canonical detections."""

    detector: Detector | None = None
    segmentor: Segmentor | None = None
    reid: AppearanceEncoder | None = None
    outputs: PipelineOutputs = PipelineOutputs()

    def __post_init__(self) -> None:
        if self.detector is not None and not isinstance(self.detector, Detector):
            raise TypeError("detector must implement capabilities and predict().")
        if self.detector is not None:
            _detector_capabilities(self.detector)
        if self.segmentor is not None and not isinstance(self.segmentor, Segmentor):
            raise TypeError("segmentor must implement segment().")
        if self.reid is not None and (
            not callable(getattr(self.reid, "encode", None))
            or inspect.getattr_static(self.reid, "embedding_dim", None) is None
            or inspect.getattr_static(self.reid, "requirements", None) is None
        ):
            raise TypeError("reid must implement embedding_dim, requirements, and encode().")
        if self.reid is not None:
            _encoder_requirements(self.reid)
        if not isinstance(self.outputs, PipelineOutputs):
            raise TypeError("outputs must be a PipelineOutputs object.")
        if self.detector is not None:
            self._validate_providers(TrackerRequirements())

    def _validate_providers(self, requirements: TrackerRequirements) -> None:
        needs_embeddings = self.outputs.embeddings or requirements.embeddings
        detector_embeddings = _declares_output(self.detector, "embeddings")
        if needs_embeddings and self.reid is None and not detector_embeddings:
            raise ValueError("Embeddings were requested, but neither the detector nor a ReID encoder provides them.")

        encoder_needs_masks = (
            needs_embeddings and self.reid is not None and _encoder_requirements(self.reid).masks
        )
        needs_masks = self.outputs.masks or requirements.masks or encoder_needs_masks
        if needs_masks and self.segmentor is None and not _declares_output(self.detector, "masks"):
            raise ValueError("Masks were requested, but neither the detector nor a segmentor provides them.")

    def _validate_detector_results(
        self,
        frames: list[Frame],
        results: object,
        *,
        check_capabilities: bool = False,
    ) -> list[Detections]:
        if not isinstance(results, list):
            raise TypeError("Detector.predict() must return a list of Detections.")
        if len(results) != len(frames):
            raise ValueError(f"Detector returned {len(results)} results for {len(frames)} frames.")
        capabilities = _detector_capabilities(self.detector)
        for index, (frame, detections) in enumerate(zip(frames, results)):
            if not isinstance(detections, Detections):
                raise TypeError(f"Detector result {index} is not a Detections object.")
            detections.validate()
            if detections.sample_id != frame.sample_id:
                raise ValueError(
                    f"Detector result {index} has sample_id {detections.sample_id!r}; "
                    f"expected {frame.sample_id!r}."
                )
            if check_capabilities and detections.is_obb and not capabilities.supports_obb:
                raise ValueError(f"Detector result {index} uses OBB geometry that its capabilities do not declare.")
            if check_capabilities and not detections.is_obb and not capabilities.supports_aabb:
                raise ValueError(f"Detector result {index} uses AABB geometry that its capabilities do not declare.")
            if detections.embeddings is not None:
                _validate_embedding_tensor(
                    detections.embeddings,
                    rows=len(detections),
                    width=None,
                    source=f"Detector result {index}",
                )
        return results

    def _attach_masks(
        self,
        frames: list[Frame],
        detections: list[Detections],
        *,
        required: bool,
    ) -> list[Detections]:
        if not required:
            return detections

        missing = [index for index, item in enumerate(detections) if item.masks is None]
        if not missing:
            return detections
        nonempty = [index for index in missing if len(detections[index])]
        attached: dict[int, MaskBatch] = {
            index: MaskBatch(torch.empty((0, frames[index].height, frames[index].width), dtype=torch.bool))
            for index in missing
            if not len(detections[index])
        }
        if nonempty:
            if self.segmentor is None:
                raise ValueError("The detector did not provide required masks and no segmentor is configured.")
            masks = self.segmentor.segment(
                [frames[index] for index in nonempty],
                [detections[index] for index in nonempty],
            )
            if not isinstance(masks, list) or len(masks) != len(nonempty):
                count = len(masks) if isinstance(masks, list) else type(masks).__name__
                raise ValueError(f"Segmentor returned {count} results for {len(nonempty)} frames.")
            for index, mask_batch in zip(nonempty, masks):
                if not isinstance(mask_batch, MaskBatch):
                    raise TypeError("Segmentor results must be MaskBatch objects.")
                if len(mask_batch) != len(detections[index]):
                    raise ValueError(
                        f"Segmentor returned {len(mask_batch)} masks for {len(detections[index])} detections."
                    )
                if mask_batch.image_size != frames[index].image_size:
                    raise ValueError(
                        f"Segmentor mask size {mask_batch.image_size} does not match frame size "
                        f"{frames[index].image_size}."
                    )
                attached[index] = mask_batch
        return [
            item.with_masks(attached[index]) if index in attached else item
            for index, item in enumerate(detections)
        ]

    def _attach_embeddings(
        self,
        frames: list[Frame],
        detections: list[Detections],
        *,
        required: bool,
    ) -> list[Detections]:
        if not required:
            return detections
        missing = [index for index, item in enumerate(detections) if item.embeddings is None]
        if not missing:
            return detections
        nonempty = [index for index in missing if len(detections[index])]
        attached: dict[int, torch.Tensor] = {}
        if nonempty:
            if self.reid is None:
                raise ValueError("The detector did not provide required embeddings and no ReID encoder is configured.")
            embeddings = self.reid.encode(
                [frames[index] for index in nonempty],
                [detections[index] for index in nonempty],
            )
            if not isinstance(embeddings, list) or len(embeddings) != len(nonempty):
                count = len(embeddings) if isinstance(embeddings, list) else type(embeddings).__name__
                raise ValueError(f"ReID encoder returned {count} results for {len(nonempty)} frames.")
            embedding_dim = _encoder_embedding_dim(self.reid)
            for index, values in zip(nonempty, embeddings):
                attached[index] = _validate_embedding_tensor(
                    values,
                    rows=len(detections[index]),
                    width=embedding_dim,
                    source="ReID encoder result",
                )

        if self.reid is None:
            if any(not len(detections[index]) for index in missing):
                raise ValueError("The detector did not declare the embedding width and no ReID encoder is configured.")
        else:
            embedding_dim = _encoder_embedding_dim(self.reid)
            for index in missing:
                if not len(detections[index]):
                    attached[index] = torch.empty((0, embedding_dim), dtype=torch.float32)
            for index, values in attached.items():
                if values.ndim != 2 or values.shape != (len(detections[index]), embedding_dim):
                    raise ValueError(
                        f"ReID encoder returned shape {tuple(values.shape)}; expected "
                        f"({len(detections[index])}, {embedding_dim})."
                    )
        return [
            item.with_embeddings(attached[index]) if index in attached else item
            for index, item in enumerate(detections)
        ]

    @staticmethod
    def _validate_frames(frames: Sequence[Frame]) -> list[Frame]:
        frame_batch = list(frames)
        for index, frame in enumerate(frame_batch):
            if not isinstance(frame, Frame):
                raise TypeError(f"frames[{index}] must be a Frame, not {type(frame).__name__}.")
        return frame_batch

    def enrich(
        self,
        frames: Sequence[Frame],
        detections: Sequence[Detections],
        requirements: TrackerRequirements = TrackerRequirements(),
    ) -> list[Detections]:
        """Attach requested and tracker-required payloads to existing detections."""
        if not isinstance(requirements, TrackerRequirements):
            raise TypeError("requirements must be a TrackerRequirements object.")
        frame_batch = self._validate_frames(frames)
        detection_batch = self._validate_detector_results(frame_batch, list(detections))
        needs_embeddings = self.outputs.embeddings or requirements.embeddings
        encoder_needs_masks = needs_embeddings and self.reid is not None and _encoder_requirements(self.reid).masks
        needs_masks = self.outputs.masks or requirements.masks or encoder_needs_masks
        detection_batch = self._attach_masks(frame_batch, detection_batch, required=needs_masks)
        detection_batch = self._attach_embeddings(
            frame_batch,
            detection_batch,
            required=needs_embeddings,
        )
        for item in detection_batch:
            item.validate()
        return detection_batch

    def process(self, frames: Sequence[Frame]) -> list[Detections]:
        """Detect and enrich an ordered frame batch according to requested outputs."""
        if self.detector is None:
            raise RuntimeError(
                "PerceptionPipeline.process() requires a detector; use enrich() for caller-supplied detections."
            )
        frame_batch = self._validate_frames(frames)
        detections = self._validate_detector_results(
            frame_batch,
            self.detector.predict(frame_batch),
            check_capabilities=True,
        )
        detections = self.enrich(frame_batch, detections, TrackerRequirements())
        for item in detections:
            item.validate()
        return detections


__all__ = ("PerceptionPipeline", "PipelineOutputs")
