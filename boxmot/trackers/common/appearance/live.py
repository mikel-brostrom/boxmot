"""Shared tracker-owned live appearance inference."""

from __future__ import annotations

import inspect

import numpy as np
import torch

from boxmot.components.timing import timed_component_call, timed_component_phase
from boxmot.reid.protocols import AppearanceEncoder, EncoderRequirements
from boxmot.reid.specs import ReIDConfig, ReIDEncoderSpec
from boxmot.structures import Boxes, Detections, Frame, OrientedBoxes

_REID_OPTION_UNSET = object()


class LiveReIDMixin:
    """Resolve supplied or tracker-generated detection embeddings.

    Consumers initialize the mixin from their constructor, expose
    ``accepts_embeddings``, ``use_embeddings``, and ``is_obb``, then call
    :meth:`_resolve_input_embeddings` at their validated public update
    boundary. Model configuration survives sequence resets; only the observed
    embedding width and first-update guard are sequence-local.
    """

    accepts_embeddings = False
    use_embeddings = False
    is_obb = False

    def _init_live_reid(
        self,
        *,
        reid: ReIDConfig | AppearanceEncoder | None = _REID_OPTION_UNSET,
    ) -> None:
        """Validate and retain lazy ReID construction settings."""

        if not self.accepts_embeddings and reid is not _REID_OPTION_UNSET:
            raise TypeError(f"{self.__class__.__name__} does not accept ReID configuration: reid.")
        reid = None if reid is _REID_OPTION_UNSET else reid
        self._reid_explicit = reid is not None
        self._reid_config: ReIDConfig | ReIDEncoderSpec | None = None
        self._reid_encoder: AppearanceEncoder | None = None
        if self.accepts_embeddings:
            if reid is None or isinstance(reid, ReIDConfig):
                self._reid_config = reid if reid is not None else ReIDConfig()
            else:
                # Inspect the dimension declaration without evaluating a lazy
                # property or constructing a model just to validate its protocol.
                if (
                    not callable(getattr(reid, "encode", None))
                    or inspect.getattr_static(reid, "embedding_dim", None) is None
                    or not isinstance(getattr(reid, "requirements", None), EncoderRequirements)
                ):
                    raise TypeError("reid must be a ReIDConfig or an AppearanceEncoder exposing encode().")
                self._reid_encoder = reid
        self._reset_live_reid_sequence()

    @property
    def generates_embeddings(self) -> bool:
        """Return whether this tracker can fill missing embeddings."""

        return bool(getattr(self, "accepts_embeddings", False) and self.use_embeddings)

    @property
    def reid_requires_masks(self) -> bool:
        """Whether the configured live appearance encoder consumes instance masks."""
        return bool(self.generates_embeddings and self._get_live_reid_encoder().requirements.masks)

    def _validate_detection_inputs(self, detections: Detections) -> None:
        """Reject supplied channels unused by the algorithm or its live encoder."""
        if detections.embeddings is not None and not self.requirements.embeddings:
            raise ValueError(f"{self.__class__.__name__} does not use detection embeddings in this configuration.")
        accepts_masks = getattr(getattr(self, "capabilities", None), "accepts_masks", self.requirements.masks)
        if detections.masks is not None and not accepts_masks:
            encoder_uses_masks = detections.embeddings is None and self.generates_embeddings
            # Empty provider batches must advance lifecycle without constructing
            # a lazy encoder merely to inspect its mask requirements.
            if encoder_uses_masks and (len(detections) > 0 or self._reid_encoder is not None):
                encoder_uses_masks = self.reid_requires_masks
            if not encoder_uses_masks:
                raise ValueError(f"{self.__class__.__name__} does not use detection masks in this configuration.")

    def configure_reid(self, spec: ReIDEncoderSpec) -> None:
        """Configure a full encoder spec for lazy tracker-owned inference.

        Configuration is intentionally separate from :class:`TrackerSpec`,
        whose options describe only the tracking algorithm. The encoder is
        constructed on the first non-empty update without supplied embeddings.
        """

        if not isinstance(spec, ReIDEncoderSpec):
            raise TypeError(f"spec must be a ReIDEncoderSpec, got {type(spec).__name__}.")
        if not self.generates_embeddings:
            raise ValueError(f"{self.__class__.__name__} is not configured to use embeddings.")
        if self._has_updated:
            raise RuntimeError("ReID must be configured before the first update of a sequence.")
        if self._reid_explicit or self._reid_encoder is not None:
            raise RuntimeError("ReID is already configured for this tracker.")
        self._reid_config = spec
        self._reid_explicit = True

    def _get_live_reid_encoder(self) -> AppearanceEncoder:
        """Return the encoder lazily constructed from the configured spec."""

        if self._reid_encoder is None:
            if self._reid_config is None:
                raise RuntimeError("No ReID configuration is available for this tracker.")
            from boxmot.reid.factory import create_reid_encoder

            with timed_component_phase("reid", "process", device=self._reid_config.device):
                self._reid_encoder = create_reid_encoder(self._reid_config)
        return self._reid_encoder

    def _record_embedding_width(self, width: int) -> None:
        """Record one stable appearance width for the current sequence."""

        if self.last_emb_size is not None and width != self.last_emb_size:
            raise ValueError(f"Embedding width changed from {self.last_emb_size} to {width}.")
        self.last_emb_size = width

    @staticmethod
    def _frame_to_bgr(frame: Frame | np.ndarray) -> np.ndarray:
        """Return OpenCV BGR pixels, converting canonical RGB frames as needed."""

        if isinstance(frame, np.ndarray):
            return frame
        return frame.image.permute(1, 2, 0).flip(-1).contiguous().numpy()

    def _resolve_input_embeddings(
        self,
        *,
        geometry: np.ndarray,
        embeddings: np.ndarray | None,
        frame: Frame | np.ndarray | None,
        detections: Detections | None = None,
        scores: np.ndarray | None = None,
        class_ids: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Resolve supplied embeddings or generate them from the current frame."""

        if embeddings is not None:
            if self.generates_embeddings:
                self._record_embedding_width(int(embeddings.shape[1]))
            return embeddings
        if not self.generates_embeddings:
            return None
        if len(geometry) == 0:
            return np.empty((0, self.last_emb_size or 1), dtype=np.float32)
        if frame is None:
            raise ValueError(f"{self.__class__.__name__} requires a frame to generate detection embeddings.")

        timing_device = self._reid_config.device if self._reid_config is not None else None
        if isinstance(frame, np.ndarray):
            # Packed input is HWC BGR; canonical encoders receive CHW RGB and
            # matching sample identity, just as they do in TrackingPipeline.
            with timed_component_phase("reid", "preprocess", device=timing_device):
                frame = Frame(
                    image=torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)[::-1])),
                    sample_id=detections.sample_id if detections is not None else "tracker:frame",
                )
        encoder_detections = detections
        if encoder_detections is None:
            geometry_values = torch.from_numpy(np.ascontiguousarray(geometry, dtype=np.float32))
            canonical_geometry = OrientedBoxes(geometry_values) if self.is_obb else Boxes(geometry_values)
            score_values = (
                np.ones(len(geometry), dtype=np.float32)
                if scores is None
                else np.ascontiguousarray(scores, dtype=np.float32)
            )
            class_id_values = (
                np.zeros(len(geometry), dtype=np.int64)
                if class_ids is None
                else np.ascontiguousarray(class_ids, dtype=np.int64)
            )
            encoder_detections = Detections(
                geometry=canonical_geometry,
                scores=torch.from_numpy(score_values),
                class_ids=torch.from_numpy(class_id_values),
                sample_id=frame.sample_id,
            )
        encoder = self._get_live_reid_encoder()
        if encoder.requirements.masks and encoder_detections.masks is None:
            raise ValueError(f"{self.__class__.__name__} live ReID requires full-frame detection masks.")
        with timed_component_call("reid", device=timing_device):
            encoded = encoder.encode((frame,), (encoder_detections,))
        if not isinstance(encoded, list) or len(encoded) != 1:
            raise ValueError("ReID encoder must return one embedding tensor for one frame.")
        if not isinstance(encoded[0], torch.Tensor):
            raise TypeError("ReID encoder outputs must be torch.Tensor objects.")
        values = encoded[0]
        width = encoder.embedding_dim
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise TypeError("reid.embedding_dim must be a positive integer.")
        if values.dtype != torch.float32:
            raise TypeError(f"ReID encoder embeddings must have dtype torch.float32, got {values.dtype}.")
        if values.device.type != "cpu":
            raise ValueError(f"ReID encoder embeddings must be on CPU, got {values.device}.")
        if values.ndim != 2 or values.shape != (len(geometry), width):
            raise ValueError(
                f"ReID encoder returned embeddings with shape {tuple(values.shape)}; "
                f"expected ({len(geometry)}, {width})."
            )
        if not values.is_contiguous():
            raise ValueError("ReID encoder embeddings must be contiguous.")
        raw_features = values.detach().numpy()

        with timed_component_phase("reid", "postprocess", device=timing_device):
            features = np.asarray(raw_features, dtype=np.float32)
            if not np.isfinite(features).all():
                raise ValueError("ReID backend returned non-finite embeddings.")
            norms = np.linalg.norm(features.astype(np.float64), axis=1, keepdims=True)
            if np.any(norms <= 1e-12):
                raise ValueError("ReID backend returned a zero-norm embedding.")
            features = np.asarray(features / norms, dtype=np.float32)
            self._record_embedding_width(int(features.shape[1]))
            return np.ascontiguousarray(features)

    def _mark_live_reid_updated(self) -> None:
        """Prevent ReID reconfiguration after the first sequence update."""

        self._has_updated = True

    def _reset_live_reid_sequence(self) -> None:
        """Clear sequence-local ReID validation state while preserving models."""

        self._has_updated = False
        self.last_emb_size = None


__all__ = ("LiveReIDMixin",)
