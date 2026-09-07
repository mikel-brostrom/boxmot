"""Shared tracker-owned live appearance inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from boxmot.components.timing import timed_component_phase
from boxmot.reid.protocols import AppearanceEncoder
from boxmot.reid.specs import ReIDEncoderSpec
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
        reid_model: Any | None = _REID_OPTION_UNSET,
        reid_weights: str | Path | list[str | Path] | tuple[str | Path, ...] | None = _REID_OPTION_UNSET,
        device: Any = _REID_OPTION_UNSET,
        half: bool = _REID_OPTION_UNSET,
        reid_preprocess: str | None = _REID_OPTION_UNSET,
    ) -> None:
        """Validate and retain lazy ReID construction settings."""

        accepts_embeddings = bool(getattr(self, "accepts_embeddings", False))
        reid_options = (
            ("reid_model", reid_model),
            ("reid_weights", reid_weights),
            ("device", device),
            ("half", half),
            ("reid_preprocess", reid_preprocess),
        )
        provided_reid_options = [name for name, value in reid_options if value is not _REID_OPTION_UNSET]
        if not accepts_embeddings and provided_reid_options:
            names = ", ".join(provided_reid_options)
            raise TypeError(f"{self.__class__.__name__} does not accept ReID model options: {names}.")

        reid_model = None if reid_model is _REID_OPTION_UNSET else reid_model
        reid_weights = None if reid_weights is _REID_OPTION_UNSET else reid_weights
        device = "cpu" if device is _REID_OPTION_UNSET else device
        half = False if half is _REID_OPTION_UNSET else half
        reid_preprocess = None if reid_preprocess is _REID_OPTION_UNSET else reid_preprocess
        if isinstance(reid_weights, str) and not reid_weights.strip():
            raise ValueError("reid_weights must not be empty.")
        if isinstance(reid_weights, (list, tuple)):
            if not reid_weights:
                raise ValueError("reid_weights must not be empty.")
            if any(not isinstance(value, (str, Path)) for value in reid_weights):
                raise TypeError("reid_weights entries must be strings or Paths.")
            if any(isinstance(value, str) and not value.strip() for value in reid_weights):
                raise ValueError("reid_weights entries must not be empty.")
        elif reid_weights is not None and not isinstance(reid_weights, (str, Path)):
            raise TypeError("reid_weights must be a string, Path, sequence of those values, or None.")
        if reid_model is not None and not callable(getattr(reid_model, "get_features", None)):
            raise TypeError("reid_model must expose get_features(boxes, image).")
        if not isinstance(half, bool):
            raise TypeError("half must be bool.")
        lazy_reid_configured = reid_weights is not None or str(device) != "cpu" or half or reid_preprocess is not None
        if reid_model is not None and lazy_reid_configured:
            raise ValueError(
                "reid_model cannot be combined with reid_weights, a non-CPU device, half=True, or reid_preprocess."
            )

        if accepts_embeddings:
            self._reid_model = reid_model
            self._reid_weights = reid_weights
            self._reid_device = device
            self._reid_half = half
            self._reid_preprocess = reid_preprocess
            self._raw_reid_configured = reid_model is not None or lazy_reid_configured
            self._reid_encoder_spec: ReIDEncoderSpec | None = None
            self._reid_encoder: AppearanceEncoder | None = None
        self._reset_live_reid_sequence()

    @property
    def generates_embeddings(self) -> bool:
        """Return whether this tracker can fill missing embeddings."""

        return bool(getattr(self, "accepts_embeddings", False) and self.use_embeddings)

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
        if self._reid_encoder_spec is not None or self._reid_encoder is not None:
            raise RuntimeError("ReID is already configured for this tracker.")
        if self._raw_reid_configured or self._reid_model is not None:
            raise ValueError("A ReIDEncoderSpec cannot be combined with direct ReID model configuration.")
        self._reid_encoder_spec = spec

    def _get_live_reid_encoder(self) -> AppearanceEncoder:
        """Return the encoder lazily constructed from the configured spec."""

        if self._reid_encoder_spec is None:
            raise RuntimeError("No ReIDEncoderSpec is configured.")
        if self._reid_encoder is None:
            from boxmot.reid.factory import create_reid_encoder

            with timed_component_phase("reid", "process", device=self._reid_encoder_spec.device):
                self._reid_encoder = create_reid_encoder(self._reid_encoder_spec)
        return self._reid_encoder

    def _get_live_reid_model(self) -> Any:
        """Return the injected or lazily constructed ReID inference backend."""

        if self._reid_encoder_spec is not None:
            raise RuntimeError("A configured ReIDEncoderSpec must be used through its appearance encoder.")
        if self._reid_model is None:
            from boxmot.reid.core import ReID

            self._reid_model = ReID(
                weights=self._reid_weights,
                device=self._reid_device,
                half=self._reid_half,
                preprocess_name=self._reid_preprocess,
            ).model
        return self._reid_model

    def _record_embedding_width(self, width: int) -> None:
        """Record one stable appearance width for the current sequence."""

        if self.last_emb_size is not None and width != self.last_emb_size:
            raise ValueError(f"Embedding width changed from {self.last_emb_size} to {width}.")
        self.last_emb_size = width

    @staticmethod
    def _frame_to_bgr(frame: Frame) -> np.ndarray:
        """Convert one canonical RGB frame to contiguous OpenCV BGR."""

        return frame.image.permute(1, 2, 0).flip(-1).contiguous().numpy()

    def _resolve_input_embeddings(
        self,
        *,
        geometry: np.ndarray,
        embeddings: np.ndarray | None,
        frame: Frame | None,
        detections: Detections | None = None,
        scores: np.ndarray | None = None,
        class_ids: np.ndarray | None = None,
        prepared_bgr: np.ndarray | None = None,
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

        timing_device = self._reid_device
        if self._reid_encoder_spec is not None:
            timing_device = self._reid_encoder_spec.device
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
            encoded = self._get_live_reid_encoder().encode((frame,), (encoder_detections,))
            if not isinstance(encoded, list) or len(encoded) != 1:
                raise ValueError("ReID encoder must return one embedding tensor for one frame.")
            if not isinstance(encoded[0], torch.Tensor):
                raise TypeError("ReID encoder outputs must be torch.Tensor objects.")
            raw_features = encoded[0].detach().cpu().numpy()
        else:
            image = prepared_bgr
            if image is None:
                with timed_component_phase("reid", "preprocess", device=timing_device):
                    image = self._frame_to_bgr(frame)
            with timed_component_phase("reid", "process", device=timing_device):
                raw_features = self._get_live_reid_model().get_features(geometry, image)

        with timed_component_phase("reid", "postprocess", device=timing_device):
            features = np.asarray(raw_features, dtype=np.float32)
            if features.ndim == 1 and len(geometry) == 1:
                features = features.reshape(1, -1)
            if features.ndim != 2 or features.shape[0] != len(geometry):
                raise ValueError(
                    f"ReID backend returned embeddings with shape {features.shape} for {len(geometry)} detections."
                )
            if features.shape[1] == 0:
                raise ValueError("ReID backend returned embeddings with no feature columns.")
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
