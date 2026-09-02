from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class TrackerInputCapabilities:
    """Inputs consumed by one initialized tracker instance."""

    uses_img: bool
    uses_embs: bool
    supports_masks: bool


class TrackerInputAdapter:
    """Route canonical tracker inputs without retrying tracker execution.

    BoxMOT trackers advertise their actual input use through ``uses_img``,
    ``uses_embs``, and ``supports_masks``. Signature inspection is retained as
    a compatibility fallback for external tracker implementations that do not
    expose those attributes.
    """

    def __init__(self, tracker: Any) -> None:
        self.tracker = tracker
        self._signature, self._signature_available = self._inspect_update_signature()
        fallback = self._capabilities_from_signature()
        self.capabilities = TrackerInputCapabilities(
            uses_img=self._declared_capability("uses_img", fallback.uses_img),
            uses_embs=self._declared_capability("uses_embs", fallback.uses_embs),
            supports_masks=self._declared_capability("supports_masks", fallback.supports_masks),
        )
        self._img_keyword = self._resolve_img_keyword()
        self._pass_img_positionally = self._resolve_positional_img()

    @property
    def uses_img(self) -> bool:
        return self.capabilities.uses_img

    @property
    def uses_embs(self) -> bool:
        return self.capabilities.uses_embs

    @property
    def supports_masks(self) -> bool:
        return self.capabilities.supports_masks

    def requires_image(
        self,
        dets: np.ndarray,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> bool:
        """Return whether this exact update consumes an image."""
        requirement = getattr(self.tracker, "requires_image", None)
        if callable(requirement):
            return bool(requirement(dets=dets, embs=embs, masks=masks))
        return self.uses_img

    def _inspect_update_signature(self) -> tuple[inspect.Signature | None, bool]:
        update = getattr(self.tracker, "update", None)
        if update is None:
            return None, False
        try:
            return inspect.signature(update), True
        except (TypeError, ValueError):
            return None, False

    def _declared_capability(self, name: str, fallback: bool) -> bool:
        value = getattr(self.tracker, name, None)
        if value is None:
            return fallback
        return bool(value)

    def _capabilities_from_signature(self) -> TrackerInputCapabilities:
        if not self._signature_available or self._signature is None:
            # Legacy/native callables without an inspectable signature normally
            # use ``update(dets, img)``. Do not speculate about extra inputs.
            return TrackerInputCapabilities(uses_img=True, uses_embs=False, supports_masks=False)

        params = self._signature.parameters
        accepts_kwargs = any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values())
        accepts_args = any(param.kind is inspect.Parameter.VAR_POSITIONAL for param in params.values())
        positional = [
            param
            for param in params.values()
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        second_positional_is_img = len(positional) >= 2 and positional[1].name not in {"embs", "masks"}
        uses_img = accepts_args or "img" in params or "image" in params or second_positional_is_img
        return TrackerInputCapabilities(
            uses_img=uses_img,
            uses_embs="embs" in params or accepts_kwargs,
            supports_masks="masks" in params or accepts_kwargs,
        )

    def _resolve_img_keyword(self) -> str:
        if self._signature is not None:
            for name in ("img", "image"):
                param = self._signature.parameters.get(name)
                if param is not None and param.kind is inspect.Parameter.KEYWORD_ONLY:
                    return name
        return "img"

    def _resolve_positional_img(self) -> bool:
        if not self._signature_available or self._signature is None:
            return True
        params = tuple(self._signature.parameters.values())
        if any(param.kind is inspect.Parameter.VAR_POSITIONAL for param in params):
            return True
        positional = [
            param
            for param in params
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) >= 2:
            return positional[1].name not in {"embs", "masks"}
        return False

    def update(
        self,
        dets: np.ndarray,
        img: np.ndarray | None = None,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> Any:
        """Call the tracker once with only the inputs it consumes."""
        kwargs: dict[str, Any] = {}
        routed_embs = embs if self.uses_embs else None
        routed_masks = masks if self.supports_masks else None
        if routed_embs is not None:
            kwargs["embs"] = routed_embs
        if routed_masks is not None:
            kwargs["masks"] = routed_masks

        if not self.requires_image(dets=dets, embs=routed_embs, masks=routed_masks):
            return self.tracker.update(dets, **kwargs)
        if self._pass_img_positionally:
            return self.tracker.update(dets, img, **kwargs)

        kwargs[self._img_keyword] = img
        return self.tracker.update(dets, **kwargs)


__all__ = ("TrackerInputAdapter", "TrackerInputCapabilities")
