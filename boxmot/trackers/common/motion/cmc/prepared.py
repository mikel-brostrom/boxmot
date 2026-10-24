"""Sequence-local consumption of previously validated camera transforms."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from boxmot.trackers.common.motion.cmc.base import BaseCMC


class PreparedCMC(BaseCMC):
    """Apply prepared affine transforms through the ordinary tracker CMC path.

    The caller selects each delivered frame explicitly, starting at zero.
    Applying a transform never advances the frame, so class-separated updates
    can consume the same camera motion repeatedly. Source provenance and the
    sample identities belong to the workflow supplying these transforms.
    """

    requires_frame_pixels = False

    def __init__(self, warps: np.ndarray) -> None:
        matrices = np.array(warps, dtype=np.float32, copy=True)
        if matrices.ndim != 3 or matrices.shape[1:] != (2, 3):
            raise ValueError("Prepared CMC requires affine transforms with shape (N, 2, 3).")
        if not np.isfinite(matrices).all() or any(not self.is_valid_transform(matrix) for matrix in matrices):
            raise ValueError("Prepared CMC transforms must be finite and nonsingular.")
        if len(matrices) and not np.array_equal(matrices[0], np.eye(2, 3, dtype=np.float32)):
            raise ValueError("Prepared CMC must start with an identity transform.")
        matrices.flags.writeable = False
        self._warps = matrices
        self.reset()

    def select_frame(self, frame_index: int) -> None:
        """Select the next delivered frame without permitting skipped transforms."""
        if isinstance(frame_index, (bool, np.bool_)) or not isinstance(frame_index, Integral):
            raise TypeError("Prepared CMC frame index must be an integer.")
        if frame_index != self._frame_index + 1 or frame_index >= len(self._warps):
            raise ValueError(f"Prepared CMC expected delivered frame {self._frame_index + 1}, got {frame_index}.")
        if self._frame_index >= 0 and not self._applied:
            raise RuntimeError("The previous prepared CMC transform was not applied.")
        self._frame_index = int(frame_index)
        self._applied = False

    def apply(self, img: np.ndarray | None = None, dets: np.ndarray | None = None) -> np.ndarray:
        """Return the selected transform without reading pixels or estimating motion."""
        if self._frame_index < 0:
            raise RuntimeError("Select a delivered frame before applying prepared CMC.")
        self._applied = True
        return self._warps[self._frame_index]

    def validate_complete(self) -> None:
        """Require every prepared frame to have reached the tracker CMC path."""
        if self._frame_index + 1 != len(self._warps) or (len(self._warps) and not self._applied):
            raise RuntimeError("Prepared CMC replay did not consume every delivered frame.")

    def reset(self) -> None:
        """Restart sequence selection while retaining immutable transforms."""
        self._frame_index = -1
        self._applied = False
