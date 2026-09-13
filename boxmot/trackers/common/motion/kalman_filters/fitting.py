"""Direct covariance moment fitting from supervised residuals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

MIN_COVARIANCE_SCALE = 1e-6
MIN_RESIDUAL_EVENTS = 2


@dataclass
class ScalarNoiseMoments:
    """Estimate one covariance multiplier from normalized squared errors."""

    events: int = 0
    components: int = 0
    total: float = 0.0

    def add(self, residual: np.ndarray, variances: np.ndarray) -> None:
        """Accumulate supervised coordinates with positive reference variance."""
        residual, variances = np.asarray(residual), np.asarray(variances)
        valid = np.isfinite(residual) & np.isfinite(variances) & (variances > 0)
        if not np.any(valid):
            return
        self.total += float(np.sum(np.square(residual[valid]) / variances[valid]))
        self.components += int(np.count_nonzero(valid))
        self.events += 1

    def estimate(self, baseline: float) -> dict[str, Any]:
        """Keep the baseline when fewer than two independent events exist."""
        result = {"baseline": baseline, "events": self.events, "components": self.components}
        if self.events < MIN_RESIDUAL_EVENTS:
            return {**result, "value": baseline, "status": "retained", "reason": "insufficient residual events"}
        raw = self.total / self.components
        if not np.isfinite(raw):
            raise ValueError("Calibration residuals produce an unrepresentable covariance scale.")
        return {**result, "value": max(MIN_COVARIANCE_SCALE, raw), "raw_estimate": raw, "status": "fitted"}


@dataclass
class ProcessNoiseMoments:
    """Fit two nonnegative process scales without replaying a filter search.

    Each state coordinate supplies the moment equation
    E[error**2] = position_scale * Q_position[j,j] + velocity_scale * Q_velocity[j,j].
    Consecutive residual pairs also supply signed cross-covariance equations,
    which distinguish position diffusion from velocity diffusion even when
    every frame interval is equal. Normalize by the sum of absolute covariance
    bases to combine mixed state units. Only a 2-by-2 normal matrix is retained,
    independent of the number of observations.
    """

    events: int = 0
    lag_pairs: int = 0
    components: int = 0
    normal: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    target: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def add(self, residual: np.ndarray, position: np.ndarray, velocity: np.ndarray) -> None:
        """Accumulate actual-interval covariance bases and GT prediction errors."""
        residual = np.asarray(residual)
        if self._add_moments(residual, residual, position, velocity):
            self.events += 1

    def add_cross_covariance(
        self,
        first_residual: np.ndarray,
        second_residual: np.ndarray,
        position_basis: np.ndarray,
        velocity_basis: np.ndarray,
    ) -> None:
        """Accumulate signed lag moments without counting another transition.

        Both residuals must describe the same measured coordinates. Their
        product estimates the cross covariance; either covariance basis may
        be negative. For example, independent position diffusion contributes
        a negative lag covariance to consecutive three-position residuals.
        """
        if self._add_moments(first_residual, second_residual, position_basis, velocity_basis):
            self.lag_pairs += 1

    def _add_moments(self, first: np.ndarray, second: np.ndarray, position: np.ndarray, velocity: np.ndarray) -> bool:
        """Accumulate valid moment equations with scale-invariant weights."""
        first, second = np.asarray(first), np.asarray(second)
        basis = np.column_stack((position, velocity))
        reference = np.abs(basis).sum(axis=1)
        valid = np.isfinite(first) & np.isfinite(second) & np.all(np.isfinite(basis), axis=1) & (reference > 0)
        if not np.any(valid):
            return False
        design = basis[valid] / reference[valid, None]
        errors = first[valid] * second[valid] / reference[valid]
        self.normal += design.T @ design
        self.target += design.T @ errors
        self.components += int(np.count_nonzero(valid))
        return True

    def estimate(self, baselines: tuple[float, float]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Solve the two-variable nonnegative quadratic directly, including edges."""
        common = {"events": self.events, "lag_pairs": self.lag_pairs, "components": self.components}
        if not np.all(np.isfinite(self.normal)) or not np.all(np.isfinite(self.target)):
            raise ValueError("Calibration process residuals produce unrepresentable moments.")
        reason = None
        if self.events < MIN_RESIDUAL_EVENTS:
            reason = "insufficient GT transitions"
        elif np.linalg.matrix_rank(self.normal) < 2:
            reason = "process position and velocity scales are not identifiable from these residuals"
        if reason is not None:
            return tuple(
                {**common, "baseline": value, "value": value, "status": "retained", "reason": reason}
                for value in baselines
            )
        unconstrained = np.linalg.solve(self.normal, self.target)
        candidates = [np.zeros(2)]
        if np.all(unconstrained >= 0):
            candidates.append(unconstrained)
        for index in range(2):
            edge = np.zeros(2)
            edge[index] = max(0.0, self.target[index] / self.normal[index, index])
            candidates.append(edge)
        selected = min(candidates, key=lambda values: float(values @ self.normal @ values - 2 * self.target @ values))
        return tuple(
            {
                **common,
                "baseline": baseline,
                "value": max(MIN_COVARIANCE_SCALE, float(value)),
                "raw_estimate": float(value),
                "status": "fitted",
            }
            for baseline, value in zip(baselines, selected, strict=True)
        )
