"""Streaming supervised covariance moments, without prediction-objective search.

Scalar fits support the 3D model. The 2D filters fit each native coordinate,
coupling process equations only when camera transforms mix those coordinates.
Confidence-aware measurement fits retain the same nonnegative moment solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

MIN_COVARIANCE_SCALE = 1e-6
MIN_RESIDUAL_EVENTS = 2


def _normalized_squared_errors(residual: np.ndarray, variances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select usable coordinates and express squared errors in reference units."""
    residual, variances = np.asarray(residual), np.asarray(variances)
    valid = np.isfinite(residual) & np.isfinite(variances) & (variances > 0)
    return valid, np.square(residual[valid]) / variances[valid]


def _process_equations(
    first: np.ndarray, second: np.ndarray, basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize variance or signed lag equations without mixing their units."""
    first, second = np.asarray(first), np.asarray(second)
    reference = np.abs(basis).sum(axis=1)
    valid = np.isfinite(first) & np.isfinite(second) & np.all(np.isfinite(basis), axis=1) & (reference > 0)
    design = basis[valid] / reference[valid, None]
    errors = first[valid] * second[valid] / reference[valid]
    return valid, design, errors


def _confidence_design(confidence: float) -> np.ndarray:
    """Validate one score and form the shared intercept/slope covariance basis."""
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("Calibration confidence must be finite and between zero and one.")
    return np.array([1.0, (1.0 - confidence) ** 2])


def _nonnegative_pair(normal: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve a full-rank two-variable nonnegative least-squares normal equation."""
    unconstrained = np.linalg.solve(normal, target)
    candidates = [np.zeros(2)]
    if np.all(unconstrained >= 0):
        candidates.append(unconstrained)
    for index in range(2):
        edge = np.zeros(2)
        edge[index] = max(0.0, target[index] / normal[index, index])
        candidates.append(edge)
    return min(candidates, key=lambda values: float(values @ normal @ values - 2 * target @ values))


@dataclass
class ScalarNoiseMoments:
    """Estimate one covariance multiplier from normalized squared errors."""

    events: int = 0
    components: int = 0
    total: float = 0.0

    def add(self, residual: np.ndarray, variances: np.ndarray) -> None:
        """Accumulate supervised coordinates with positive reference variance."""
        valid, errors = _normalized_squared_errors(residual, variances)
        if not np.any(valid):
            return
        self.total += float(np.sum(errors))
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
        valid, design, errors = _process_equations(first, second, np.column_stack((position, velocity)))
        if not np.any(valid):
            return False
        self.normal += design.T @ design
        self.target += design.T @ errors
        self.components += int(np.count_nonzero(valid))
        return True

    def estimate(
        self, baselines: tuple[float, float], *, has_velocity: bool = True
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Solve the two-variable nonnegative quadratic directly, including edges."""
        common = {"events": self.events, "lag_pairs": self.lag_pairs, "components": self.components}
        if not np.all(np.isfinite(self.normal)) or not np.all(np.isfinite(self.target)):
            raise ValueError("Calibration process residuals produce unrepresentable moments.")
        reason = None
        if self.events < MIN_RESIDUAL_EVENTS:
            reason = "insufficient GT transitions"
        elif not has_velocity and self.normal[0, 0] > 0.0:
            raw = max(0.0, float(self.target[0] / self.normal[0, 0]))
            return (
                {
                    **common,
                    "baseline": baselines[0],
                    "value": max(MIN_COVARIANCE_SCALE, raw),
                    "raw_estimate": raw,
                    "status": "fitted",
                },
                {
                    **common,
                    "baseline": baselines[1],
                    "value": baselines[1],
                    "status": "excluded",
                    "reason": "coordinate has no velocity state",
                },
            )
        elif np.linalg.matrix_rank(self.normal) < 2:
            reason = "process position and velocity scales are not identifiable from these residuals"
        if reason is not None:
            return tuple(
                {**common, "baseline": value, "value": value, "status": "retained", "reason": reason}
                for value in baselines
            )
        selected = _nonnegative_pair(self.normal, self.target)
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


class CoupledProcessNoiseMoments:
    """Fit native Q diagonals jointly when camera transforms mix state axes.

    The caller supplies one projected covariance basis per native coordinate.
    Variance and signed lag-covariance equations use the same normalization as
    ``ProcessNoiseMoments``. Storage is bounded by the number of state axes.
    """

    def __init__(self, size: int) -> None:
        self.normal = np.zeros((size, size))
        self.target = np.zeros(size)
        self.events = 0
        self.lag_pairs = 0
        self.components = np.zeros(size, dtype=int)
        self.coordinate_events = np.zeros(size, dtype=int)

    def add(self, residual: np.ndarray, bases: np.ndarray) -> None:
        """Accumulate projected variance bases shaped (parameter, residual)."""
        self._add_moments(residual, residual, bases, cross_covariance=False)

    def add_cross_covariance(self, first: np.ndarray, second: np.ndarray, bases: np.ndarray) -> None:
        """Accumulate lag moments after transporting both residual frames."""
        self._add_moments(first, second, bases, cross_covariance=True)

    def _add_moments(self, first: np.ndarray, second: np.ndarray, bases: np.ndarray, *, cross_covariance: bool) -> None:
        valid, design, errors = _process_equations(first, second, np.asarray(bases).T)
        if not np.any(valid):
            return
        self.normal += design.T @ design
        self.target += design.T @ errors
        self.components += np.count_nonzero(design, axis=0)
        if cross_covariance:
            self.lag_pairs += 1
        else:
            self.events += 1
            self.coordinate_events += np.any(design != 0, axis=0)

    def estimate(self, baselines: np.ndarray) -> tuple[dict[str, Any], ...]:
        """Solve a nonnegative covariance fit, retaining unidentifiable priors."""
        from scipy.optimize import nnls

        baselines = np.asarray(baselines, dtype=float)
        if not np.all(np.isfinite(self.normal)) or not np.all(np.isfinite(self.target)):
            raise ValueError("Calibration process residuals produce unrepresentable moments.")
        column_scales = np.sqrt(np.maximum(np.diag(self.normal), np.finfo(float).tiny))
        normalized = self.normal / column_scales[:, None] / column_scales[None, :]
        reason = None
        if np.any(self.coordinate_events < MIN_RESIDUAL_EVENTS):
            reason = "insufficient GT transitions"
        elif np.linalg.matrix_rank(normalized) != len(baselines):
            reason = "camera-projected process scales are not identifiable from these residuals"
        if reason is None:
            # Factor the normal equations back into a least-squares problem;
            # passing the normal matrix directly to NNLS squares conditioning.
            # Native area and ratio diffusion can differ by many orders of
            # magnitude. Equilibrate columns before rank testing and solving.
            factor = np.linalg.cholesky(normalized)
            selected, _ = nnls(factor.T, np.linalg.solve(factor, self.target / column_scales))
            selected /= column_scales
        estimates = []
        for index, baseline in enumerate(baselines):
            common = {
                "baseline": float(baseline),
                "events": int(self.coordinate_events[index]),
                "lag_pairs": self.lag_pairs,
                "components": int(self.components[index]),
            }
            if reason is not None:
                estimates.append({**common, "value": float(baseline), "status": "retained", "reason": reason})
            else:
                value = float(selected[index])
                estimates.append(
                    {
                        **common,
                        "value": max(MIN_COVARIANCE_SCALE, value),
                        "raw_estimate": value,
                        "status": "fitted",
                    }
                )
        return tuple(estimates)


class DiagonalNoiseMoments:
    """Keep independent residual moments for coordinates with different units."""

    def __init__(self, size: int) -> None:
        self.moments = tuple(ScalarNoiseMoments() for _ in range(size))

    def add(self, residual: np.ndarray, variances: np.ndarray) -> None:
        """Accumulate each coordinate without pooling its magnitude with others."""
        valid, totals = _normalized_squared_errors(residual, variances)
        for index, total in zip(np.flatnonzero(valid), totals, strict=True):
            moments = self.moments[index]
            moments.total += float(total)
            moments.components += 1
            moments.events += 1


@dataclass
class ConfidenceNoiseMoments:
    """Fit E[error² / base_R] = intercept + slope * (1-confidence)².

    Both coefficients are nonnegative. Unlike scaling residuals by a confidence
    variance that approaches zero, this moment fit permits a nonzero localization
    error at confidence one. The returned runtime parameters are the covariance
    scale ``intercept+slope`` and confidence floor ``intercept/(intercept+slope)``.
    Only a 2-by-2 normal matrix is stored, independent of dataset size.
    """

    events: int = 0
    components: int = 0
    normal: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    target: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def add(self, residual: np.ndarray, variances: np.ndarray, *, confidence: float) -> None:
        """Accumulate squared errors against confidence-independent reference R."""
        design = _confidence_design(confidence)
        valid, errors = _normalized_squared_errors(residual, variances)
        if not np.any(valid):
            return
        count = int(np.count_nonzero(valid))
        self.normal += count * np.outer(design, design)
        self.target += design * float(np.sum(errors))
        self.events += 1
        self.components += count

    def estimate(self, scale: float, confidence_floor: float) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fit one curve, retaining its scale and floor together if unsupported."""
        common = {"events": self.events, "components": self.components, "method": "nonnegative_confidence_moments"}
        if not np.isfinite(self.normal).all() or not np.isfinite(self.target).all():
            raise ValueError("Calibration confidence residuals produce unrepresentable moments.")
        reason = None
        if self.events < MIN_RESIDUAL_EVENTS:
            reason = "insufficient residual events"
        elif np.linalg.matrix_rank(self.normal) < 2:
            reason = "confidence variance intercept and slope are not identifiable from these confidence values"
        if reason is not None:
            return tuple(
                {**common, "baseline": value, "value": value, "status": "retained", "reason": reason}
                for value in (scale, confidence_floor)
            )
        intercept, slope = map(float, _nonnegative_pair(self.normal, self.target))
        raw_scale = intercept + slope
        fitted_scale = max(MIN_COVARIANCE_SCALE, raw_scale)
        # A vanishing error curve contains no evidence about confidence shape.
        # Preserve that shape while retaining strictly positive covariance scale.
        fitted_floor = intercept / raw_scale if raw_scale > 0.0 else confidence_floor
        common.update(variance_intercept=intercept, variance_slope=slope, status="fitted")
        return (
            {**common, "baseline": scale, "value": fitted_scale, "raw_estimate": raw_scale},
            {
                **common,
                "baseline": confidence_floor,
                "value": fitted_floor,
                "raw_estimate": fitted_floor,
                **(
                    {"status": "fixed", "reason": "zero residual variance does not identify confidence shape"}
                    if raw_scale == 0.0
                    else {}
                ),
            },
        )


class DiagonalConfidenceNoiseMoments:
    """Fit an independent confidence variance curve for each geometry coordinate."""

    def __init__(self, size: int) -> None:
        self.moments = tuple(ConfidenceNoiseMoments() for _ in range(size))

    def add(self, residual: np.ndarray, variances: np.ndarray, *, confidence: float) -> None:
        """Keep different geometry units and confidence curves independent."""
        design = _confidence_design(confidence)
        valid, errors = _normalized_squared_errors(residual, variances)
        normal = np.outer(design, design)
        for index, error in zip(np.flatnonzero(valid), errors, strict=True):
            moments = self.moments[index]
            moments.normal += normal
            moments.target += design * float(error)
            moments.components += 1
            moments.events += 1


class DiagonalProcessNoiseMoments:
    """Fit a position/velocity diffusion pair independently for every axis."""

    def __init__(self, size: int) -> None:
        self.moments = tuple(ProcessNoiseMoments() for _ in range(size))
        self.events = 0
        self.lag_pairs = 0

    def add(self, residual: np.ndarray, position: np.ndarray, velocity: np.ndarray) -> None:
        """Keep per-coordinate equations and count each GT transition once."""
        self._add_moments(residual, residual, position, velocity, cross_covariance=False)

    def add_cross_covariance(
        self, first: np.ndarray, second: np.ndarray, position: np.ndarray, velocity: np.ndarray
    ) -> None:
        """Identify position versus velocity diffusion from signed lag moments."""
        self._add_moments(first, second, position, velocity, cross_covariance=True)

    def _add_moments(
        self,
        first: np.ndarray,
        second: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        *,
        cross_covariance: bool,
    ) -> None:
        """Form independent coordinate equations together, retaining event order."""
        valid, design, errors = _process_equations(first, second, np.column_stack((position, velocity)))
        indices = np.flatnonzero(valid)
        if not len(indices):
            return
        normals = design[:, :, None] * design[:, None, :]
        targets = design * errors[:, None]
        for index, normal, target in zip(indices, normals, targets, strict=True):
            moments = self.moments[index]
            moments.normal += normal
            moments.target += target
            moments.components += 1
            if cross_covariance:
                moments.lag_pairs += 1
            else:
                moments.events += 1
        if cross_covariance:
            self.lag_pairs += 1
        else:
            self.events += 1
