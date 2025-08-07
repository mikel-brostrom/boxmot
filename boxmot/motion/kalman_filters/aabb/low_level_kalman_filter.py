from typing import Tuple
import numpy as np
import sys
from collections import deque
from copy import deepcopy
from math import exp, log
from filterpy.common import reshape_z

from boxmot.motion.kalman_filters.aabb.base_kalman_filter import BaseKalmanFilter


class LowLevelKalmanFilter(BaseKalmanFilter):
    """
    A base class for Kalman filters that need direct matrix manipulation.
    This bridges the gap between the high-level BaseKalmanFilter interface
    and low-level filterpy-style implementations while maintaining API compatibility.
    
    This class provides both the BaseKalmanFilter interface and the direct
    matrix access needed by existing filterpy-style implementations.
    """

    def __init__(self, ndim: int, dim_x: int, dim_z: int, dim_u: int = 0, max_obs: int = 50):
        super().__init__(ndim)
        
        # Low-level filter properties needed by filterpy-style implementations
        if dim_x < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_z < 1:
            raise ValueError("dim_z must be 1 or greater")
        if dim_u < 0:
            raise ValueError("dim_u must be 0 or greater")

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # State vector and covariance (filterpy style)
        self.x = np.zeros((dim_x, 1))        # state
        self.P = np.eye(dim_x)               # uncertainty covariance
        self.Q = np.eye(dim_x)               # process uncertainty
        self.B = None                        # control transition matrix
        self.F = np.eye(dim_x)               # state transition matrix
        self.H = np.zeros((dim_z, dim_x))    # measurement function
        self.R = np.eye(dim_z)               # measurement uncertainty
        self._alpha_sq = 1.                  # fading memory control
        self.M = np.zeros((dim_x, dim_z))    # process-measurement cross correlation
        self.z = np.array([[None]*self.dim_z]).T

        # Gain and residual computed during innovation
        self.K = np.zeros((dim_x, dim_z))    # kalman gain
        self.y = np.zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))    # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))   # inverse system uncertainty

        # Identity matrix
        self._I = np.eye(dim_x)

        # Prior and posterior state copies
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # Log-likelihood computation
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # Observation history for freeze/unfreeze functionality
        self.max_obs = max_obs
        self.history_obs = deque([], maxlen=self.max_obs)

        self.inv = np.linalg.inv
        self.attr_saved = None
        self.observed = False
        self.last_measurement = None

    # Default implementations for BaseKalmanFilter abstract methods
    # Subclasses can override these for custom behavior
    
    def _get_initial_covariance_std(self, measurement: np.ndarray) -> np.ndarray:
        """
        Default implementation using standard weight factors.
        Override for custom behavior.
        """
        # Use generic approach based on measurement size
        std = []
        for i in range(len(measurement)):
            if i < 2:  # x, y positions
                std.append(2 * self._std_weight_position * np.mean(measurement))
            else:  # other dimensions (w, h, s, r, angle, etc.)
                std.append(self._std_weight_position * np.mean(measurement))
        
        # Add velocity standard deviations
        for i in range(len(measurement)):
            std.append(10 * self._std_weight_velocity * np.mean(measurement))
            
        return std

    def _get_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Default implementation using standard weight factors.
        Override for custom behavior.
        """
        n_pos = self.ndim
        mean_scale = np.mean(np.abs(mean[:n_pos]))
        
        std_pos = [self._std_weight_position * mean_scale] * n_pos
        std_vel = [self._std_weight_velocity * mean_scale] * n_pos
        
        return std_pos, std_vel

    def _get_measurement_noise_std(self, mean: np.ndarray, confidence: float) -> np.ndarray:
        """
        Default implementation using standard weight factors.
        Override for custom behavior.
        """
        n_pos = self.ndim
        mean_scale = np.mean(np.abs(mean[:n_pos]))
        
        return [self._std_weight_position * mean_scale] * n_pos

    def _enforce_constraints(self):
        """
        Enforce physical constraints on state variables.
        Override in subclasses for specific constraint handling.
        """
        pass  # Default: no constraints

    def _get_multi_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Default implementation for vectorized process noise.
        Override for custom behavior.
        """
        n_pos = self.ndim
        batch_size = mean.shape[0]
        
        mean_scale = np.mean(np.abs(mean[:, :n_pos]), axis=1)
        
        std_pos = [self._std_weight_position * mean_scale] * n_pos
        std_vel = [self._std_weight_velocity * mean_scale] * n_pos
        
        return std_pos, std_vel

    # High-level BaseKalmanFilter interface using low-level matrices
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the filter with a measurement, using the low-level matrices.
        """
        # Set initial state
        self.x[:len(measurement), 0] = measurement
        
        # Set initial covariance using the high-level interface
        std = self._get_initial_covariance_std(measurement)
        self.P = np.diag(np.square(std))
        
        return self.x.flatten(), self.P

    def predict(self, mean: np.ndarray = None, covariance: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        High-level predict interface that delegates to low-level implementation.
        """
        if mean is not None:
            self.x = mean.reshape(-1, 1)
        if covariance is not None:
            self.P = covariance
            
        self.predict_low_level()
        return self.x.flatten(), self.P

    def predict_low_level(self, u=None, B=None, F=None, Q=None):
        """
        Low-level predict method compatible with filterpy-style usage.
        """
        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self.dim_x) * Q

        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = np.dot(F, self.x) + np.dot(B, u)
        else:
            self.x = np.dot(F, self.x)

        # P = FPF' + Q
        self.P = self._alpha_sq * np.dot(np.dot(F, self.P), F.T) + Q

        # Save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # Enforce constraints if needed (can be overridden by subclasses)
        self._enforce_constraints()

    def update(self, 
               z, 
               R=None, 
               H=None, 
               mean: np.ndarray = None, 
               covariance: np.ndarray = None, 
               measurement: np.ndarray = None, 
               confidence: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update method that supports both high-level and low-level interfaces.
        """
        if mean is not None:
            self.x = mean.reshape(-1, 1)
        if covariance is not None:
            self.P = covariance
            
        # If using high-level measurement parameter
        if measurement is not None:
            z = measurement.reshape(-1, 1)
            
        self.update_low_level(z, R, H)
        return self.x.flatten(), self.P

    def update_low_level(self, z, R=None, H=None):
        """
        Low-level update method compatible with filterpy-style usage.
        """
        # Reset log-likelihood computations
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # Save observation
        self.history_obs.append(z)

        if z is None:
            if self.observed:
                self.last_measurement = self.history_obs[-2]
                self.freeze()
            self.observed = False
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim_z, 1))
            return

        if not self.observed:
            self.unfreeze()
        self.observed = True

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R
        if H is None:
            z = reshape_z(z, self.dim_z, self.x.ndim)
            H = self.H

        # y = z - Hx (innovation/residual)
        self.y = z - np.dot(H, self.x)

        # S = HPH' + R (innovation covariance)
        PHT = np.dot(self.P, H.T)
        self.S = np.dot(H, PHT) + R
        self.SI = self.inv(self.S)

        # K = PH'S^-1 (Kalman gain)
        self.K = PHT.dot(self.SI)

        # x = x + Ky (state update)
        self.x = self.x + np.dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK' (covariance update - Joseph form)
        I_KH = self._I - np.dot(self.K, H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(self.K, R), self.K.T)

        # Save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # Enforce constraints after update
        self._enforce_constraints()

    def freeze(self):
        """Save the parameters before non-observation forward."""
        self.attr_saved = deepcopy(self.__dict__)

    def unfreeze(self):
        """Restore saved parameters and interpolate missing observations."""
        # This is a placeholder - real implementation depends on specific filter needs
        if self.attr_saved is not None:
            self.__dict__ = self.attr_saved

    def log_likelihood_of(self, z=None):
        """Compute log-likelihood of measurement z."""
        if z is None:
            z = self.z
        from filterpy.stats import logpdf
        return logpdf(z, np.dot(self.H, self.x), self.S)

    def likelihood_of(self, z=None):
        """Compute likelihood of measurement z."""
        return exp(self.log_likelihood_of(z))

    @property
    def log_likelihood(self):
        """log-likelihood of the last measurement."""
        return self._log_likelihood

    @property
    def likelihood(self):
        """likelihood of the last measurement."""
        return self._likelihood