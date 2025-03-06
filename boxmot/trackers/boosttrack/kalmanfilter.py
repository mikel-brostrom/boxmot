# vim: expandtab:ts=4:sw=4
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class CovariancePolicy(ABC):

    def __init__(self, x_dim: int, z_dim: int):
        self.x_dim = x_dim
        self.z_dim = z_dim

    @abstractmethod
    def get_init_state_cov(self, z: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_R(self, x: np.ndarray, confidence: float = 0.0) -> np.ndarray:
        ...

    @abstractmethod
    def get_Q(self, x: np.ndarray) -> np.ndarray:
        ...


class ConstantNoise(CovariancePolicy):

    def get_init_state_cov(self, z: np.ndarray) -> np.ndarray:

        P = np.eye(self.x_dim)
        P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        P *= 10.0

        return P

    def get_R(self, x: np.ndarray, confidence: float = 0.0) -> np.ndarray:
        return np.diag([1, 1, 10, 0.01])

    def get_Q(self, x: np.ndarray) -> np.ndarray:
        Q = np.eye(self.x_dim)
        Q[4:, 4:] *= 0.01

        return Q


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, h, a, vx, vy, vh, va

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, h, a) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self, z: np.ndarray, ndim: int = 8, dt: int = 1,
                 cov_update_policy: CovariancePolicy = ConstantNoise,
                 id: int = -1):
        if z.ndim == 2:
            z = deepcopy(z.reshape((-1, )))

        self.dt = dt
        self.ndim = ndim
        self.cov_update_policy: CovariancePolicy = cov_update_policy(ndim, z.size)
        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(ndim, ndim)
        for i in range(4 - (ndim % 2)):
            self._motion_mat[i, i + 4] = dt

        self._update_mat = np.eye(4, ndim)

        self.x = np.zeros((ndim,))
        self.x[:4] = z[:]

        self.covariance = self.cov_update_policy.get_init_state_cov(z)
        self.id = id

    def predict(self, mean: Optional[np.ndarray] = None,
                covariance: Optional[np.ndarray] = None):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        update = False
        if mean is None:
            mean = self.x
            covariance = self.covariance
            update = True
        motion_cov = self.cov_update_policy.get_Q(mean)

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        if update:
            self.x = mean
            self.covariance = covariance

        return mean, covariance

    def project(self, confidence=.0):
        """Project state distribution to measurement space.

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """

        innovation_cov = self.cov_update_policy.get_R(self.x, 0)

        mean = np.dot(self._update_mat, self.x)
        covariance = np.linalg.multi_dot((
            self._update_mat, self.covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, z: np.ndarray, confidence=.0):
        """Run Kalman filter correction step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """

        if z.ndim == 2:
            z = deepcopy(z.reshape((-1, )))
        projected_mean, projected_cov = self.project(confidence)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(self.covariance, self._update_mat.T).T,
            check_finite=False).T

        innovation = z - projected_mean

        self.x = self.x + np.dot(innovation, kalman_gain.T)
        self.covariance = self.covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))

        return self.x, self.covariance
