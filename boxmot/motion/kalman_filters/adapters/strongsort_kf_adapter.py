# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
from filterpy.common import reshape_z

from ..kalman_filter import KalmanFilter


class StrongSortKalmanFilterAdapter(KalmanFilter):
    ndim = 4

    def __init__(self, dt=1):
        super().__init__(dim_x=2 * self.ndim, dim_z=self.ndim)

        # Set transition matrix
        for i in range(self.ndim):
            self.F[i, self.ndim + i] = dt

        # Set observation matrix
        self.H = np.eye(self.ndim, 2 * self.ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, w, h) with center position (x, y),
            width w, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        self.x = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[0],  # the center point x
            2 * self._std_weight_position * measurement[1],  # the center point y
            1 * measurement[2],  # the ratio of width/height
            2 * self._std_weight_position * measurement[3],  # the height
            10 * self._std_weight_velocity * measurement[0],
            10 * self._std_weight_velocity * measurement[1],
            0.1 * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        self.P = np.diag(np.square(std))

        return self.x, self.P

    def predict(self, mean, covariance):
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
        std_pos = [
            self._std_weight_position * mean[0],
            self._std_weight_position * mean[1],
            1 * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[0],
            self._std_weight_velocity * mean[1],
            0.1 * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        super().predict(Q=motion_cov)

        return self.x, self.P

    def update(self, mean, covariance, measurement, confidence=0.0):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, w, h), where (x, y)
            is the center position, w the width, and h the height of the
            bounding box.
        confidence : float
            Confidence level of measurement

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        self.x = mean
        self.P = covariance

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        std = [(1 - confidence) * x for x in std]

        innovation_cov = np.diag(np.square(std))

        super().update(measurement, R=innovation_cov)

        return self.x, self.P

    def gating_distance(self, measurements, only_position=False):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : bool
            Whether to use only the positional attributes of the track for
            calculating the gating distance
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """

        squared_maha = np.zeros((measurements.shape[0],))
        for i, measurement in enumerate(measurements):
            if not only_position:
                squared_maha[i] = super().md_for_measurement(measurement)
            else:
                # TODO (henriksod): Needs to be tested!
                z = reshape_z(measurements, self.dim_z, 2)
                H = self.H[:2, :2]
                y = z - np.dot(H, self.x[:2])
                squared_maha[i] = np.sqrt(float(np.dot(np.dot(y.T, self.SI[:2, :2]), y)))
        return squared_maha
