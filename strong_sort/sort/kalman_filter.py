# vim: expandtab:ts=4:sw=4
import time
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


class KalmanFilter(object):
    """
    A Square Root Unscented Kalman filter for tracking bounding boxes in
    image space.
    The 8-dimensional state space
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self):
        self.ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = dt

        self._update_mat = np.eye(self.ndim, 2 * self.ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        # Scaling parameter for spread of sigma points (1e-4 <= alpha <= 1)
        self._alpha = 1
        # Parameter for prior knowledge about the distribution
        # (beta = 2 optimal for Gaussian noise)
        self._beta = 2
        # Secondary scaling parameter (usually 0)
        self._kappa = 0
        # Number of sigma points to estimate covariance
        self._sigma_point_count = 2 * 2 * self.ndim + 1

        self._sigma_points = np.zeros((2 * self.ndim, self._sigma_point_count))

    def initiate(self, measurement):
        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and square root covariance matrix
            (8x8 dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        self._compute_weights()

        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            20 * self._std_weight_position * measurement[3],  # the center point x
            20 * self._std_weight_position * measurement[3],  # the center point y
            1 * measurement[3],  # the ratio of width/height
            20 * self._std_weight_position * measurement[3],  # the height
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            0.1 * measurement[3],
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))

        covariance_sqrt, _ = scipy.linalg.cho_factor(
            covariance, lower=True, check_finite=False)

        return mean, covariance_sqrt

    def predict(self, mean, covariance_sqrt, delta_time=1):
        """Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance_sqrt : ndarray
            The 8x8 dimensional square root covariance matrix of the object state
            at the previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = delta_time

        # Compute sigma points
        points = self._sigma_point_count
        self._sigma_points[:, 0] = mean
        self._sigma_points[:, 1:int((points - 1) / 2 + 1)] = \
            mean[:, None] + self._gamma * covariance_sqrt
        self._sigma_points[:, int((points - 1) / 2 + 1):points] = \
            mean[:, None] - self._gamma * covariance_sqrt

        # Compute predicted state
        self._sigma_points = np.dot(self._motion_mat, self._sigma_points)
        mean = np.dot(self._sigma_points, self._sigma_weights_m)

        # Compute predicted covariance
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1 * mean[3],
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            0.1 * mean[3],
            self._std_weight_velocity * mean[3]]
        motion_noise_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        cov_sqrt = self._compute_covariance_square_root_from_sigma_points(
            mean, motion_noise_cov, self._sigma_points)

        # covariance = cov_sq * np.conj(cov_sq).T

        # Return predicted state
        return mean, cov_sqrt

    def project(self, mean, covariance_sqrt, confidence=.0):
        """Project state distribution to measurement space.
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance_sqrt : ndarray
            The state's square root covariance matrix (8x8 dimensional).
        confidence: (dyh)
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """

        # Predict measurements for each sigma point
        pred_meas_sigma_points = np.dot(self._update_mat, self._sigma_points)

        ## Predict measurement from sigma measurement points
        pred_meas = np.dot(pred_meas_sigma_points, self._sigma_weights_m)

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]

        # Limit confidence to max 0.99 in order to avoid singular matrix
        std = [(1 - min(0.99, confidence)) * x for x in std]

        measurement_noise_cov = np.diag(np.square(std))

        meas_cov_sqrt = self._compute_covariance_square_root_from_sigma_points(
            pred_meas, measurement_noise_cov, pred_meas_sigma_points)

        return pred_meas, meas_cov_sqrt, pred_meas_sigma_points

    def update(self, mean, covariance_sqrt, measurement, confidence=.0):
        """Run Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's square root covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        confidence: (dyh)
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        
        try:
            # Predict measurement
            (
                projected_mean,
                projected_cov_sqrt,
                pred_meas_sigma_points
            ) = self.project(mean, covariance_sqrt, confidence)

            # Compute Kalman Gain
            W = np.repeat(
                self._sigma_weights_c[:, None],
                self._sigma_points.shape[0],
                axis=1
            ).T

            P = np.dot(
                np.multiply(self._sigma_points - mean[:, None], W),
                (pred_meas_sigma_points - projected_mean[:, None]).T
            )

            K = scipy.linalg.cho_solve(
                (projected_cov_sqrt, True), P.T, check_finite=False).T

            # Update state and covariance
            innovation = measurement - projected_mean
            new_mean = mean + np.dot(K, innovation)

            U = np.dot(K, projected_cov_sqrt)

            new_covariance_sqrt = covariance_sqrt
            for i in range(U.shape[1]):
                new_covariance_sqrt = self._rank_update(
                    new_covariance_sqrt, U[:, i], -1)

            return new_mean, new_covariance_sqrt
        # Skip update on numerical errors
        except np.linalg.linalg.LinAlgError:
            return mean, covariance_sqrt

    def gating_distance(self, mean, covariance_sqrt, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance_sqrt : ndarray
            Square Root Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance_sqrt = self.project(mean, covariance_sqrt)

        if only_position:
            mean, covariance_sqrt = mean[:2], covariance_sqrt[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            covariance_sqrt, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

    def _compute_weights(self):
        self._lambda = (self._alpha ** 2) * (2 * self.ndim + self._kappa) - 2 * self.ndim
        self._gamma = np.sqrt(2 * self.ndim + self._lambda)

        W_m_0 = self._lambda / (2 * self.ndim + self._lambda)
        W_c_0 = W_m_0 + (1 - self._alpha ** 2 + self._beta)
        W_i = 1 / (2 * (self._alpha ** 2) * (2 * self.ndim + self._kappa))

        # Ensure W_i > 0 to avoid square-root of negative number
        assert W_i > 0

        self._sigma_weights_m = [W_i for _ in range(self._sigma_point_count)]
        self._sigma_weights_m[0] = W_m_0
        self._sigma_weights_m = np.array(self._sigma_weights_m)

        self._sigma_weights_c = [W_i for _ in range(self._sigma_point_count)]
        self._sigma_weights_c[0] = W_c_0
        self._sigma_weights_c = np.array(self._sigma_weights_c)

    def _rank_update(self, L, v, nu):
        L_tril = np.tril(L)

        L_upd, _ = scipy.linalg.cho_factor(
            L_tril @ L_tril.T + nu * np.outer(v, v),
            lower=True,
            overwrite_a=True,
            check_finite=False,
        )

        L_upd[np.triu_indices(L.shape[0], k=1)] = L[np.triu_indices(L.shape[0], k=1)]

        return L_upd

    def _compute_covariance_square_root_from_sigma_points(
            self,
            mean,
            noise_cov,
            sigma_points
    ):
        ## Build augmented matrix from sigma points and upper cholesky factor
        noise_cov_cholesky_upper, _ = scipy.linalg.cho_factor(
            noise_cov, lower=False, check_finite=False)

        aug_matrix = np.zeros(
            (sigma_points.shape[1] + sigma_points.shape[0], sigma_points.shape[0]))
        aug_matrix[:sigma_points.shape[1], :] = np.sqrt(self._sigma_weights_c[1]) * (sigma_points - mean[:, None]).T
        aug_matrix[-sigma_points.shape[0]:, :] = noise_cov_cholesky_upper

        ## QR decomposition
        _, R = scipy.linalg.qr(aug_matrix, pivoting=False, check_finite=False)

        ## Update lower triangular matrix of covariance
        covariance_sqrt = self._rank_update(
            np.conj(R).T,
            sigma_points[:, 0] - mean,
            self._sigma_weights_c[0]
        )

        return covariance_sqrt