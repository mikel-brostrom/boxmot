from __future__ import absolute_import, division

import sys
from collections import deque
from copy import deepcopy
from math import exp, log, pi
from typing import Optional, Tuple

import numpy as np
import numpy.linalg as linalg
from filterpy.stats import logpdf
from numpy import dot

from boxmot.motion.kalman_filters.aabb.base_kalman_filter import BaseKalmanFilter


def speed_direction_obb(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[1]
    cx2, cy2 = bbox2[0], bbox2[1]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return float((angle + pi) % (2.0 * pi) - pi)


class KalmanBoxTrackerOBB(object):
    """
    This class represents the internal state of individual tracked objects observed as oriented bbox.
    """

    count = 0

    def __init__(
        self,
        bbox,
        cls,
        det_ind,
        delta_t=3,
        max_obs=50,
        Q_xy_scaling=0.01,
        Q_a_scaling=0.01,
    ):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.det_ind = det_ind

        self.Q_xy_scaling = Q_xy_scaling
        self.Q_a_scaling = Q_a_scaling

        self.kf = KalmanFilterXYWHA(dim_x=10, dim_z=5, max_obs=max_obs)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # cx = cx + vx
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # cy = cy + vy
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # w = w + vw
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # h = h + vh
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # a = a + va
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # cx
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # cy
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # w
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # h
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # angle
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[
            5:, 5:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0

        self.kf.Q[5:7, 5:7] *= self.Q_xy_scaling
        self.kf.Q[-1, -1] *= self.Q_a_scaling

        self.kf.x[:5] = bbox[:5].reshape(
            (5, 1)
        )  # x, y, w, h, angle   (dont take confidence score)
        self.kf._enforce_obb_constraints()
        self.time_since_update = 0
        self.id = KalmanBoxTrackerOBB.count
        KalmanBoxTrackerOBB.count += 1
        self.max_obs = max_obs
        self.history = deque([], maxlen=self.max_obs)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.conf = bbox[-1]
        self.cls = cls
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1, -1])  #WARNING : -1 is a valid angle value 
        self.observations = dict()
        self.history_observations = deque([], maxlen=self.max_obs)
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bbox, cls, det_ind):
        """
        Updates the state vector with observed bbox.
        """
        self.det_ind = det_ind
        if bbox is not None:
            bbox = np.asarray(bbox, dtype=float).copy()
            bbox[:5] = self.kf.prepare_obb_measurement(
                bbox[:5], reference_angle=None
            ).reshape(-1)
            self.conf = bbox[-1]
            self.cls = cls
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                # Estimate the track speed direction with observations Δt steps away
                self.velocity = speed_direction_obb(previous_box, bbox)

            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(bbox[:5].reshape((5, 1))) # x, y, w, h, angle as column vector   (dont take confidence score)
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[7] + self.kf.x[2]) <= 0:  # Negative width
            self.kf.x[7] *= 0.0
        if (self.kf.x[8] + self.kf.x[3]) <= 0:  # Negative Height
            self.kf.x[8] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[0:5].reshape((1, 5)))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[0:5].reshape((1, 5))


class KalmanFilterXYWHA(BaseKalmanFilter):
    """
    Implements a Kalman Filter specialized for tracking Oriented Bounding Boxes.
    The default state vector is [x, y, w, h, a]^T:

        - (x, y): center of the bounding box
        - w, h  : width and height of the bounding box
        - a     : orientation angle (radians)

    This filter supports "freeze" and "unfreeze" methods to handle missing
    observations (no measurements) or out-of-sequence (OOS) smoothing logic.
    """

    ANGLE_IDX = 4
    WIDTH_IDX = 2
    HEIGHT_IDX = 3
    POS_IDX = slice(0, 2)
    SIZE_IDX = slice(2, 4)
    VEL_POS_IDX = slice(5, 7)
    VEL_SIZE_IDX = slice(7, 9)
    MIN_SIZE = 1e-4

    def __init__(
        self, dim_x: int, dim_z: int, dim_u: int = 0, max_obs: int = 50
    ) -> None:
        """
        Parameters
        ----------
        dim_x : int
            Dimensionality of the state vector. Typically 5 if [x, y, w, h, a].
        dim_z : int
            Dimensionality of the measurement vector. Typically also 5.
        dim_u : int
            Dimensionality of the control vector. Default is 0 (no control).
        max_obs : int
            Maximum number of stored observations for freeze/unfreeze logic.
        """
        if dim_x < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_z < 1:
            raise ValueError("dim_z must be 1 or greater")
        if dim_u < 0:
            raise ValueError("dim_u must be 0 or greater")
        if dim_x < 5:
            raise ValueError("dim_x must be at least 5 for OBB state [x,y,w,h,a,...]")
        if dim_z < 5:
            raise ValueError("dim_z must be at least 5 for OBB measurements [x,y,w,h,a]")

        super().__init__(ndim=dim_z, dim_x=dim_x, dim_z=dim_z, max_obs=max_obs)
        self.dim_u = dim_u

        # Internal log-likelihood computations
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # For potential smoothing usage
        self.inv = np.linalg.inv

    def _enforce_obb_constraints(self) -> None:
        """Keep width/height positive and angle wrapped."""
        if self.dim_x > self.WIDTH_IDX:
            self.x[self.WIDTH_IDX, 0] = max(
                float(self.x[self.WIDTH_IDX, 0]), self.MIN_SIZE
            )
        if self.dim_x > self.HEIGHT_IDX:
            self.x[self.HEIGHT_IDX, 0] = max(
                float(self.x[self.HEIGHT_IDX, 0]), self.MIN_SIZE
            )
        if self.dim_x > self.ANGLE_IDX:
            self.x[self.ANGLE_IDX, 0] = wrap_angle(float(self.x[self.ANGLE_IDX, 0]))

    def _symmetrize_covariance(self) -> None:
        """Limit numerical drift by forcing covariance symmetry."""
        self.P = 0.5 * (self.P + self.P.T)

    def _measurement_reference_angle(self) -> Optional[float]:
        if self.dim_x > self.ANGLE_IDX:
            return float(self.x[self.ANGLE_IDX, 0])
        return None

    def prepare_obb_measurement(
        self, z: np.ndarray, reference_angle: Optional[float] = None
    ) -> np.ndarray:
        """
        Normalize OBB measurement and, when provided, align angle to the
        closest equivalent around a reference angle.
        """
        measurement = self._reshape_measurement(z, self.dim_z)

        if self.dim_z > self.WIDTH_IDX:
            measurement[self.WIDTH_IDX, 0] = max(
                float(measurement[self.WIDTH_IDX, 0]), self.MIN_SIZE
            )
        if self.dim_z > self.HEIGHT_IDX:
            measurement[self.HEIGHT_IDX, 0] = max(
                float(measurement[self.HEIGHT_IDX, 0]), self.MIN_SIZE
            )

        if self.dim_z > self.ANGLE_IDX:
            raw_angle = wrap_angle(float(measurement[self.ANGLE_IDX, 0]))
            if reference_angle is None:
                measurement[self.ANGLE_IDX, 0] = raw_angle
            else:
                measurement[self.ANGLE_IDX, 0] = (
                    reference_angle + wrap_angle(raw_angle - reference_angle)
                )
        return measurement

    @staticmethod
    def _affine_components(m: np.ndarray) -> Tuple[float, float, float]:
        """Extract approximate x/y scaling and rotation from an affine matrix."""
        u, _, vh = np.linalg.svd(m)
        rot = u @ vh
        if np.linalg.det(rot) < 0:
            u[:, -1] *= -1.0
            rot = u @ vh
        angle = np.arctan2(rot[1, 0], rot[0, 0])
        scale_x = max(float(np.linalg.norm(m[:, 0])), 1e-6)
        scale_y = max(float(np.linalg.norm(m[:, 1])), 1e-6)
        return scale_x, scale_y, angle

    def _apply_affine_to_state(
        self,
        x: np.ndarray,
        P: np.ndarray,
        m: np.ndarray,
        t: np.ndarray,
        *,
        scale_x: float,
        scale_y: float,
        delta_angle: float,
    ) -> None:
        """Apply affine transform to state/covariance using a linearized Jacobian."""
        x[self.POS_IDX] = m @ x[self.POS_IDX] + t
        x[self.SIZE_IDX.start, 0] = max(
            float(x[self.SIZE_IDX.start, 0]) * scale_x, self.MIN_SIZE
        )
        x[self.SIZE_IDX.start + 1, 0] = max(
            float(x[self.SIZE_IDX.start + 1, 0]) * scale_y, self.MIN_SIZE
        )
        x[self.ANGLE_IDX, 0] = wrap_angle(float(x[self.ANGLE_IDX, 0]) + delta_angle)

        if self.dim_x >= self.VEL_POS_IDX.stop:
            x[self.VEL_POS_IDX] = m @ x[self.VEL_POS_IDX]
        if self.dim_x >= self.VEL_SIZE_IDX.stop:
            x[self.VEL_SIZE_IDX.start, 0] *= scale_x
            x[self.VEL_SIZE_IDX.start + 1, 0] *= scale_y

        jacobian = np.eye(self.dim_x, dtype=float)
        jacobian[self.POS_IDX, self.POS_IDX] = m
        jacobian[self.SIZE_IDX.start, self.SIZE_IDX.start] = scale_x
        jacobian[self.SIZE_IDX.start + 1, self.SIZE_IDX.start + 1] = scale_y
        if self.dim_x >= self.VEL_POS_IDX.stop:
            jacobian[self.VEL_POS_IDX, self.VEL_POS_IDX] = m
        if self.dim_x >= self.VEL_SIZE_IDX.stop:
            jacobian[self.VEL_SIZE_IDX.start, self.VEL_SIZE_IDX.start] = scale_x
            jacobian[
                self.VEL_SIZE_IDX.start + 1, self.VEL_SIZE_IDX.start + 1
            ] = scale_y

        P[:] = jacobian @ P @ jacobian.T
        P[:] = 0.5 * (P + P.T)

    def _apply_affine_to_measurement(
        self,
        z: np.ndarray,
        m: np.ndarray,
        t: np.ndarray,
        *,
        scale_x: float,
        scale_y: float,
        delta_angle: float,
    ) -> np.ndarray:
        measurement = self.prepare_obb_measurement(z, reference_angle=None)
        measurement[self.POS_IDX] = m @ measurement[self.POS_IDX] + t
        measurement[self.SIZE_IDX.start, 0] *= scale_x
        measurement[self.SIZE_IDX.start + 1, 0] *= scale_y
        measurement[self.ANGLE_IDX, 0] = wrap_angle(
            float(measurement[self.ANGLE_IDX, 0]) + delta_angle
        )
        return measurement

    def apply_affine_correction(self, m: np.ndarray, t: np.ndarray) -> None:
        """
        Apply an affine transform to the current state and covariance.
        This is useful if the image or reference frame is warped.

        Parameters
        ----------
        m : np.array(2x2)
            Affine transform (rotation/scale) to be applied to x,y and maybe w,h
        t : np.array(2x1)
            Translation vector to be added after applying the transform.

        TODO: adapt for oriented bounding box (especially if the orientation
        is also changed by the transform).
        """
        m = np.asarray(m, dtype=float).reshape((2, 2))
        t = np.asarray(t, dtype=float).reshape((2, 1))
        scale_x, scale_y, delta_angle = self._affine_components(m)

        self._apply_affine_to_state(
            self.x,
            self.P,
            m,
            t,
            scale_x=scale_x,
            scale_y=scale_y,
            delta_angle=delta_angle,
        )
        self._apply_affine_to_state(
            self.x_prior,
            self.P_prior,
            m,
            t,
            scale_x=scale_x,
            scale_y=scale_y,
            delta_angle=delta_angle,
        )
        self._apply_affine_to_state(
            self.x_post,
            self.P_post,
            m,
            t,
            scale_x=scale_x,
            scale_y=scale_y,
            delta_angle=delta_angle,
        )
        self._enforce_obb_constraints()
        self._symmetrize_covariance()

        # If we froze the filter, also update the frozen state
        if not self.observed and self.attr_saved is not None:
            self._apply_affine_to_state(
                self.attr_saved["x"],
                self.attr_saved["P"],
                m,
                t,
                scale_x=scale_x,
                scale_y=scale_y,
                delta_angle=delta_angle,
            )
            self._apply_affine_to_state(
                self.attr_saved["x_prior"],
                self.attr_saved["P_prior"],
                m,
                t,
                scale_x=scale_x,
                scale_y=scale_y,
                delta_angle=delta_angle,
            )
            self._apply_affine_to_state(
                self.attr_saved["x_post"],
                self.attr_saved["P_post"],
                m,
                t,
                scale_x=scale_x,
                scale_y=scale_y,
                delta_angle=delta_angle,
            )

            # last_measurement might need updating similarly
            if self.attr_saved["last_measurement"] is not None:
                self.attr_saved["last_measurement"] = self._apply_affine_to_measurement(
                    self.attr_saved["last_measurement"],
                    m,
                    t,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    delta_angle=delta_angle,
                )

    def predict(
        self,
        u: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        F: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
    ) -> None:
        """
        Predict next state (prior) using the state transition matrix F
        and process noise Q.

        Parameters
        ----------
        u : np.array(dim_u, 1), optional
            Control vector. If not provided, assumed 0.
        B : np.array(dim_x, dim_u), optional
            Control transition matrix. If None, self.B is used.
        F : np.array(dim_x, dim_x), optional
            State transition matrix. If None, self.F is used.
        Q : np.array(dim_x, dim_x) or scalar, optional
            Process noise matrix. If None, self.Q is used. If scalar,
            Q = scalar * I.
        """
        self.predict_state(u=u, B=B, F=F, Q=Q)
        self._enforce_obb_constraints()
        self._symmetrize_covariance()

    def freeze(self):
        """
        Save the current filter parameters in attr_saved so that if the next
        observation is missing, we can revert to these parameters for
        out-of-sequence or offline smoothing.
        """
        self.attr_saved = deepcopy(self.__dict__)

    def unfreeze(self) -> None:
        """
        Revert the filter parameters to the saved (frozen) state, then "replay"
        the missing measurements from history to smooth the intermediate states.
        """
        if self.attr_saved is not None:
            new_history = deepcopy(list(self.history_obs))
            # revert to the frozen attributes
            self.__dict__ = self.attr_saved
            # remove last measurement from history (since we'll re-apply them)
            self.history_obs = deque(list(self.history_obs)[:-1], maxlen=self.max_obs)

            # naive approach: re-update states between the two known measurements
            occur = [int(d is None) for d in new_history]
            indices = np.where(np.array(occur) == 0)[0]
            if len(indices) < 2:
                return  # not enough measurements to replay

            index1, index2 = indices[-2], indices[-1]
            box1 = np.asarray(new_history[index1], dtype=float).reshape(-1)
            box2 = np.asarray(new_history[index2], dtype=float).reshape(-1)
            if box1.size < 5 or box2.size < 5:
                return

            x1, y1, w1, h1, a1 = box1[:5]
            x2, y2, w2, h2, a2 = box2[:5]
            time_gap = index2 - index1
            if time_gap <= 0:
                return
            dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
            dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap
            da = wrap_angle(a2 - a1) / time_gap

            for i in range(index2 - index1):
                x_ = x1 + (i + 1) * dx
                y_ = y1 + (i + 1) * dy
                w_ = w1 + (i + 1) * dw
                h_ = h1 + (i + 1) * dh
                a_ = wrap_angle(a1 + (i + 1) * da)

                new_box = np.array([x_, y_, w_, h_, a_], dtype=float).reshape((5, 1))
                self.update(new_box)
                if i != (index2 - index1 - 1):
                    self.predict()
                    self.history_obs.pop()
            self.history_obs.pop()

    def update(
        self,
        z: Optional[np.ndarray],
        R: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
    ) -> None:
        """
        Incorporate a new measurement z into the state estimate.

        Parameters
        ----------
        z : np.array(dim_z, 1)
            Measurement vector. If None, skip update step (missing measurement).
        R : np.array(dim_z, dim_z), scalar, or None
            Measurement noise matrix. If None, self.R is used.
        H : np.array(dim_z, dim_x) or None
            Measurement function. If None, self.H is used.
        """
        # reset log-likelihood computations
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        measurement = None
        if z is not None:
            measurement = self.prepare_obb_measurement(
                z, reference_angle=self._measurement_reference_angle()
            )
        self.history_obs.append(
            None if measurement is None else deepcopy(measurement)
        )

        # If measurement is missing
        if measurement is None:
            if self.observed:
                # freeze the current parameters for future potential smoothing
                self.last_measurement = self.history_obs[-2]
                self.freeze()
            self.observed = False
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim_z, 1))
            return

        # If we haven't observed for a while, revert to the frozen state
        if not self.observed:
            self.unfreeze()
        self.observed = True

        R = self._resolve_matrix(R, self.R)
        H = self._resolve_matrix(H, self.H)

        if np.isscalar(R):
            R = np.eye(self.dim_z) * float(R)

        z = measurement

        # y = z - Hx   (residual)
        self.y = z - dot(H, self.x)

        PHT = dot(self.P, H.T)
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)

        # K = PHT * SI
        self.K = PHT.dot(self.SI)

        # x = x + K y
        self.x = self.x + dot(self.K, self.y)

        # P = (I - K H) P (I - K H)^T + K R K^T (Joseph form)
        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        # Save measurement and posterior
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        self._enforce_obb_constraints()
        self._symmetrize_covariance()

    def update_steadystate(self, z, H=None):
        """
        Update using precomputed steady-state gain (K_steady_state) and
        steady-state covariance P. Only x is updated here.
        P remains unchanged.
        """
        if z is None:
            self.history_obs.append(z)
            return

        if H is None:
            H = self.H

        measurement = self.prepare_obb_measurement(
            z, reference_angle=self._measurement_reference_angle()
        )

        # residual
        self.y = measurement - dot(H, self.x)

        # x = x + K_steady_state * y
        self.x = self.x + dot(self.K_steady_state, self.y)

        # Save measurement and posterior
        self.z = deepcopy(measurement)
        self.x_post = self.x.copy()
        self._enforce_obb_constraints()
        self.history_obs.append(measurement)

    def log_likelihood_of(self, z=None):
        """
        Compute the log-likelihood of measurement z given the current
        measurement prediction. This uses logpdf from filterpy.stats.
        """
        if z is None:
            z = self.z
        return logpdf(z, dot(self.H, self.x), self.S)

    def likelihood_of(self, z=None):
        """
        Compute the likelihood (probability) of measurement z given
        the current measurement prediction.
        """
        return exp(self.log_likelihood_of(z))

    @property
    def log_likelihood(self):
        """log-likelihood of the last measurement."""
        return self._log_likelihood

    @property
    def likelihood(self):
        """likelihood of the last measurement."""
        return self._likelihood


def batch_filter(
    x, P, zs, Fs, Qs, Hs, Rs, Bs=None, us=None, update_first=False, saver=None
):
    """
    Batch processes a sequence of measurements.

    Parameters
    ----------
    x : np.array(dim_x, 1)
        Initial state.
    P : np.array(dim_x, dim_x)
        Initial covariance.
    zs : list-like
        List of measurements at each time step (None for missing).
    Fs : list-like
        State transition matrices for each step.
    Qs : list-like
        Process noise covariances for each step.
    Hs : list-like
        Measurement matrices for each step.
    Rs : list-like
        Measurement noise covariances for each step.
    Bs : list-like, optional
        Control transition matrices for each step.
    us : list-like, optional
        Control vectors for each step.
    update_first : bool
        If True, apply update->predict. Otherwise predict->update.
    saver : filterpy.common.Saver, optional
        If provided, saver.save() is called at each step.

    Returns
    -------
    means : np.array((n,dim_x,1))
    covariances : np.array((n,dim_x,dim_x))
    means_p : np.array((n,dim_x,1))
        Predictions after each step
    covariances_p : np.array((n,dim_x,dim_x))
        Covariances after prediction each step
    """
    n = np.size(zs, 0)
    dim_x = x.shape[0]

    # Arrays to store results
    if x.ndim == 1:
        means = np.zeros((n, dim_x))
        means_p = np.zeros((n, dim_x))
    else:
        means = np.zeros((n, dim_x, 1))
        means_p = np.zeros((n, dim_x, 1))

    covariances = np.zeros((n, dim_x, dim_x))
    covariances_p = np.zeros((n, dim_x, dim_x))

    if us is None:
        us = [0.0] * n
        Bs = [0.0] * n

    # Procedural version of predict->update or update->predict
    for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

        if update_first:
            # Update step
            x, P = update(x, P, z, R=R, H=H)
            means[i, :] = x
            covariances[i, :, :] = P

            # Predict step
            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P

        else:
            # Predict step
            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P

            # Update step
            x, P = update(x, P, z, R=R, H=H)
            means[i, :] = x
            covariances[i, :, :] = P

        if saver is not None:
            saver.save()

    return (means, covariances, means_p, covariances_p)


def update(x, P, z, R, H):
    """
    Procedural form of the update step of the Kalman Filter.
    """
    if z is None:
        return x, P

    # y = z - Hx
    y = z - dot(H, x)
    PHT = dot(P, H.T)
    S = dot(H, PHT) + R
    SI = linalg.inv(S)
    K = dot(PHT, SI)

    # x = x + Ky
    x = x + dot(K, y)

    # P = (I - KH)P(I - KH)' + KRK'
    I_KH = np.eye(x.shape[0]) - dot(K, H)
    P = dot(dot(I_KH, P), I_KH.T) + dot(dot(K, R), K.T)

    return x, P


def predict(x, P, F, Q, B=None, u=None):
    """
    Procedural form of the predict step of the Kalman Filter.
    """
    if B is not None and u is not None:
        x = dot(F, x) + dot(B, u)
    else:
        x = dot(F, x)

    P = dot(dot(F, P), F.T) + Q
    return x, P
