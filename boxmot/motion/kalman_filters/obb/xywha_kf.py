from __future__ import absolute_import, division

import sys
from collections import deque
from copy import deepcopy
from math import exp, log, pi

import numpy as np
import numpy.linalg as linalg
from filterpy.common import reshape_z
from filterpy.stats import logpdf
from numpy import dot, eye, isscalar, zeros

from boxmot.motion.kalman_filters.aabb.low_level_kalman_filter import LowLevelKalmanFilter


def speed_direction_obb(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[1]
    cx2, cy2 = bbox2[0], bbox2[1]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


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

        self.kf.x[:5] = bbox[:5].reshape((5, 1)) # x, y, w, h, angle   (dont take confidence score)
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


class KalmanFilterXYWHA(LowLevelKalmanFilter):
    """
    Implements a Kalman Filter specialized for tracking Oriented Bounding Boxes.
    The default state vector is [x, y, w, h, a]^T:

        - (x, y): center of the bounding box
        - w, h  : width and height of the bounding box
        - a     : orientation angle (radians)

    This filter supports "freeze" and "unfreeze" methods to handle missing
    observations (no measurements) or out-of-sequence (OOS) smoothing logic.
    """

    def __init__(self, dim_x, dim_z, dim_u=0, max_obs=50):
        # Call parent constructor - use 5 as ndim for x,y,w,h,angle
        super().__init__(ndim=5, dim_x=dim_x, dim_z=dim_z, dim_u=dim_u, max_obs=max_obs)

    def apply_affine_correction(self, m, t):
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
        # For demonstration, we apply the transform to [x, y] and [x_dot, y_dot], etc.
        # But for your OBB case, consider carefully how w, h, and angle should transform.

        # Example basic approach: transform x, y
        self.x[:2] = m @ self.x[:2] + t

        # Possibly transform w, h. But if w,h are not purely length in x,y directions,
        # you might have to do something more elaborate. For demonstration:
        self.x[2:4] = np.abs(m @ self.x[2:4])  # naive approach: scale w,h

        # P block for positions:
        self.P[:2, :2] = m @ self.P[:2, :2] @ m.T
        # P block for widths/heights (again naive if we treat w,h as x,y scale):
        self.P[2:4, 2:4] = m @ self.P[2:4, 2:4] @ m.T

        # If angle is included, consider adjusting it or leaving it if the transform
        # is purely in the plane with no orientation offset. Could do angle wrap here.

        # If we froze the filter, also update the frozen state
        if not self.observed and self.attr_saved is not None:
            self.attr_saved["x"][:2] = m @ self.attr_saved["x"][:2] + t
            self.attr_saved["x"][2:4] = np.abs(m @ self.attr_saved["x"][2:4])
            self.attr_saved["P"][:2, :2] = m @ self.attr_saved["P"][:2, :2] @ m.T
            self.attr_saved["P"][2:4, 2:4] = m @ self.attr_saved["P"][2:4, 2:4] @ m.T

            # last_measurement might need updating similarly
            self.attr_saved["last_measurement"][:2] = (
                m @ self.attr_saved["last_measurement"][:2] + t
            )

    def _enforce_constraints(self):
        """
        Enforce bounding box and angle constraints for oriented bounding boxes.
        """
        if self.dim_x >= 5:
            # clamp w, h > 0
            self.x[2, 0] = max(self.x[2, 0], 1e-4)
            self.x[3, 0] = max(self.x[3, 0], 1e-4)

            # wrap angle to [-pi, pi]
            from math import pi
            self.x[4, 0] = (self.x[4, 0] + pi) % (2 * pi) - pi

    def predict(self, u=None, B=None, F=None, Q=None):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations. This delegates to the parent class low-level implementation.
        """
        self.predict_low_level(u, B, F, Q)

    def freeze(self):
        """
        Save the current filter parameters in attr_saved so that if the next
        observation is missing, we can revert to these parameters for
        out-of-sequence or offline smoothing.
        """
        self.attr_saved = deepcopy(self.__dict__)

    def unfreeze(self):
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
            box1, box2 = new_history[index1], new_history[index2]
            x1, y1, w1, h1, a1 = box1
            x2, y2, w2, h2, a2 = box2
            time_gap = index2 - index1
            dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
            dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap
            da = (a2 - a1) / time_gap

            for i in range(index2 - index1):
                x_ = x1 + (i + 1) * dx
                y_ = y1 + (i + 1) * dy
                w_ = w1 + (i + 1) * dw
                h_ = h1 + (i + 1) * dh
                a_ = a1 + (i + 1) * da

                new_box = np.array([x_, y_, w_, h_, a_]).reshape((5, 1))
                self.update(new_box)
                if i != (index2 - index1 - 1):
                    self.predict()
                    self.history_obs.pop()
            self.history_obs.pop()

    def update(self, z, R=None, H=None):
        """
        Incorporate a new measurement z into the state estimate. 
        This delegates to the parent class low-level implementation.
        """
        self.update_low_level(z, R, H)
        # Apply constraints after update as well
        self._enforce_constraints()

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

        # residual
        self.y = z - dot(H, self.x)

        # x = x + K_steady_state * y
        self.x = self.x + dot(self.K_steady_state, self.y)

        # Save measurement and posterior
        self.z = deepcopy(z)
        self.x_post = self.x.copy()

        self.history_obs.append(z)

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
