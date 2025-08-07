"""
This module implements the linear Kalman filter in both an object
oriented and procedural form. The KalmanFilter class implements
the filter by storing the various matrices in instance variables,
minimizing the amount of bookkeeping you have to do.
All Kalman filters operate with a predict->update cycle. The
predict step, implemented with the method or function predict(),
uses the state transition matrix F to predict the state in the next
time period (epoch). The state is stored as a gaussian (x, P), where
x is the state (column) vector, and P is its covariance. Covariance
matrix Q specifies the process covariance. In Bayesian terms, this
prediction is called the *prior*, which you can think of colloquially
as the estimate prior to incorporating the measurement.
The update step, implemented with the method or function `update()`,
incorporates the measurement z with covariance R, into the state
estimate (x, P). The class stores the system uncertainty in S,
the innovation (residual between prediction and measurement in
measurement space) in y, and the Kalman gain in k. The procedural
form returns these variables to you. In Bayesian terms this computes
the *posterior* - the estimate after the information from the
measurement is incorporated.
Whether you use the OO form or procedural form is up to you. If
matrices such as H, R, and F are changing each epoch, you'll probably
opt to use the procedural form. If they are unchanging, the OO
form is perhaps easier to use since you won't need to keep track
of these matrices. This is especially useful if you are implementing
banks of filters or comparing various KF designs for performance;
a trivial coding bug could lead to using the wrong sets of matrices.
This module also offers an implementation of the RTS smoother, and
other helper functions, such as log likelihood computations.
The Saver class allows you to easily save the state of the
KalmanFilter class after every update.
"""

from __future__ import absolute_import, division

import sys
from collections import deque
from copy import deepcopy
from math import exp, log

import numpy as np
from filterpy.common import reshape_z
from filterpy.stats import logpdf
from numpy import dot, eye, isscalar, zeros

from boxmot.motion.kalman_filters.aabb.base_kalman_filter import BaseKalmanFilter


from boxmot.motion.kalman_filters.aabb.low_level_kalman_filter import LowLevelKalmanFilter


class KalmanFilterXYSR(LowLevelKalmanFilter):
    """Implements a Kalman filter. You are responsible for setting the
    various state variables to reasonable values; the defaults will
    not give you a functional filter.
    """

    def __init__(self, dim_x, dim_z, dim_u=0, max_obs=50):
        # Call parent constructor - use dim_z as ndim since it represents state dimensionality
        super().__init__(ndim=dim_z, dim_x=dim_x, dim_z=dim_z, dim_u=dim_u, max_obs=max_obs)

    def apply_affine_correction(self, m, t):
        """
        Apply to both last state and last observation for OOS smoothing.

        Messy due to internal logic for kalman filter being messy.
        """

        scale = np.linalg.norm(m[:, 0])
        self.x[:2] = m @ self.x[:2] + t
        self.x[4:6] = m @ self.x[4:6]

        self.P[:2, :2] = m @ self.P[:2, :2] @ m.T
        self.P[4:6, 4:6] = m @ self.P[4:6, 4:6] @ m.T

        # If frozen, also need to update the frozen state for OOS
        if not self.observed and self.attr_saved is not None:
            self.attr_saved["x"][:2] = m @ self.attr_saved["x"][:2] + t
            self.attr_saved["x"][4:6] = m @ self.attr_saved["x"][4:6]

            self.attr_saved["P"][:2, :2] = m @ self.attr_saved["P"][:2, :2] @ m.T
            self.attr_saved["P"][4:6, 4:6] = m @ self.attr_saved["P"][4:6, 4:6] @ m.T

            self.attr_saved["last_measurement"][:2] = (
                m @ self.attr_saved["last_measurement"][:2] + t
            )

    def predict(self, u=None, B=None, F=None, Q=None):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations. This delegates to the parent class low-level implementation.
        """
        self.predict_low_level(u, B, F, Q)

    def freeze(self):
        """Save the parameters before non-observation forward."""
        super().freeze()

    def unfreeze(self):
        if self.attr_saved is not None:
            new_history = deepcopy(list(self.history_obs))
            self.__dict__ = self.attr_saved
            self.history_obs = deque(list(self.history_obs)[:-1], maxlen=self.max_obs)
            occur = [int(d is None) for d in new_history]
            indices = np.where(np.array(occur) == 0)[0]
            index1, index2 = indices[-2], indices[-1]
            box1, box2 = new_history[index1], new_history[index2]
            x1, y1, s1, r1 = box1
            w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
            x2, y2, s2, r2 = box2
            w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)
            time_gap = index2 - index1
            dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
            dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap

            for i in range(index2 - index1):
                x, y = x1 + (i + 1) * dx, y1 + (i + 1) * dy
                w, h = w1 + (i + 1) * dw, h1 + (i + 1) * dh
                s, r = w * h, w / float(h)
                new_box = np.array([x, y, s, r]).reshape((4, 1))
                self.update(new_box)
                if not i == (index2 - index1 - 1):
                    self.predict()
                    self.history_obs.pop()
            self.history_obs.pop()

    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter. This delegates to the parent class.
        """
        self.update_low_level(z, R, H)

    def update_steadystate(self, z, H=None):
        """Update Kalman filter using the Kalman gain and state covariance
        matrix as computed for the steady state. Only x is updated, and the
        new value is stored in self.x. P is left unchanged. Must be called
        after a prior call to compute_steady_state().
        """
        if z is None:
            self.history_obs.append(z)
            return

        if H is None:
            H = self.H

        H = np.asarray(H)
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # x = x + Ky
        self.x = self.x + dot(self.K_steady_state, self.y)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()

        # save history of observations
        self.history_obs.append(z)

    # Use inherited log_likelihood_of and likelihood_of methods from parent


def batch_filter(
    x, P, zs, Fs, Qs, Hs, Rs, Bs=None, us=None, update_first=False, saver=None
):
    """
    Batch processes a sequences of measurements.
    Parameters
    ----------
    zs : list-like
        list of measurements at each time step. Missing measurements must be
        represented by None.
    Fs : list-like
        list of values to use for the state transition matrix matrix.
    Qs : list-like
        list of values to use for the process error
        covariance.
    Hs : list-like
        list of values to use for the measurement matrix.
    Rs : list-like
        list of values to use for the measurement error
        covariance.
    Bs : list-like, optional
        list of values to use for the control transition matrix;
        a value of None in any position will cause the filter
        to use `self.B` for that time step.
    us : list-like, optional
        list of values to use for the control input vector;
        a value of None in any position will cause the filter to use
        0 for that time step.
    update_first : bool, optional
        controls whether the order of operations is update followed by
        predict, or predict followed by update. Default is predict->update.
        saver : filterpy.common.Saver, optional
            filterpy.common.Saver object. If provided, saver.save() will be
            called after every epoch
    Returns
    -------
    means : np.array((n,dim_x,1))
        array of the state for each time step after the update. Each entry
        is an np.array. In other words `means[k,:]` is the state at step
        `k`.
    covariance : np.array((n,dim_x,dim_x))
        array of the covariances for each time step after the update.
        In other words `covariance[k,:,:]` is the covariance at step `k`.
    means_predictions : np.array((n,dim_x,1))
        array of the state for each time step after the predictions. Each
        entry is an np.array. In other words `means[k,:]` is the state at
        step `k`.
    covariance_predictions : np.array((n,dim_x,dim_x))
        array of the covariances for each time step after the prediction.
        In other words `covariance[k,:,:]` is the covariance at step `k`.
    Examples
    --------
    .. code-block:: Python
        zs = [t + random.randn()*4 for t in range (40)]
        Fs = [kf.F for t in range (40)]
        Hs = [kf.H for t in range (40)]
        (mu, cov, _, _) = kf.batch_filter(zs, Rs=R_list, Fs=Fs, Hs=Hs, Qs=None,
                                          Bs=None, us=None, update_first=False)
        (xs, Ps, Ks, Pps) = kf.rts_smoother(mu, cov, Fs=Fs, Qs=None)
    """

    n = np.size(zs, 0)
    dim_x = x.shape[0]

    # mean estimates from Kalman Filter
    if x.ndim == 1:
        means = zeros((n, dim_x))
        means_p = zeros((n, dim_x))
    else:
        means = zeros((n, dim_x, 1))
        means_p = zeros((n, dim_x, 1))

    # state covariances from Kalman Filter
    covariances = zeros((n, dim_x, dim_x))
    covariances_p = zeros((n, dim_x, dim_x))

    if us is None:
        us = [0.0] * n
        Bs = [0.0] * n

    if update_first:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

            x, P = update(x, P, z, R=R, H=H)
            means[i, :] = x
            covariances[i, :, :] = P

            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P
            if saver is not None:
                saver.save()
    else:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P

            x, P = update(x, P, z, R=R, H=H)
            means[i, :] = x
            covariances[i, :, :] = P
            if saver is not None:
                saver.save()

    return (means, covariances, means_p, covariances_p)

    def batch_filter(self, zs, Rs=None):
        """
        Batch process a sequence of measurements. This method is suitable
        for cases where the measurement noise varies with each measurement.
        """
        means, covariances = [], []
        for z, R in zip(zs, Rs):
            self.predict()
            self.update(z, R=R)
            means.append(self.x.copy())
            covariances.append(self.P.copy())
        return np.array(means), np.array(covariances)
