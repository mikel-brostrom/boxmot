# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from ..kalman_filter import KalmanFilter


class OCSortKalmanFilterAdapter(KalmanFilter):
    def __init__(self, dim_x, dim_z):
        super().__init__(dim_x=dim_x, dim_z=dim_z)
