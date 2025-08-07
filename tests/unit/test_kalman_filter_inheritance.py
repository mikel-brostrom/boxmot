"""
Test to verify all Kalman Filters inherit from BaseKalmanFilter.
"""
import pytest
import numpy as np

from boxmot.motion.kalman_filters.aabb.base_kalman_filter import BaseKalmanFilter
from boxmot.motion.kalman_filters.aabb.xyah_kf import KalmanFilterXYAH
from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR
from boxmot.motion.kalman_filters.obb.xywha_kf import KalmanFilterXYWHA


class TestKalmanFilterInheritance:
    """Test that all Kalman Filters inherit from BaseKalmanFilter."""

    def test_xyah_inherits_from_base(self):
        """Test XYAH KF inherits from BaseKalmanFilter."""
        kf = KalmanFilterXYAH()
        assert isinstance(kf, BaseKalmanFilter), "XYAH KF should inherit from BaseKalmanFilter"

    def test_xywh_inherits_from_base(self):
        """Test XYWH KF inherits from BaseKalmanFilter."""
        kf = KalmanFilterXYWH()
        assert isinstance(kf, BaseKalmanFilter), "XYWH KF should inherit from BaseKalmanFilter"

    def test_xysr_inherits_from_base(self):
        """Test XYSR KF inherits from BaseKalmanFilter."""
        kf = KalmanFilterXYSR(dim_x=7, dim_z=4)
        assert isinstance(kf, BaseKalmanFilter), "XYSR KF should inherit from BaseKalmanFilter"

    def test_xywha_inherits_from_base(self):
        """Test XYWHA KF inherits from BaseKalmanFilter."""
        kf = KalmanFilterXYWHA(dim_x=10, dim_z=5)
        assert isinstance(kf, BaseKalmanFilter), "XYWHA KF should inherit from BaseKalmanFilter"

    def test_all_kfs_work_with_base_interface(self):
        """Test that all KFs work with the BaseKalmanFilter interface."""
        # Test XYAH and XYWH (high-level interface)
        measurement_xyah = np.array([100, 100, 0.5, 50])  # x, y, aspect, height
        kf_xyah = KalmanFilterXYAH()
        mean, cov = kf_xyah.initiate(measurement_xyah)
        assert mean.shape == (8,), "XYAH KF should have 8D state"
        assert cov.shape == (8, 8), "XYAH KF should have 8x8 covariance"

        measurement_xywh = np.array([100, 100, 50, 50])  # x, y, width, height  
        kf_xywh = KalmanFilterXYWH()
        mean, cov = kf_xywh.initiate(measurement_xywh)
        assert mean.shape == (8,), "XYWH KF should have 8D state"
        assert cov.shape == (8, 8), "XYWH KF should have 8x8 covariance"

    def test_low_level_kfs_maintain_functionality(self):
        """Test that low-level KFs maintain their original functionality."""
        # Test XYSR KF (low-level interface)
        kf_xysr = KalmanFilterXYSR(dim_x=7, dim_z=4)
        assert kf_xysr.x.shape == (7, 1), "XYSR KF should have 7D state vector"
        assert kf_xysr.P.shape == (7, 7), "XYSR KF should have 7x7 covariance"
        
        # Test matrix assignment (important for OCSORT)
        kf_xysr.F = np.eye(7)
        kf_xysr.H = np.zeros((4, 7))
        kf_xysr.H[:4, :4] = np.eye(4)
        
        # Test predict/update cycle
        kf_xysr.predict()
        bbox_xysr = np.array([100, 100, 1000, 0.5]).reshape((4, 1))
        kf_xysr.update(bbox_xysr)

        # Test XYWHA KF (low-level interface with constraints)
        kf_xywha = KalmanFilterXYWHA(dim_x=10, dim_z=5)
        assert kf_xywha.x.shape == (10, 1), "XYWHA KF should have 10D state vector"
        assert kf_xywha.P.shape == (10, 10), "XYWHA KF should have 10x10 covariance"
        
        # Test matrix assignment
        kf_xywha.F = np.eye(10)
        kf_xywha.H = np.zeros((5, 10))
        kf_xywha.H[:5, :5] = np.eye(5)
        
        # Test predict/update cycle
        kf_xywha.predict()
        bbox_xywha = np.array([100, 100, 50, 50, 0.5]).reshape((5, 1))  # x,y,w,h,angle
        kf_xywha.update(bbox_xywha)

    def test_angle_constraints_work(self):
        """Test that angle constraints work for XYWHA KF."""
        kf = KalmanFilterXYWHA(dim_x=10, dim_z=5)
        
        # Set an angle outside [-pi, pi]
        kf.x[4, 0] = 5.0  # > pi
        kf._enforce_constraints()
        
        # Check angle is wrapped to [-pi, pi]
        from math import pi
        assert -pi <= kf.x[4, 0] <= pi, "Angle should be wrapped to [-pi, pi]"
        
        # Test negative width/height clamping
        kf.x[2, 0] = -10.0  # negative width
        kf.x[3, 0] = -5.0   # negative height
        kf._enforce_constraints()
        
        assert kf.x[2, 0] >= 1e-4, "Width should be clamped to positive value"
        assert kf.x[3, 0] >= 1e-4, "Height should be clamped to positive value"