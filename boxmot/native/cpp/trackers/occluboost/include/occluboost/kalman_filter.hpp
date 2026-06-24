#pragma once

#include <Eigen/Dense>

namespace occluboost {

// Constant-noise XYHR Kalman filter mirroring boxmot.motion.kalman_filters.xyhr.
// AABB mode (dim_z=4, dim_x=8): state [x, y, h, r, vx, vy, vh, vr]
// OBB  mode (dim_z=5, dim_x=10): state [x, y, h, r, theta, vx, vy, vh, vr, vtheta]
class KalmanFilterXYHR {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;

    // Default-constructed AABB filter for backward compatibility.
    KalmanFilterXYHR();

    // Initialize state vector + covariance from a measurement. Latches the
    // filter to AABB (size 4) or OBB (size 5) mode based on ``measurement``.
    void Initiate(const Vector& measurement);

    void Predict();

    // Standard update step with optional alpha-scaled Kalman gain.
    void Update(const Vector& measurement, double alpha = 1.0);

    const Vector& mean() const { return mean_; }
    Vector& mutable_mean() { return mean_; }
    const Matrix& covariance() const { return covariance_; }

    int dim_x() const { return dim_x_; }
    int dim_z() const { return dim_z_; }
    bool is_obb() const { return is_obb_; }

private:
    void Configure(int dim_z);
    void EnforceConstraints();
    std::pair<Vector, Matrix> Project() const;

    int dim_x_ = 8;
    int dim_z_ = 4;
    bool is_obb_ = false;

    Matrix motion_mat_;
    Matrix update_mat_;
    Matrix process_noise_;       // Q
    Matrix measurement_noise_;   // R
    Matrix initial_covariance_;  // P0

    Vector mean_;
    Matrix covariance_;
};

}  // namespace occluboost
