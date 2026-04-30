#pragma once

#include <Eigen/Dense>

namespace occluboost {

// Constant-noise XYHR Kalman filter mirroring boxmot.motion.kalman_filters.xyhr.
// State: [x, y, h, r, vx, vy, vh, vr] (dim_x = 8, dim_z = 4).
class KalmanFilterXYHR {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;

    KalmanFilterXYHR();

    // Initialize state vector + covariance from a measurement [x, y, h, r].
    void Initiate(const Vector& measurement);

    // Standard predict step (overwrites internal state).
    void Predict();

    // Standard update step with optional alpha-scaled Kalman gain.
    // alpha=1.0 reproduces a standard correction; alpha<1 dampens the mean
    // update only — covariance shrinks normally.
    void Update(const Vector& measurement, double alpha = 1.0);

    // Get current state vector [cx, cy, h, r, vx, vy, vh, vr].
    const Vector& mean() const { return mean_; }
    Vector& mutable_mean() { return mean_; }
    const Matrix& covariance() const { return covariance_; }

    int dim_x() const { return dim_x_; }
    int dim_z() const { return dim_z_; }

private:
    void EnforceConstraints();
    std::pair<Vector, Matrix> Project() const;

    static constexpr int dim_x_ = 8;
    static constexpr int dim_z_ = 4;

    Matrix motion_mat_;
    Matrix update_mat_;
    Matrix process_noise_;       // Q
    Matrix measurement_noise_;   // R
    Matrix initial_covariance_;  // P0

    Vector mean_;
    Matrix covariance_;
};

}  // namespace occluboost
