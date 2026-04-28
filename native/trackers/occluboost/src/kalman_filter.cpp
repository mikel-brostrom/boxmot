#include "occluboost/kalman_filter.hpp"

#include <algorithm>
#include <stdexcept>

namespace occluboost {

namespace {

constexpr double kMinSize = 1.0e-4;

KalmanFilterXYHR::Vector EnforceSize(KalmanFilterXYHR::Vector v) {
    v[2] = std::max(v[2], kMinSize);
    v[3] = std::max(v[3], kMinSize);
    return v;
}

}  // namespace

KalmanFilterXYHR::KalmanFilterXYHR()
    : motion_mat_(Matrix::Identity(dim_x_, dim_x_)),
      update_mat_(Matrix::Zero(dim_z_, dim_x_)),
      process_noise_(Matrix::Identity(dim_x_, dim_x_)),
      measurement_noise_(Matrix::Zero(dim_z_, dim_z_)),
      initial_covariance_(Matrix::Identity(dim_x_, dim_x_)),
      mean_(Vector::Zero(dim_x_)),
      covariance_(Matrix::Identity(dim_x_, dim_x_)) {
    // motion model: position += velocity * dt (dt=1).
    for (int i = 0; i < dim_z_; ++i) {
        motion_mat_(i, dim_z_ + i) = 1.0;
    }
    // measurement model: observe positions only.
    for (int i = 0; i < dim_z_; ++i) {
        update_mat_(i, i) = 1.0;
    }
    // Q = eye(dim_x); velocity block scaled by 0.01.
    for (int i = dim_z_; i < dim_x_; ++i) {
        process_noise_(i, i) = 0.01;
    }
    // R = diag(1, 1, 10, 0.01).
    measurement_noise_(0, 0) = 1.0;
    measurement_noise_(1, 1) = 1.0;
    measurement_noise_(2, 2) = 10.0;
    measurement_noise_(3, 3) = 0.01;
    // Initial covariance: eye(dim_x) * 10; velocity block also * 1000.
    for (int i = dim_z_; i < dim_x_; ++i) {
        initial_covariance_(i, i) *= 1000.0;
    }
    initial_covariance_ *= 10.0;
}

void KalmanFilterXYHR::Initiate(const Vector& measurement) {
    if (measurement.size() < dim_z_) {
        throw std::runtime_error("OccluBoost KF initiate: measurement too short");
    }
    mean_ = Vector::Zero(dim_x_);
    mean_.head(dim_z_) = measurement.head(dim_z_);
    mean_ = EnforceSize(mean_);
    covariance_ = initial_covariance_;
}

void KalmanFilterXYHR::Predict() {
    mean_ = motion_mat_ * mean_;
    mean_ = EnforceSize(mean_);
    covariance_ = motion_mat_ * covariance_ * motion_mat_.transpose() + process_noise_;
    EnforceConstraints();
}

std::pair<KalmanFilterXYHR::Vector, KalmanFilterXYHR::Matrix> KalmanFilterXYHR::Project() const {
    Vector projected_mean = update_mat_ * mean_;
    Matrix projected_cov = update_mat_ * covariance_ * update_mat_.transpose() + measurement_noise_;
    return {projected_mean, projected_cov};
}

void KalmanFilterXYHR::Update(const Vector& measurement, const double alpha) {
    if (measurement.size() < dim_z_) {
        throw std::runtime_error("OccluBoost KF update: measurement too short");
    }
    Vector z = measurement.head(dim_z_);

    const auto [projected_mean, projected_cov] = Project();

    Eigen::LLT<Matrix> llt(projected_cov);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("OccluBoost KF update: projected covariance not PD");
    }

    // K = covariance * H^T * (H P H^T + R)^{-1}
    const Matrix projected_inv = llt.solve(Matrix::Identity(dim_z_, dim_z_));
    const Matrix kalman_gain = covariance_ * update_mat_.transpose() * projected_inv;

    const Vector innovation = z - projected_mean;
    mean_ = mean_ + alpha * (kalman_gain * innovation);
    covariance_ = covariance_ - kalman_gain * projected_cov * kalman_gain.transpose();
    EnforceConstraints();
}

void KalmanFilterXYHR::EnforceConstraints() {
    mean_ = EnforceSize(mean_);
    covariance_ = 0.5 * (covariance_ + covariance_.transpose());
}

}  // namespace occluboost
