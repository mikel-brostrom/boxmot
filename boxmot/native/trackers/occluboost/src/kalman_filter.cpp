#include "occluboost/kalman_filter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace occluboost {

namespace {

constexpr double kMinSize = 1.0e-4;
constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfPi = kPi / 2.0;

double WrapAngle(const double angle) {
    const double period = 2.0 * kPi;
    return std::fmod(std::fmod(angle + kPi, period) + period, period) - kPi;
}

KalmanFilterXYHR::Vector AlignObbMeasurement(
    const KalmanFilterXYHR::Vector& measurement,
    const KalmanFilterXYHR::Vector& reference
) {
    // measurement / reference layout: [x, y, h, r, theta]
    KalmanFilterXYHR::Vector aligned = measurement;
    const double ref_h = std::max(reference[2], 1.0e-6);
    const double ref_r = std::max(reference[3], 1.0e-6);
    const double ref_w = ref_h * ref_r;
    const double ref_theta = reference[4];

    const double meas_h = std::max(measurement[2], 1.0e-6);
    const double meas_r = std::max(measurement[3], 1.0e-6);
    const double meas_w = meas_h * meas_r;
    const double theta = measurement[4];

    // Equivalent OBB parameterizations: same rectangle, different (w,h,θ).
    struct Cand { double w; double h; double theta; };
    const Cand candidates[4] = {
        {meas_w, meas_h, theta},
        {meas_w, meas_h, theta + kPi},
        {meas_h, meas_w, theta + kHalfPi},
        {meas_h, meas_w, theta - kHalfPi},
    };

    double best_cost = std::numeric_limits<double>::infinity();
    double best_h = meas_h;
    double best_r = meas_r;
    double best_theta = theta;
    for (const auto& cand : candidates) {
        const double w = std::max(cand.w, 1.0e-6);
        const double h = std::max(cand.h, 1.0e-6);
        const double theta_aligned = ref_theta + WrapAngle(cand.theta - ref_theta);
        const double angle_cost = std::abs(theta_aligned - ref_theta);
        const double size_cost =
            std::abs(std::log(w / ref_w)) + std::abs(std::log(h / ref_h));
        const double cost = angle_cost + (0.05 * size_cost);
        if (cost < best_cost) {
            best_cost = cost;
            best_h = h;
            best_r = w / h;
            best_theta = theta_aligned;
        }
    }
    aligned[2] = best_h;
    aligned[3] = best_r;
    aligned[4] = best_theta;
    return aligned;
}

}  // namespace

KalmanFilterXYHR::KalmanFilterXYHR() {
    Configure(4);
}

void KalmanFilterXYHR::Configure(const int dim_z) {
    if (dim_z != 4 && dim_z != 5) {
        throw std::runtime_error("KalmanFilterXYHR: dim_z must be 4 (AABB) or 5 (OBB).");
    }
    dim_z_ = dim_z;
    dim_x_ = 2 * dim_z;
    is_obb_ = (dim_z == 5);

    motion_mat_ = Matrix::Identity(dim_x_, dim_x_);
    for (int i = 0; i < dim_z_; ++i) {
        motion_mat_(i, dim_z_ + i) = 1.0;
    }
    update_mat_ = Matrix::Zero(dim_z_, dim_x_);
    for (int i = 0; i < dim_z_; ++i) {
        update_mat_(i, i) = 1.0;
    }

    // Q = eye(dim_x); velocity block scaled by 0.01; theta diag scaled by 0.01 in OBB.
    process_noise_ = Matrix::Identity(dim_x_, dim_x_);
    for (int i = dim_z_; i < dim_x_; ++i) {
        process_noise_(i, i) = 0.01;
    }
    if (is_obb_) {
        process_noise_(4, 4) = 0.01;
    }

    // R = diag(1, 1, 10, 0.01[, 0.01]).
    measurement_noise_ = Matrix::Zero(dim_z_, dim_z_);
    measurement_noise_(0, 0) = 1.0;
    measurement_noise_(1, 1) = 1.0;
    measurement_noise_(2, 2) = 10.0;
    measurement_noise_(3, 3) = 0.01;
    if (is_obb_) {
        measurement_noise_(4, 4) = 0.01;
    }

    // P0 = eye(dim_x); P0[dim_z:, dim_z:] *= 1000; then *= 10.
    initial_covariance_ = Matrix::Identity(dim_x_, dim_x_);
    for (int i = dim_z_; i < dim_x_; ++i) {
        initial_covariance_(i, i) *= 1000.0;
    }
    initial_covariance_ *= 10.0;

    mean_ = Vector::Zero(dim_x_);
    covariance_ = initial_covariance_;
}

void KalmanFilterXYHR::Initiate(const Vector& measurement) {
    if (measurement.size() != 4 && measurement.size() != 5) {
        throw std::runtime_error("OccluBoost KF initiate: measurement must have 4 (AABB) or 5 (OBB) elements.");
    }
    if (measurement.size() != dim_z_) {
        Configure(static_cast<int>(measurement.size()));
    }
    mean_ = Vector::Zero(dim_x_);
    mean_.head(dim_z_) = measurement.head(dim_z_);
    EnforceConstraints();
    covariance_ = initial_covariance_;
}

void KalmanFilterXYHR::Predict() {
    mean_ = motion_mat_ * mean_;
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
    if (is_obb_) {
        z = AlignObbMeasurement(z, mean_.head(dim_z_));
    }

    const auto [projected_mean, projected_cov] = Project();

    Eigen::LLT<Matrix> llt(projected_cov);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("OccluBoost KF update: projected covariance not PD");
    }

    const Matrix projected_inv = llt.solve(Matrix::Identity(dim_z_, dim_z_));
    const Matrix kalman_gain = covariance_ * update_mat_.transpose() * projected_inv;

    const Vector innovation = z - projected_mean;
    mean_ = mean_ + alpha * (kalman_gain * innovation);
    covariance_ = covariance_ - kalman_gain * projected_cov * kalman_gain.transpose();
    EnforceConstraints();
}

void KalmanFilterXYHR::EnforceConstraints() {
    mean_[2] = std::max(mean_[2], kMinSize);
    mean_[3] = std::max(mean_[3], kMinSize);
    if (is_obb_) {
        mean_[4] = WrapAngle(mean_[4]);
    }
    covariance_ = 0.5 * (covariance_ + covariance_.transpose());
}

}  // namespace occluboost
