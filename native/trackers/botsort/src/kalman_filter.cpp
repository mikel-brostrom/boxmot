#include "botsort/kalman_filter.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace botsort {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfPi = kPi / 2.0;

double WrapAngle(const double angle) {
    const double period = 2.0 * kPi;
    return std::fmod(std::fmod(angle + kPi, period) + period, period) - kPi;
}

KalmanFilterXYWH::Vector AlignObbMeasurement(
    const KalmanFilterXYWH::Vector& measurement,
    const KalmanFilterXYWH::Vector& reference
) {
    KalmanFilterXYWH::Vector aligned = measurement;
    const double ref_w = std::max(reference[2], 1.0e-6);
    const double ref_h = std::max(reference[3], 1.0e-6);
    const double ref_theta = reference[4];
    const double width = std::max(aligned[2], 1.0e-6);
    const double height = std::max(aligned[3], 1.0e-6);
    const double theta = aligned[4];

    const std::array<std::array<double, 3>, 4> candidates = {{
        {width, height, theta},
        {width, height, theta + kPi},
        {height, width, theta + kHalfPi},
        {height, width, theta - kHalfPi},
    }};

    double best_cost = std::numeric_limits<double>::infinity();
    std::array<double, 3> best = candidates.front();
    for (const auto& candidate : candidates) {
        const double theta_aligned = ref_theta + WrapAngle(candidate[2] - ref_theta);
        const double angle_cost = std::abs(theta_aligned - ref_theta);
        const double size_cost =
            std::abs(std::log(std::max(candidate[0], 1.0e-6) / ref_w)) +
            std::abs(std::log(std::max(candidate[1], 1.0e-6) / ref_h));
        const double cost = angle_cost + (0.05 * size_cost);
        if (cost < best_cost) {
            best_cost = cost;
            best = {candidate[0], candidate[1], theta_aligned};
        }
    }

    aligned[2] = best[0];
    aligned[3] = best[1];
    aligned[4] = best[2];
    return aligned;
}

KalmanFilterXYWH::Vector EnforceXywhConstraints(KalmanFilterXYWH::Vector mean, const bool is_obb) {
    mean[2] = std::max(mean[2], 1.0e-4);
    mean[3] = std::max(mean[3], 1.0e-4);
    if (is_obb && mean.size() >= 5) {
        mean[4] = WrapAngle(mean[4]);
    }
    return mean;
}

}  // namespace

KalmanFilterXYWH::KalmanFilterXYWH(const int ndim)
    : ndim_(ndim),
      dim_x_(2 * ndim),
      is_obb_(ndim == 5),
      motion_mat_(dim_x_, dim_x_),
      update_mat_(ndim_, dim_x_) {
    if (ndim_ != 4 && ndim_ != 5) {
        throw std::invalid_argument("KalmanFilterXYWH ndim must be 4 (AABB) or 5 (OBB).");
    }
    motion_mat_.setIdentity();
    for (int i = 0; i < ndim_; ++i) {
        motion_mat_(i, ndim_ + i) = 1.0;
    }
    update_mat_.setZero();
    update_mat_.block(0, 0, ndim_, ndim_).setIdentity();
}

KalmanFilterXYWH::Vector KalmanFilterXYWH::InitialCovarianceStd(const Vector& measurement) const {
    Vector std(dim_x_);
    std.setZero();
    std[0] = 2.0 * std_weight_position_ * measurement[2];
    std[1] = 2.0 * std_weight_position_ * measurement[3];
    std[2] = 2.0 * std_weight_position_ * measurement[2];
    std[3] = 2.0 * std_weight_position_ * measurement[3];
    std[ndim_] = 10.0 * std_weight_velocity_ * measurement[2];
    std[ndim_ + 1] = 10.0 * std_weight_velocity_ * measurement[3];
    std[ndim_ + 2] = 10.0 * std_weight_velocity_ * measurement[2];
    std[ndim_ + 3] = 10.0 * std_weight_velocity_ * measurement[3];
    if (is_obb_) {
        std[4] = 1.0e-2;
        std[dim_x_ - 1] = 1.0e-5;
    }
    return std;
}

std::pair<KalmanFilterXYWH::Vector, KalmanFilterXYWH::Vector> KalmanFilterXYWH::ProcessNoiseStd(
    const Vector& mean
) const {
    Vector std_pos(ndim_);
    Vector std_vel(ndim_);
    std_pos.setZero();
    std_vel.setZero();
    std_pos[0] = std_weight_position_ * mean[2];
    std_pos[1] = std_weight_position_ * mean[3];
    std_pos[2] = std_weight_position_ * mean[2];
    std_pos[3] = std_weight_position_ * mean[3];
    std_vel[0] = std_weight_velocity_ * mean[2];
    std_vel[1] = std_weight_velocity_ * mean[3];
    std_vel[2] = std_weight_velocity_ * mean[2];
    std_vel[3] = std_weight_velocity_ * mean[3];
    if (is_obb_) {
        std_pos[4] = 1.0e-2;
        std_vel[4] = 1.0e-5;
    }
    return {std_pos, std_vel};
}

KalmanFilterXYWH::Vector KalmanFilterXYWH::MeasurementNoiseStd(const Vector& mean, const float confidence) const {
    const double weight = 1.0 - static_cast<double>(confidence);
    Vector std(ndim_);
    std.setZero();
    std[0] = weight * std_weight_position_ * mean[2];
    std[1] = weight * std_weight_position_ * mean[3];
    std[2] = weight * std_weight_position_ * mean[2];
    std[3] = weight * std_weight_position_ * mean[3];
    if (is_obb_) {
        std[4] = weight * 1.0e-1;
    }
    return std;
}

std::pair<KalmanFilterXYWH::Vector, KalmanFilterXYWH::Matrix> KalmanFilterXYWH::Initiate(
    const Vector& measurement
) const {
    if (measurement.size() != ndim_) {
        throw std::runtime_error("KalmanFilterXYWH initiate measurement size mismatch.");
    }
    Vector mean(dim_x_);
    mean << measurement, Vector::Zero(ndim_);
    mean = EnforceXywhConstraints(mean, is_obb_);

    const Vector std = InitialCovarianceStd(measurement);
    Matrix covariance = std.array().square().matrix().asDiagonal();
    return {mean, covariance};
}

std::pair<KalmanFilterXYWH::Vector, KalmanFilterXYWH::Matrix> KalmanFilterXYWH::Predict(
    const Vector& mean,
    const Matrix& covariance
) const {
    Vector predicted_mean = motion_mat_ * mean;
    predicted_mean = EnforceXywhConstraints(predicted_mean, is_obb_);

    const auto [std_pos, std_vel] = ProcessNoiseStd(predicted_mean);
    Vector std(dim_x_);
    std << std_pos, std_vel;
    Matrix motion_cov = std.array().square().matrix().asDiagonal();
    Matrix predicted_covariance = motion_mat_ * covariance * motion_mat_.transpose() + motion_cov;
    return {predicted_mean, predicted_covariance};
}

std::pair<KalmanFilterXYWH::Vector, KalmanFilterXYWH::Matrix> KalmanFilterXYWH::Project(
    const Vector& mean,
    const Matrix& covariance,
    const float confidence
) const {
    Vector projected_mean = update_mat_ * mean;
    Matrix projected_covariance = update_mat_ * covariance * update_mat_.transpose();
    const Vector std = MeasurementNoiseStd(mean, confidence);
    projected_covariance += std.array().square().matrix().asDiagonal();
    return {projected_mean, projected_covariance};
}

std::pair<KalmanFilterXYWH::Vector, KalmanFilterXYWH::Matrix> KalmanFilterXYWH::Update(
    const Vector& mean,
    const Matrix& covariance,
    const Vector& measurement,
    const float confidence
) const {
    if (measurement.size() != ndim_) {
        throw std::runtime_error("KalmanFilterXYWH update measurement size mismatch.");
    }
    Vector aligned_measurement = measurement;
    if (is_obb_) {
        aligned_measurement = AlignObbMeasurement(measurement, mean.head(ndim_));
    }

    const auto [projected_mean, projected_covariance] = Project(mean, covariance, confidence);
    Eigen::LLT<Matrix> llt(projected_covariance);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("Kalman update failed: projected covariance is not positive definite");
    }

    const Matrix projected_cov_inv = llt.solve(Matrix::Identity(ndim_, ndim_));
    const Matrix kalman_gain = covariance * update_mat_.transpose() * projected_cov_inv;
    const Vector innovation = aligned_measurement - projected_mean;

    Vector updated_mean = mean + kalman_gain * innovation;
    Matrix updated_covariance = covariance - kalman_gain * projected_covariance * kalman_gain.transpose();
    if (is_obb_ && updated_mean.size() >= dim_x_) {
        updated_mean[dim_x_ - 1] *= 0.8;
    }
    updated_mean = EnforceXywhConstraints(updated_mean, is_obb_);
    return {updated_mean, updated_covariance};
}

}  // namespace botsort
