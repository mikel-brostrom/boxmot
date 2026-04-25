#include "ocsort/kalman_filter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ocsort {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfPi = kPi / 2.0;

}  // namespace

KalmanFilterXYSR::KalmanFilterXYSR(const int dim_x, const int dim_z, const int max_obs)
    : x(Eigen::VectorXd::Zero(dim_x)),
      P(Eigen::MatrixXd::Identity(dim_x, dim_x)),
      F(Eigen::MatrixXd::Identity(dim_x, dim_x)),
      H(Eigen::MatrixXd::Zero(dim_z, dim_x)),
      Q(Eigen::MatrixXd::Identity(dim_x, dim_x)),
      R(Eigen::MatrixXd::Identity(dim_z, dim_z)),
      dim_x_(dim_x),
      dim_z_(dim_z),
    is_obb_(dim_x == 9 && dim_z == 5),
    max_obs_(std::max(max_obs, 1)) {
    if ((dim_x_ == 7 && dim_z_ == 4) || (dim_x_ == 9 && dim_z_ == 5)) {
        H.block(0, 0, dim_z_, dim_z_).setIdentity();
        if (is_obb_) {
            F(0, 5) = 1.0;
            F(1, 6) = 1.0;
            F(2, 7) = 1.0;
            F(4, 8) = 1.0;
        } else {
            F(0, 4) = 1.0;
            F(1, 5) = 1.0;
            F(2, 6) = 1.0;
        }
        return;
    }
    throw std::invalid_argument("KalmanFilterXYSR expects (dim_x, dim_z) of (7,4) or (9,5).");
}

double KalmanFilterXYSR::WrapAngle(const double angle) {
    const double period = 2.0 * kPi;
    return std::fmod(std::fmod(angle + kPi, period) + period, period) - kPi;
}

void KalmanFilterXYSR::AppendHistory(std::optional<Vector> measurement) {
    if (static_cast<int>(history_obs_.size()) >= max_obs_) {
        history_obs_.pop_front();
    }
    history_obs_.push_back(std::move(measurement));
}

KalmanFilterXYSR::Vector KalmanFilterXYSR::AlignObbMeasurement(const Vector& measurement) const {
    Vector aligned = measurement;
    const double ref_r = std::max(x[3], 1.0e-6);
    const double ref_theta = x[4];
    const double s = std::max(aligned[2], 1.0e-6);
    const double r = std::max(aligned[3], 1.0e-6);
    const double theta = aligned[4];

    const std::array<std::array<double, 3>, 4> candidates = {{
        {s, r, theta},
        {s, r, theta + kPi},
        {s, 1.0 / r, theta + kHalfPi},
        {s, 1.0 / r, theta - kHalfPi},
    }};

    double best_cost = std::numeric_limits<double>::infinity();
    std::array<double, 3> best = candidates.front();
    for (const auto& candidate : candidates) {
        const double theta_aligned = ref_theta + WrapAngle(candidate[2] - ref_theta);
        const double angle_cost = std::abs(theta_aligned - ref_theta);
        const double ratio_cost = std::abs(std::log(std::max(candidate[1], 1.0e-6) / ref_r));
        const double cost = angle_cost + (0.05 * ratio_cost);
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

void KalmanFilterXYSR::EnforceStateConstraints() {
    x[2] = std::max(x[2], 1.0e-6);
    x[3] = std::max(x[3], 1.0e-6);
    if (is_obb_) {
        x[4] = WrapAngle(x[4]);
    }
    P = 0.5 * (P + P.transpose());
}

void KalmanFilterXYSR::Predict() {
    x = F * x;
    P = (F * P * F.transpose()) + Q;
    EnforceStateConstraints();
}

void KalmanFilterXYSR::Freeze() {
    SavedState saved;
    saved.x = x;
    saved.P = P;
    saved.history_obs = history_obs_;
    saved.last_measurement = last_measurement_;
    saved.observed = observed_;
    saved_state_ = std::move(saved);
}

void KalmanFilterXYSR::Unfreeze() {
    if (!saved_state_.has_value()) {
        return;
    }

    const std::deque<std::optional<Vector>> new_history = history_obs_;
    const SavedState saved = *saved_state_;

    x = saved.x;
    P = saved.P;
    history_obs_ = saved.history_obs;
    last_measurement_ = saved.last_measurement;
    observed_ = saved.observed;
    saved_state_.reset();

    if (!history_obs_.empty()) {
        history_obs_.pop_back();
    }

    std::vector<int> observed_indices;
    observed_indices.reserve(new_history.size());
    for (int index = 0; index < static_cast<int>(new_history.size()); ++index) {
        if (new_history[static_cast<std::size_t>(index)].has_value()) {
            observed_indices.push_back(index);
        }
    }
    if (observed_indices.size() < 2) {
        return;
    }

    const int index1 = observed_indices[observed_indices.size() - 2];
    const int index2 = observed_indices[observed_indices.size() - 1];
    const Vector& box1 = *new_history[static_cast<std::size_t>(index1)];
    const Vector& box2 = *new_history[static_cast<std::size_t>(index2)];
    if (box1.size() < dim_z_ || box2.size() < dim_z_) {
        return;
    }

    const double x1 = box1[0];
    const double y1 = box1[1];
    const double s1 = box1[2];
    const double r1 = box1[3];
    const double w1 = std::sqrt(std::max(s1 * r1, 1.0e-12));
    const double h1 = std::sqrt(std::max(s1 / std::max(r1, 1.0e-6), 1.0e-12));

    const double x2 = box2[0];
    const double y2 = box2[1];
    const double s2 = box2[2];
    const double r2 = box2[3];
    const double w2 = std::sqrt(std::max(s2 * r2, 1.0e-12));
    const double h2 = std::sqrt(std::max(s2 / std::max(r2, 1.0e-6), 1.0e-12));

    const int time_gap = index2 - index1;
    if (time_gap <= 0) {
        return;
    }

    const double dx = (x2 - x1) / static_cast<double>(time_gap);
    const double dy = (y2 - y1) / static_cast<double>(time_gap);
    const double dw = (w2 - w1) / static_cast<double>(time_gap);
    const double dh = (h2 - h1) / static_cast<double>(time_gap);
    double t1 = 0.0;
    double dtheta = 0.0;
    if (is_obb_) {
        t1 = box1[4];
        dtheta = WrapAngle(box2[4] - t1) / static_cast<double>(time_gap);
    }

    for (int index = 0; index < time_gap; ++index) {
        const double interp_x = x1 + static_cast<double>(index + 1) * dx;
        const double interp_y = y1 + static_cast<double>(index + 1) * dy;
        const double interp_w = w1 + static_cast<double>(index + 1) * dw;
        const double interp_h = h1 + static_cast<double>(index + 1) * dh;
        const double interp_s = interp_w * interp_h;
        const double interp_r = interp_w / std::max(interp_h, 1.0e-6);

        Vector interpolated(dim_z_);
        interpolated << interp_x, interp_y, interp_s, interp_r;
        if (is_obb_) {
            interpolated.conservativeResize(5);
            interpolated[4] = WrapAngle(t1 + static_cast<double>(index + 1) * dtheta);
        }

        Update(interpolated);
        if (index != (time_gap - 1)) {
            Predict();
            if (!history_obs_.empty()) {
                history_obs_.pop_back();
            }
        }
    }

    if (!history_obs_.empty()) {
        history_obs_.pop_back();
    }
}

void KalmanFilterXYSR::Update(const Vector& measurement) {
    if (measurement.size() != dim_z_) {
        throw std::runtime_error("KalmanFilterXYSR update measurement size mismatch.");
    }

    Vector aligned = measurement;
    aligned[2] = std::max(aligned[2], 1.0e-6);
    aligned[3] = std::max(aligned[3], 1.0e-6);
    if (is_obb_) {
        aligned = AlignObbMeasurement(aligned);
    }

    AppendHistory(aligned);
    if (!observed_) {
        Unfreeze();
    }
    observed_ = true;

    const Matrix innovation_covariance = (H * P * H.transpose()) + R;
    Eigen::LLT<Matrix> llt(innovation_covariance);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("Kalman update failed: innovation covariance is not positive definite.");
    }

    const Matrix innovation_covariance_inv = llt.solve(Matrix::Identity(dim_z_, dim_z_));
    const Matrix kalman_gain = P * H.transpose() * innovation_covariance_inv;
    const Vector innovation = aligned - (H * x);

    x = x + (kalman_gain * innovation);
    P = P - (kalman_gain * innovation_covariance * kalman_gain.transpose());
    if (is_obb_ && dim_x_ >= 9) {
        x[8] *= 0.8;
    }
    EnforceStateConstraints();
    AppendHistory(aligned);
}

void KalmanFilterXYSR::UpdateMissing() {
    AppendHistory(std::nullopt);
    if (observed_ && history_obs_.size() >= 2) {
        const auto& previous = history_obs_[history_obs_.size() - 2];
        if (previous.has_value()) {
            last_measurement_ = previous;
        }
        Freeze();
    }
    observed_ = false;
}

}  // namespace ocsort